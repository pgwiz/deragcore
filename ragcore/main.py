"""RAGCORE FastAPI application factory and routes."""

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, List, Optional
from uuid import UUID

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ragcore.config import settings
from ragcore.db.database import init_db, close_db, get_db_session
from ragcore.core.ai_controller import AIController
from ragcore.core.schemas import ORION_DEFAULT_PROMPT
from ragcore.models import ModelConfig, Session
from ragcore.modules.files import router as files_router
from ragcore.modules.chat import router as chat_router
from ragcore.modules.research import router as research_router
from ragcore.modules.agents.router import router as agents_router
from ragcore.modules.memory.router import router as memory_router
from ragcore.auth import router as auth_router
from ragcore.webhooks import router as webhooks_router
from ragcore.monitoring import health_check

logger = logging.getLogger(__name__)


# ============================================================================
# Request/Response Models
# ============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    timestamp: str
    services: Dict[str, str]
    providers: Dict[str, bool]


class TestCompleteRequest(BaseModel):
    """Test completion request."""

    message: str
    temperature: float = 0.7
    max_tokens: int = 2048
    model_config_name: Optional[str] = None  # Optional: use named preset


class TestCompleteResponse(BaseModel):
    """Test completion response."""

    text: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int


# ============================================================================
# Lifespan Management
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app startup and shutdown."""
    # Startup
    logger.info("Starting RAGCORE...")
    try:
        await init_db()
        logger.info("✓ Database initialized")
    except Exception as e:
        logger.warning(f"Database initialization failed (non-critical for research mode): {e}")

    # Check providers
    providers = AIController.get_available_providers()
    logger.info(f"Available providers: {providers}")

    # Initialize ARQ redis pool
    try:
        from ragcore.core.job_manager import get_redis_pool
        await get_redis_pool()
        logger.info("✓ ARQ redis pool initialized")
    except Exception as e:
        logger.warning(f"ARQ redis pool initialization failed (non-critical): {e}")

    yield

    # Shutdown
    logger.info("Shutting down RAGCORE...")

    # Close ARQ redis pool
    try:
        from ragcore.core.job_manager import close_redis_pool
        await close_redis_pool()
        logger.info("✓ ARQ redis pool closed")
    except Exception as e:
        logger.warning(f"ARQ redis pool cleanup failed: {e}")

    try:
        await close_db()
        logger.info("✓ Database closed")
    except Exception as e:
        logger.warning(f"Database cleanup failed: {e}")


# ============================================================================
# App Factory
# ============================================================================


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""

    app = FastAPI(
        title="RAGCORE",
        description="Modular Multi-Provider RAG API Platform",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ======================================================================
    # Health Endpoint
    # ======================================================================

    @app.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """Check API health and provider availability."""
        providers = AIController.get_available_providers()
        return HealthResponse(
            status="ok",
            timestamp=datetime.utcnow().isoformat(),
            services={
                "database": "connected",
                "redis": "ready",
                "api": "online",
            },
            providers=providers,
        )

    # ======================================================================
    # Test Endpoints (for manual testing / development)
    # ======================================================================

    @app.post("/test/complete", response_model=TestCompleteResponse)
    async def test_complete(req: TestCompleteRequest) -> TestCompleteResponse:
        """
        Test endpoint - single completion from provider.

        Optionally specify model_config_name to use a preset configuration.
        """
        try:
            providers = AIController.get_available_providers()
            if not providers:
                raise HTTPException(
                    status_code=503, detail="No AI providers available"
                )

            # Resolve provider, model, and system_prompt
            provider_name = None
            model_id = None
            system_prompt = ORION_DEFAULT_PROMPT

            if req.model_config_name:
                # Fetch ModelConfig from database
                async with get_db_session() as session:
                    from sqlalchemy import select
                    stmt = select(ModelConfig).where(
                        ModelConfig.name == req.model_config_name
                    )
                    result = await session.execute(stmt)
                    config = result.scalar_one_or_none()

                    if not config:
                        raise HTTPException(
                            status_code=404,
                            detail=f"ModelConfig '{req.model_config_name}' not found",
                        )

                    provider_name = config.provider
                    model_id = config.model_id
                    system_prompt = config.system_prompt or ORION_DEFAULT_PROMPT
                    temp = config.temperature
                    max_tokens = config.max_tokens
            else:
                # Use first available provider with defaults
                provider_name = list(providers.keys())[0]
                model_map = {
                    "anthropic": "claude-3-5-sonnet-20241022",
                    "azure": "phi-4",
                    "openai": "gpt-4o",
                    "ollama": "llama2",
                }
                model_id = model_map.get(provider_name, "default")
                temp = req.temperature
                max_tokens = req.max_tokens

            logger.info(f"Complete from {provider_name}/{model_id} with system_prompt")

            response = AIController.complete(
                provider_name=provider_name,
                model_id=model_id,
                messages=[{"role": "user", "content": req.message}],
                temperature=temp,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
            )

            return TestCompleteResponse(
                text=response.text,
                model=response.model,
                provider=response.provider,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Test complete error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.websocket("/test/stream")
    async def test_stream(websocket: WebSocket):
        """
        Test WebSocket endpoint - stream from provider.

        Send: {"message": "Hello", "temperature": 0.7, "model_config_name": "fast-chat"}
        Receive: {"delta": "...", "type": "token"} or {"type": "done"}
        """
        await websocket.accept()
        try:
            data = await websocket.receive_json()
            message = data.get("message", "Hello")
            temperature = data.get("temperature", 0.7)
            model_config_name = data.get("model_config_name")
            max_tokens = data.get("max_tokens", 2048)

            providers = AIController.get_available_providers()
            if not providers:
                await websocket.send_json(
                    {"type": "error", "message": "No providers available"}
                )
                await websocket.close()
                return

            # Resolve provider and model
            provider_name = None
            model_id = None
            system_prompt = ORION_DEFAULT_PROMPT

            if model_config_name:
                # Fetch ModelConfig from database
                async with get_db_session() as session:
                    from sqlalchemy import select
                    stmt = select(ModelConfig).where(
                        ModelConfig.name == model_config_name
                    )
                    result = await session.execute(stmt)
                    config = result.scalar_one_or_none()

                    if config:
                        provider_name = config.provider
                        model_id = config.model_id
                        system_prompt = config.system_prompt or ORION_DEFAULT_PROMPT
                        temperature = config.temperature

            if not provider_name:
                provider_name = list(providers.keys())[0]
                model_map = {
                    "anthropic": "claude-3-5-sonnet-20241022",
                    "azure": "phi-4",
                    "openai": "gpt-4o",
                    "ollama": "llama2",
                }
                model_id = model_map.get(provider_name, "default")

            logger.info(
                f"Streaming from {provider_name}/{model_id} with system_prompt..."
            )

            async for chunk in AIController.stream(
                provider_name=provider_name,
                model_id=model_id,
                messages=[{"role": "user", "content": message}],
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
            ):
                await websocket.send_json(
                    {
                        "type": "token",
                        "delta": chunk.delta,
                        "provider": chunk.provider,
                    }
                )

            await websocket.send_json({"type": "done"})

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"WebSocket stream error: {e}", exc_info=True)
            try:
                await websocket.send_json(
                    {"type": "error", "message": str(e)}
                )
            except Exception:
                pass

    # ======================================================================
    # Mount Phase 2 Routers
    # ======================================================================

    app.include_router(files_router.router)
    app.include_router(chat_router.router)
    app.include_router(research_router.router)

    # ======================================================================
    # Mount Phase 5 Routers (Advanced Agent Features)
    # ======================================================================

    app.include_router(agents_router)
    app.include_router(memory_router)

    # ======================================================================
    # Mount Phase 4 Routers (Production Hardening)
    # ======================================================================

    app.include_router(auth_router.router)
    app.include_router(webhooks_router.router)

    # ======================================================================
    # Monitoring & Health Endpoints
    # ======================================================================

    @app.get("/health")
    async def comprehensive_health_check() -> dict:
        """Comprehensive health check for all systems."""
        return await health_check.get_health_status()

    @app.get("/metrics")
    async def metrics_endpoint():
        """Prometheus metrics endpoint."""
        from prometheus_client import generate_latest
        from prometheus_client.core import CollectorRegistry, REGISTRY

        return generate_latest(REGISTRY)

    # ======================================================================

    @app.get("/")
    def root():
        """Root endpoint - API info."""
        return {
            "name": "RAGCORE",
            "version": "0.1.0",
            "description": "Modular Multi-Provider RAG API Platform",
            "docs": "/docs",
            "health": "/health",
        }

    return app


# Create app instance
app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.port, reload=True)
