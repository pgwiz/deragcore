"""Chat endpoints - RAG completion with context retrieval."""

import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from sqlalchemy import select

from ragcore.config import settings
from ragcore.db.database import get_db_session
from ragcore.models import ModelConfig, Session as SessionModel
from ragcore.core.ai_controller import AIController
from ragcore.core.schemas import ORION_DEFAULT_PROMPT
from ragcore.core.websocket_manager import websocket_manager
from ragcore.modules.chat.retriever import VectorRetriever, RetrievedChunk
from ragcore.modules.chat.context_builder import ContextBuilder
from ragcore.modules.chat.history import ChatHistoryManager, ChatTurn
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


# ============================================================================
# Response Models
# ============================================================================


class SourceAttribution(BaseModel):
    """Source chunk attribution in response."""

    chunk_id: UUID
    file_id: UUID
    filename: str
    similarity_score: float
    excerpt: str


class ChatCompleteRequest(BaseModel):
    """Chat completion request."""

    message: str
    session_id: Optional[UUID] = None
    model_config_name: Optional[str] = None
    file_ids: Optional[List[UUID]] = None  # Scope retrieval to these files
    top_k: int = 5
    enable_research: bool = False  # NEW: Enable web search for compound mode


class ChatCompleteResponse(BaseModel):
    """Chat completion response."""

    text: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    sources: List[SourceAttribution]


# ============================================================================
# Endpoints
# ============================================================================


@router.post("/complete", response_model=ChatCompleteResponse)
async def chat_complete(req: ChatCompleteRequest) -> ChatCompleteResponse:
    """
    Single RAG completion with context retrieval.

    Flow:
    1. Retrieve relevant chunks for the query
    2. Build context from chunks + conversation history
    3. Generate completion using AI provider
    4. Return response with source attribution

    Args:
        message: User query
        session_id: Optional session (for history)
        model_config_name: Optional model preset
        file_ids: Optional file scope for retrieval
        top_k: Number of chunks to retrieve

    Returns:
        ChatCompleteResponse with text and sources

    Raises:
        404: Session not found
        422: No chunks retrieved or embedding failed
    """
    logger.info(f"Chat complete: query='{req.message[:50]}...', session={req.session_id}")

    try:
        # =====================================================================
        # Resolve Model Config
        # =====================================================================
        provider_name = None
        model_id = None
        system_prompt = ORION_DEFAULT_PROMPT

        if req.model_config_name:
            async with get_db_session() as session:
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

        if not provider_name:
            # Use default provider
            providers = AIController.get_available_providers()
            provider_name = list(providers.keys())[0] if providers else "anthropic"
            model_map = {
                "anthropic": "claude-3-5-sonnet-20241022",
                "azure": "phi-4",
                "openai": "gpt-4o",
                "ollama": "llama2",
            }
            model_id = model_map.get(provider_name, "default")

        # =====================================================================
        # Retrieve Context
        # =====================================================================
        retriever = VectorRetriever(
            embedding_provider=settings.embedding_provider,
            embedding_model=settings.embedding_model,
        )

        try:
            retrieved_chunks = await retriever.retrieve(
                query=req.message,
                file_ids=req.file_ids,
                top_k=req.top_k,
            )
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
        except RuntimeError as e:
            logger.error(f"Retrieval failed: {str(e)}")
            raise HTTPException(
                status_code=422,
                detail=f"Failed to retrieve context: {str(e)}",
            )

        if not retrieved_chunks:
            logger.warning("No chunks retrieved for query")

        # =====================================================================
        # Research Phase (Optional - Compound Mode)
        # =====================================================================
        research_findings = None
        research_sources = []

        if req.enable_research:
            logger.info("Compound mode: Executing research phase")
            try:
                from ragcore.modules.research.pipeline import pipeline as research_pipeline
                research_response, research_sources, session_state = await research_pipeline.research(
                    query=req.message,
                    session_id=req.session_id or UUID(int=0),
                    research_mode="standard",
                    file_ids=req.file_ids,
                )
                research_findings = research_response
                logger.info(f"Research phase: {len(research_sources)} sources")
            except Exception as e:
                logger.warning(f"Research phase failed: {str(e)}")
                # Continue without research
                research_findings = None
                research_sources = []
        history_manager = ChatHistoryManager()

        history = []
        if req.session_id:
            try:
                history = await history_manager.get_recent(req.session_id, limit=10)
            except Exception as e:
                logger.warning(f"Failed to load history: {str(e)}")
                history = []

        # Build context (use compound mode if research enabled)
        if req.enable_research and research_findings:
            # Use compound mode context builder with mixed sources
            system_prompt_for_context = "You are Orion in compound intelligence mode. You have access to both uploaded documents and web research findings. Clearly distinguish between document-sourced answers [DOC:] and web-sourced answers [WEB:]."
            messages = ContextBuilder.build_compound(
                system_prompt=system_prompt_for_context,
                query=req.message,
                retrieved_chunks=retrieved_chunks,
                research_findings=research_findings,
                research_sources=research_sources,
                history=history,
            )
            logger.info("Built compound context with doc + web sources")
        else:
            # Normal RAG mode
            messages = ContextBuilder.build(
                system_prompt=system_prompt,
                query=req.message,
                retrieved_chunks=retrieved_chunks,
                history=history,
            )
            logger.debug(f"Built context: {len(messages)} messages")

        # =====================================================================
        # Generate Completion
        # =====================================================================
        try:
            response = AIController.complete(
                provider_name=provider_name,
                model_id=model_id,
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
                system_prompt=system_prompt,
            )
            logger.info(
                f"Completion generated: {response.output_tokens} output tokens"
            )
        except Exception as e:
            logger.error(f"Completion failed: {str(e)}")
            raise HTTPException(
                status_code=422,
                detail=f"Failed to generate completion: {str(e)}",
            )

        # =====================================================================
        # Format Response with Sources
        # =====================================================================
        sources = [
            SourceAttribution(
                chunk_id=chunk.chunk_id,
                file_id=chunk.file_id,
                filename=chunk.filename,
                similarity_score=chunk.similarity_score,
                excerpt=chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text,
            )
            for chunk in retrieved_chunks
        ]

        # Add research sources if available (compound mode)
        if req.enable_research and research_sources:
            for src in research_sources[:5]:  # Limit to top 5 web sources
                sources.append(
                    SourceAttribution(
                        chunk_id=UUID(int=0),  # No chunk for web sources
                        file_id=UUID(int=0),
                        filename=f"[WEB] {src.get('title', 'Source')}",
                        similarity_score=0.5,  # Default score for web sources
                        excerpt=src.get("snippet", "")[:100],
                    )
                )

        return ChatCompleteResponse(
            text=response.text,
            model=response.model,
            provider=response.provider,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            sources=sources,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat complete error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}",
        )


@router.get("/health")
async def chat_health():
    """Health check for chat module."""
    return {
        "status": "ok",
        "module": "chat",
        "features": ["retrieval", "history", "streaming"],
    }


@router.websocket("/stream")
async def chat_stream(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for streaming RAG completion.

    Flow:
    1. Accept WebSocket and register with manager
    2. Receive message with query and optional file_ids
    3. Retrieve context chunks
    4. Stream tokens in real-time
    5. Send sources on completion

    Expected client message:
    {
        "message": "What is the main topic?",
        "file_ids": ["uuid1", "uuid2"],  # Optional
        "model_config_name": "deep-analysis"  # Optional
    }

    Sent to client:
    { "type": "token", "delta": "Hello" }
    { "type": "token", "delta": " world" }
    { "type": "sources", "sources": [...] }
    { "type": "done" }
    """
    session_id_str = str(session_id)

    await websocket_manager.connect(session_id_str, websocket)
    logger.info(f"Chat stream connected: session={session_id_str}")

    try:
        # Receive initial message with query
        data = await websocket.receive_json()
        query = data.get("message", "")
        file_ids_raw = data.get("file_ids", [])
        model_config_name = data.get("model_config_name")

        if not query:
            await websocket.send_json({
                "type": "error",
                "message": "Missing 'message' field"
            })
            return

        # Convert file_ids to UUIDs
        file_ids = [UUID(fid) if isinstance(fid, str) else fid for fid in file_ids_raw]

        logger.debug(
            f"Stream start: query='{query[:50]}...', "
            f"files={len(file_ids)}, model={model_config_name}"
        )

        # =====================================================================
        # Resolve Model Config
        # =====================================================================
        provider_name = None
        model_id = None
        system_prompt = ORION_DEFAULT_PROMPT

        if model_config_name:
            async with get_db_session() as session:
                stmt = select(ModelConfig).where(
                    ModelConfig.name == model_config_name
                )
                result = await session.execute(stmt)
                config = result.scalar_one_or_none()

                if config:
                    provider_name = config.provider
                    model_id = config.model_id
                    system_prompt = config.system_prompt or ORION_DEFAULT_PROMPT

        if not provider_name:
            providers = AIController.get_available_providers()
            provider_name = list(providers.keys())[0] if providers else "anthropic"
            model_map = {
                "anthropic": "claude-3-5-sonnet-20241022",
                "azure": "phi-4",
                "openai": "gpt-4o",
                "ollama": "llama2",
            }
            model_id = model_map.get(provider_name, "default")

        # =====================================================================
        # Retrieve Context
        # =====================================================================
        retriever = VectorRetriever()

        try:
            retrieved_chunks = await retriever.retrieve(
                query=query,
                file_ids=file_ids if file_ids else None,
                top_k=5,
            )
        except RuntimeError as e:
            logger.error(f"Retrieval failed: {str(e)}")
            await websocket.send_json({
                "type": "error",
                "message": f"Retrieval failed: {str(e)}"
            })
            return

        # =====================================================================
        # Build Context
        # =====================================================================
        history_manager = ChatHistoryManager()
        history = []

        try:
            session_uuid = UUID(session_id_str)
            history = await history_manager.get_recent(session_uuid, limit=10)
        except Exception as e:
            logger.warning(f"Failed to load history: {str(e)}")

        messages = ContextBuilder.build(
            system_prompt=system_prompt,
            query=query,
            retrieved_chunks=retrieved_chunks,
            history=history,
        )

        # =====================================================================
        # Stream Completion
        # =====================================================================
        logger.debug(f"Streaming from {provider_name}/{model_id}")

        try:
            async for chunk in AIController.stream(
                provider_name=provider_name,
                model_id=model_id,
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
                system_prompt=system_prompt,
            ):
                # Send token to all connections in session
                await websocket_manager.send_to_session(
                    session_id_str,
                    "token",
                    {"delta": chunk.delta, "provider": chunk.provider}
                )

        except Exception as e:
            logger.error(f"Streaming error: {str(e)}", exc_info=True)
            await websocket.send_json({
                "type": "error",
                "message": f"Streaming error: {str(e)}"
            })
            return

        # =====================================================================
        # Send Sources
        # =====================================================================
        sources = ContextBuilder.format_sources(retrieved_chunks)
        await websocket.send_json({
            "type": "sources",
            "sources": sources
        })

        # Send completion marker
        await websocket.send_json({"type": "done"})
        logger.info(f"Stream completed for session {session_id_str}")

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: session={session_id_str}")
    except Exception as e:
        logger.error(f"Stream error: {str(e)}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e)
            })
        except Exception:
            pass
