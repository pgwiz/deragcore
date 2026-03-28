"""Research endpoints - POST /research, WS /research/stream."""

import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Query
from pydantic import BaseModel

from ragcore.config import settings
from ragcore.db.database import get_db_session
from ragcore.models import Session as SessionModel, ModelConfig
from ragcore.core.ai_controller import AIController
from ragcore.core.schemas import ORION_RESEARCH_PROMPT
from ragcore.core.websocket_manager import websocket_manager
from ragcore.modules.research.pipeline import pipeline
from ragcore.modules.research.models import ResearchSessionState
from sqlalchemy import select

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/research", tags=["research"])


# ============================================================================
# Request/Response Models
# ============================================================================


class ResearchRequest(BaseModel):
    """Research query request."""

    query: str
    """Research query"""

    session_id: Optional[UUID] = None
    """Session ID for history context"""

    file_ids: Optional[List[UUID]] = None
    """Optional: scope search to these files (compound mode)"""

    research_mode: str = "standard"
    """'standard' (web) or 'deep' (multi-source)"""

    model_config_name: Optional[str] = None
    """Optional: model config for synthesis"""


class SourceAttribution(BaseModel):
    """Source reference."""

    title: str
    url: str
    snippet: str
    source: str  # "tavily" | "serpapi" | "duckduckgo" | "gpt-researcher"


class ResearchResponse(BaseModel):
    """Research completion response."""

    response: str
    """Final synthesized response"""

    sources: List[SourceAttribution]
    """Source references"""

    session_id: UUID
    """Session identifier"""

    turns_executed: int
    """Number of research turns completed"""

    tool_calls_total: int
    """Total tool executions"""


# ============================================================================
# HTTP Endpoints
# ============================================================================


@router.post("/", response_model=ResearchResponse)
async def research_query(req: ResearchRequest) -> ResearchResponse:
    """
    Single research query with multi-turn agent reasoning.

    Flow:
    1. Initialize research session
    2. Loop: Plan → Search → Synthesize (max 3 turns)
    3. Agent decides when to finalize
    4. Return synthesized response with sources

    Args:
        query: Research question or topic
        session_id: Optional session for history
        file_ids: Optional files for compound mode
        research_mode: "standard" or "deep"

    Returns:
        ResearchResponse with synthesis + sources
    """
    logger.info(f"Research query: '{req.query[:60]}' mode={req.research_mode}")

    try:
        # Use provided session or create new
        session_id = req.session_id or UUID(int=0)  # Will be replaced in real usage

        # Execute research pipeline
        final_response, sources, session_state = await pipeline.research(
            query=req.query,
            session_id=session_id,
            research_mode=req.research_mode,
            file_ids=req.file_ids,
        )

        # Format sources
        formatted_sources = [
            SourceAttribution(
                title=s.get("title", ""),
                url=s.get("url", ""),
                snippet=s.get("snippet", ""),
                source=s.get("source", "unknown"),
            )
            for s in sources
        ]

        return ResearchResponse(
            response=final_response,
            sources=formatted_sources,
            session_id=session_id,
            turns_executed=session_state.current_turn,
            tool_calls_total=session_state.total_tool_calls,
        )

    except Exception as e:
        logger.error(f"Research query error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Research failed: {str(e)}",
        )


@router.get("/session/{session_id}")
async def get_research_session(session_id: UUID) -> dict:
    """
    Get research findings for a session.

    Args:
        session_id: Session identifier

    Returns:
        Research session state with findings and decisions
    """
    try:
        async with get_db_session() as session:
            stmt = select(SessionModel).where(SessionModel.id == session_id)
            result = await session.execute(stmt)
            session_record = result.scalar_one_or_none()

            if not session_record:
                raise HTTPException(
                    status_code=404,
                    detail=f"Session {session_id} not found",
                )

            # Return session info
            # Note: Full research state would be in extended schema
            return {
                "session_id": str(session_id),
                "created_at": session_record.created_at.isoformat(),
                "title": session_record.title,
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get session error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get session: {str(e)}",
        )


# ============================================================================
# WebSocket Streaming Endpoint
# ============================================================================


@router.websocket("/stream")
async def research_stream(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for streaming research with real-time tool execution.

    Flow:
    1. Accept WebSocket
    2. Receive query + options
    3. Execute research with real-time updates:
       - tool_call: {"type": "tool_call", "tool": "...", "query": "..."}
       - tool_result: {"type": "tool_result", "results_count": N}
       - finding: {"type": "finding", "synthesis": "..."}
       - token: {"type": "token", "delta": "..."}
    4. Send sources on completion
    5. Send done marker

    Expected client message:
    {
        "query": "Research the latest AI trends",
        "file_ids": ["uuid1", "uuid2"],  # Optional
        "research_mode": "standard"  # Optional
    }

    Sent to client:
    { "type": "tool_call", "tool": "tavily", "query": "..." }
    { "type": "tool_result", "count": 5 }
    { "type": "finding", "synthesis": "..." }
    { "type": "token", "delta": "Hello" }
    { "type": "sources", "sources": [...] }
    { "type": "done" }
    """
    session_id_str = str(session_id)

    await websocket_manager.connect(session_id_str, websocket)
    logger.info(f"Research stream connected: session={session_id_str}")

    try:
        # Receive query
        data = await websocket.receive_json()
        query = data.get("query", "")
        file_ids_raw = data.get("file_ids", [])
        research_mode = data.get("research_mode", "standard")

        if not query:
            await websocket.send_json({
                "type": "error",
                "message": "Missing 'query' field",
            })
            return

        # Convert file_ids to UUIDs
        file_ids = []
        try:
            file_ids = [UUID(fid) if isinstance(fid, str) else fid for fid in file_ids_raw]
        except Exception as e:
            logger.warning(f"Invalid file_ids: {str(e)}")

        logger.info(f"Research stream: query='{query[:50]}' mode={research_mode}")

        # =====================================================================
        # Stream Research Execution
        # =====================================================================

        final_response, sources, session_state = await pipeline.research(
            query=query,
            session_id=UUID(int=0),  # Real session would be tracked
            research_mode=research_mode,
            file_ids=file_ids if file_ids else None,
        )

        # Stream tool calls from session
        for turn in session_state.turns:
            # Tool calls
            for tool_call in turn.tool_calls:
                await websocket_manager.send_to_session(
                    session_id_str,
                    "tool_call",
                    {
                        "tool": tool_call.tool_name,
                        "query": tool_call.query,
                        "status": tool_call.status,
                    },
                )

            # Tool results
            for tool_result in turn.tool_results:
                await websocket_manager.send_to_session(
                    session_id_str,
                    "tool_result",
                    {
                        "count": len(tool_result.sources),
                        "sources": tool_result.sources,
                    },
                )

            # Findings
            for finding in turn.research_findings:
                await websocket_manager.send_to_session(
                    session_id_str,
                    "finding",
                    {
                        "query": finding.query,
                        "synthesis": finding.synthesis,
                        "tool": finding.tool_used,
                    },
                )

        # =====================================================================
        # Stream Final Response Synthesis
        # =====================================================================

        # Stream synthesis tokens (simulated - real would be from streaming provider)
        for i in range(0, len(final_response), 50):
            chunk = final_response[i : i + 50]
            await websocket_manager.send_to_session(
                session_id_str,
                "token",
                {"delta": chunk},
            )

        # =====================================================================
        # Send Sources & Done
        # =====================================================================

        await websocket.send_json({
            "type": "sources",
            "sources": [
                {
                    "title": s.get("title", ""),
                    "url": s.get("url", ""),
                    "snippet": s.get("snippet", ""),
                    "source": s.get("source", ""),
                }
                for s in sources
            ],
        })

        await websocket.send_json({
            "type": "done",
            "turns": session_state.current_turn,
            "tool_calls": session_state.total_tool_calls,
        })

        logger.info(f"Research stream completed: session={session_id_str}")

    except WebSocketDisconnect:
        logger.info(f"Research stream disconnected: session={session_id_str}")
    except Exception as e:
        logger.error(f"Research stream error: {str(e)}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "message": str(e),
            })
        except Exception:
            pass


@router.get("/health")
async def research_health():
    """Health check for research module."""
    from ragcore.modules.research.tool_registry import executor

    tools_status = executor.get_tool_status()

    return {
        "status": "ok",
        "module": "research",
        "tools": tools_status,
        "max_turns": settings.research_max_turns,
    }
