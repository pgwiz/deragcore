"""HTTP router for memory operations."""

import logging
from typing import Optional, List
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel

from ragcore.auth.dependencies import get_current_api_key_id
from ragcore.modules.memory.long_term import memory_store
from ragcore.modules.memory.episodic import episodic_memory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/memory", tags=["memory"])


# Request/Response Models
class SaveMemoryRequest(BaseModel):
    """Request to save a memory."""

    session_id: UUID
    memory_type: str  # "finding" | "decision" | "insight" | "error"
    content: str
    summary: Optional[str] = None
    source: Optional[str] = None
    tags: Optional[List[str]] = None
    importance_score: float = 0.5
    context_data: Optional[dict] = None


class MemoryResponse(BaseModel):
    """Memory response."""

    id: UUID
    memory_type: str
    content: str
    summary: Optional[str]
    source: Optional[str]
    tags: List[str]
    importance_score: float
    access_count: int
    created_at: str
    last_accessed_at: str

    class Config:
        from_attributes = True


class SaveEpisodeRequest(BaseModel):
    """Request to save an episode."""

    session_id: UUID
    episode_type: str  # "research" | "chat" | "analysis"
    title: str
    description: str
    input_query: str
    output_summary: Optional[str] = None
    key_findings: Optional[List[dict]] = None
    sources_used: Optional[List[str]] = None
    tools_invoked: Optional[List[dict]] = None
    tags: Optional[List[str]] = None
    tokens_used: int = 0
    success: bool = True


class EpisodeResponse(BaseModel):
    """Episode response."""

    id: UUID
    session_id: UUID
    episode_number: int
    episode_type: str
    title: str
    description: str
    input_query: str
    output_summary: Optional[str]
    key_findings: List[dict]
    sources_used: List[str]
    tags: List[str]
    tokens_used: int
    success: bool
    created_at: str

    class Config:
        from_attributes = True


# Endpoints - Long-Term Memory
@router.post("/memories", summary="Save a memory")
async def save_memory(
    request: SaveMemoryRequest,
    api_key_id: UUID = Depends(get_current_api_key_id),
):
    """Save a long-term memory entry."""
    try:
        memory_id = await memory_store.save_memory(
            session_id=request.session_id,
            memory_type=request.memory_type,
            content=request.content,
            summary=request.summary,
            source=request.source,
            tags=request.tags,
            importance_score=request.importance_score,
            context_data=request.context_data,
        )

        if not memory_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save memory",
            )

        return {"id": str(memory_id), "message": "Memory saved successfully"}

    except Exception as e:
        logger.error(f"Error saving memory: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save memory",
        )


@router.get("/memories/{memory_id}", summary="Get a memory")
async def get_memory(
    memory_id: UUID,
    api_key_id: UUID = Depends(get_current_api_key_id),
):
    """Get a specific memory entry."""
    try:
        memory = await memory_store.get_memory(memory_id)

        if not memory:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Memory {memory_id} not found",
            )

        # Log access
        await memory_store.log_access(
            memory_id=memory_id,
            session_id=memory.session_id,
            access_type="retrieval",
            user_id=memory.user_id,
        )

        return MemoryResponse.from_orm(memory)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting memory: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve memory",
        )


@router.get("/sessions/{session_id}/memories", summary="Search session memories")
async def search_memories(
    session_id: UUID,
    query: Optional[str] = None,
    memory_type: Optional[str] = None,
    tags: Optional[str] = None,
    limit: int = 10,
    api_key_id: UUID = Depends(get_current_api_key_id),
):
    """Search memories in a session."""
    try:
        # Parse tags if provided (comma-separated)
        tag_list = tags.split(",") if tags else None

        memories = await memory_store.search_session_memory(
            session_id=session_id,
            query=query,
            memory_type=memory_type,
            tags=tag_list,
            limit=limit,
        )

        return {
            "total": len(memories),
            "memories": [MemoryResponse.from_orm(m).dict() for m in memories],
        }

    except Exception as e:
        logger.error(f"Error searching memories: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search memories",
        )


@router.delete("/memories/{memory_id}", summary="Delete a memory")
async def delete_memory(
    memory_id: UUID,
    api_key_id: UUID = Depends(get_current_api_key_id),
):
    """Delete/deactivate a memory."""
    try:
        success = await memory_store.delete_memory(memory_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Memory {memory_id} not found",
            )

        return {"message": "Memory deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting memory: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete memory",
        )


# Endpoints - Episodic Memory
@router.post("/episodes", summary="Save an episode")
async def save_episode(
    request: SaveEpisodeRequest,
    api_key_id: UUID = Depends(get_current_api_key_id),
):
    """Save an episodic memory snapshot."""
    try:
        episode_id = await episodic_memory.save_episode(
            session_id=request.session_id,
            episode_type=request.episode_type,
            title=request.title,
            description=request.description,
            input_query=request.input_query,
            output_summary=request.output_summary,
            key_findings=request.key_findings,
            sources_used=request.sources_used,
            tools_invoked=request.tools_invoked,
            tags=request.tags,
            tokens_used=request.tokens_used,
            success=request.success,
        )

        if not episode_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save episode",
            )

        return {"id": str(episode_id), "message": "Episode saved successfully"}

    except Exception as e:
        logger.error(f"Error saving episode: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save episode",
        )


@router.get("/episodes/{episode_id}", summary="Get an episode")
async def get_episode(
    episode_id: UUID,
    api_key_id: UUID = Depends(get_current_api_key_id),
):
    """Get an episodic memory snapshot."""
    try:
        episode = await episodic_memory.get_episode(episode_id)

        if not episode:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Episode {episode_id} not found",
            )

        return EpisodeResponse.from_orm(episode)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting episode: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve episode",
        )


@router.get("/sessions/{session_id}/episodes", summary="Get session episodes")
async def get_session_episodes(
    session_id: UUID,
    episode_type: Optional[str] = None,
    limit: int = 20,
    api_key_id: UUID = Depends(get_current_api_key_id),
):
    """Get all episodes in a session."""
    try:
        episodes = await episodic_memory.get_session_episodes(
            session_id=session_id,
            episode_type=episode_type,
            limit=limit,
        )

        return {
            "total": len(episodes),
            "episodes": [EpisodeResponse.from_orm(e).dict() for e in episodes],
        }

    except Exception as e:
        logger.error(f"Error retrieving episodes: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve episodes",
        )


@router.get("/sessions/{session_id}/summary", summary="Get session memory summary")
async def get_session_summary(
    session_id: UUID,
    api_key_id: UUID = Depends(get_current_api_key_id),
):
    """Get a summary of recent episodes and memories."""
    try:
        summary = await episodic_memory.get_episode_summary(
            session_id=session_id,
            num_recent=5,
        )

        return {"session_id": str(session_id), "summary": summary}

    except Exception as e:
        logger.error(f"Error getting session summary: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get summary",
        )


# ========== ChromaDB Hybrid Search Endpoints (Phase 5) ==========


class SemanticSearchRequest(BaseModel):
    """Request for semantic search."""

    session_id: UUID
    query: str
    limit: int = 5
    memory_type: Optional[str] = None
    min_similarity: float = 0.5


class SemanticSearchResult(BaseModel):
    """Single semantic search result."""

    memory_id: UUID
    memory_type: str
    content: str
    similarity_score: float
    source: Optional[str]
    tags: List[str]


@router.post("/search/semantic/hybrid", summary="Semantic search with ChromaDB fallback")
async def semantic_search_hybrid(
    request: SemanticSearchRequest,
    api_key_id: UUID = Depends(get_current_api_key_id),
):
    """Search memories by semantic similarity using hybrid store.

    Uses ChromaDB if available, falls back to pgvector.
    """
    try:
        # Check if memory_store is a HybridMemoryStore
        if not hasattr(memory_store, "search_semantic"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Semantic search not available (HybridMemoryStore not configured)",
            )

        results = await memory_store.search_semantic(
            session_id=request.session_id,
            query=request.query,
            limit=request.limit,
            memory_type=request.memory_type,
            min_similarity=request.min_similarity,
        )

        return {
            "query": request.query,
            "results": [
                SemanticSearchResult(
                    memory_id=memory.id,
                    memory_type=memory.memory_type,
                    content=memory.content[:200],
                    similarity_score=similarity,
                    source=memory.source,
                    tags=memory.tags,
                ).dict()
                for memory, similarity in results
            ],
            "count": len(results),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Semantic search failed",
        )


@router.get(
    "/search/performance-stats/{session_id}",
    summary="Get search backend performance stats",
)
async def get_performance_stats(
    session_id: UUID,
    api_key_id: UUID = Depends(get_current_api_key_id),
):
    """Get performance metrics for ChromaDB vs pgvector backends."""
    try:
        if not hasattr(memory_store, "get_search_performance_stats"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Performance stats not available",
            )

        stats = await memory_store.get_search_performance_stats()

        return {
            "session_id": str(session_id),
            "chroma_p50_latency_ms": stats.get("chroma", {}).get("p50_latency_ms"),
            "pgvector_p50_latency_ms": stats.get("pgvector", {}).get("p50_latency_ms"),
            "chroma_error_rate": stats.get("chroma", {}).get("error_rate"),
            "pgvector_error_rate": stats.get("pgvector", {}).get("error_rate"),
            "preferred_backend": stats.get("preferred_backend"),
            "stats": stats,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting performance stats: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get performance stats",
        )


@router.get("/sync-status/{session_id}", summary="Get ChromaDB sync status")
async def get_sync_status(
    session_id: UUID,
    api_key_id: UUID = Depends(get_current_api_key_id),
):
    """Get ChromaDB sync status and queue information."""
    try:
        if not hasattr(memory_store, "get_chroma_sync_status"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Sync status not available",
            )

        status_info = await memory_store.get_chroma_sync_status(session_id)

        return {
            "session_id": str(session_id),
            **status_info,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting sync status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get sync status",
        )


@router.post("/sync/force-resync/{session_id}", summary="Force full resync to ChromaDB")
async def force_resync(
    session_id: UUID,
    api_key_id: UUID = Depends(get_current_api_key_id),
):
    """Force a complete resync of all memories for a session to ChromaDB."""
    try:
        if not hasattr(memory_store, "sync_all_memories_to_chroma"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Resync not available",
            )

        result = await memory_store.sync_all_memories_to_chroma(session_id)

        return {
            "session_id": str(session_id),
            "success": True,
            **result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during resync: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Resync failed",
        )


@router.get("/health/chroma", summary="Get ChromaDB health status")
async def get_chroma_health(
    api_key_id: UUID = Depends(get_current_api_key_id),
):
    """Get ChromaDB health and connectivity status."""
    try:
        if not hasattr(memory_store, "get_chroma_health"):
            return {
                "status": "unavailable",
                "message": "ChromaDB not configured",
            }

        health = await memory_store.get_chroma_health()

        return {
            "timestamp": str(datetime.utcnow()),
            **health,
        }

    except Exception as e:
        logger.error(f"Error getting ChromaDB health: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": f"Health check failed: {str(e)}",
        }
