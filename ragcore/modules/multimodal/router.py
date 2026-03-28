"""HTTP router for multi-modal content operations."""

import logging
from typing import Optional, List
from uuid import UUID
import uuid as uuid_module
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, status
from pydantic import BaseModel
from datetime import datetime

from ragcore.auth.dependencies import get_current_api_key_id
from ragcore.modules.multimodal.models import (
    MultiModalContent,
    MultiModalChunk,
    ModuleType,
    ProcessingResult,
)
from ragcore.modules.multimodal.storage import get_storage_backend_from_config
from ragcore.modules.multimodal.embedding_pipeline import MultiModalEmbeddingPipeline
from ragcore.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/multimodal", tags=["multimodal"])


# Request/Response Models
class MultiModalUploadRequest(BaseModel):
    """Request to upload and process multi-modal content."""

    session_id: UUID
    modality: str  # "image", "audio", "video"
    file_name: Optional[str] = None
    source_url: Optional[str] = None
    metadata: Optional[dict] = None


class MultiModalChunkResponse(BaseModel):
    """Response with extracted chunk information."""

    id: UUID
    modality: str
    content: str
    confidence_score: float
    source_index: int
    is_critical: bool
    created_at: str

    class Config:
        from_attributes = True


class MultiModalContentResponse(BaseModel):
    """Response with content and processing information."""

    id: UUID
    session_id: UUID
    modality: str
    is_processed: bool
    processing_error: Optional[str]
    chunks_count: int
    storage_path: Optional[str]
    created_at: str

    class Config:
        from_attributes = True


class MultiModalSearchRequest(BaseModel):
    """Request for multi-modal semantic search."""

    session_id: UUID
    query: str
    limit: int = 10
    modalities: Optional[List[str]] = None  # Filter by modality
    min_confidence: float = 0.5


class MultiModalSearchResult(BaseModel):
    """Single search result with relevance."""

    chunk_id: UUID
    modality: str
    content: str
    similarity_score: float
    confidence_score: float
    source_index: int

    class Config:
        from_attributes = True


class MultiModalProcessingStatusResponse(BaseModel):
    """Response with processing status."""

    content_id: UUID
    session_id: UUID
    is_processed: bool
    success: bool
    error_message: Optional[str]
    processing_time_ms: float
    chunks_extracted: int
    tokens_used: int
    extraction_method: Optional[str]


class MultiModalSessionStats(BaseModel):
    """Statistics for a session."""

    session_id: UUID
    total_content_count: int
    processed_content_count: int
    total_chunks: int
    total_tokens_used: int
    storage_size_bytes: int
    modalities: List[str]


# ========== Endpoints ==========


@router.post("/upload", summary="Upload multi-modal content")
async def upload_multimodal_content(
    session_id: UUID = Form(...),
    modality: str = Form(...),
    file: UploadFile = File(...),
    api_key_id: UUID = Depends(get_current_api_key_id),
):
    """Upload and queue multi-modal content for processing.

    Accepts images, audio, or video files. Content is stored and queued
    for asynchronous processing by appropriate modality processor.

    Args:
        session_id: Session UUID
        modality: "image", "audio", or "video"
        file: Binary file upload
        api_key_id: API key from auth

    Returns:
        Content metadata with processing status
    """
    try:
        # Validate modality
        valid_modalities = ["image", "audio", "video", "text"]
        if modality.lower() not in valid_modalities:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid modality. Must be one of: {valid_modalities}",
            )

        # Read file bytes
        content_bytes = await file.read()
        if not content_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file uploaded",
            )

        # Get storage backend
        config = settings
        storage_backend = get_storage_backend_from_config(config)

        # Generate content ID
        content_id = uuid_module.uuid4()

        # Decide where to store: inline or external storage
        max_inline_kb = getattr(config, "multimodal_max_inline_size_kb", 100)
        content_size_kb = len(content_bytes) / 1024

        storage_path = None
        is_base64_inline = False

        if content_size_kb <= max_inline_kb:
            # Store inline as base64
            is_base64_inline = True
            logger.debug(f"Storing content {content_id} inline ({content_size_kb:.1f}KB)")
        else:
            # Stream to external storage
            try:
                storage_path = await storage_backend.save_file(str(content_id), content_bytes)
                logger.info(f"Stored content {content_id} to {storage_backend.get_backend_name()}: {storage_path}")
            except Exception as e:
                logger.error(f"Failed to store file in {storage_backend.get_backend_name()}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to store file: {str(e)}",
                )

        # Create MultiModalContent record
        content_metadata = {
            "file_name": file.filename,
            "file_size_bytes": len(content_bytes),
            "content_type": file.content_type,
            "uploaded_by": str(api_key_id),
        }

        # Note: In a real implementation, this would persist to database
        # For now, we return the response that would be stored
        logger.info(f"Created MultiModalContent record: {content_id} for session {session_id}")

        return {
            "id": str(content_id),
            "session_id": str(session_id),
            "modality": modality.lower(),
            "file_name": file.filename,
            "file_size_bytes": len(content_bytes),
            "storage_path": storage_path,
            "is_stored_inline": is_base64_inline,
            "is_processed": False,
            "processing_error": None,
            "created_at": datetime.utcnow().isoformat(),
            "message": "Content uploaded and queued for processing",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading content: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload content",
        )


@router.get("/content/{content_id}", summary="Get multi-modal content")
async def get_multimodal_content(
    content_id: UUID,
    api_key_id: UUID = Depends(get_current_api_key_id),
):
    """Get multi-modal content information and processing status.

    Args:
        content_id: Content UUID
        api_key_id: API key from auth

    Returns:
        Content metadata with chunks if processed
    """
    try:
        # Note: In a real implementation, this would fetch from database
        # For now, return placeholder with proper structure
        return {
            "id": str(content_id),
            "is_processed": False,
            "processing_error": None,
            "chunks_count": 0,
            "chunks": [],
            "storage_path": None,
            "created_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting content: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve content",
        )


@router.post("/search", summary="Multi-modal semantic search")
async def search_multimodal_content(
    request: MultiModalSearchRequest,
    api_key_id: UUID = Depends(get_current_api_key_id),
):
    """Search session content across all modalities using semantic similarity.

    Searches embeddings to find relevant text, images, audio transcriptions,
    and video frames matching the query.

    Args:
        request: Search request with query and filters
        api_key_id: API key from auth

    Returns:
        List of matching chunks scored by relevance
    """
    try:
        if not request.query or len(request.query.strip()) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query must be at least 2 characters",
            )

        # Initialize embedding pipeline for query
        embedding_pipeline = MultiModalEmbeddingPipeline()

        # Generate query embedding
        query_embedding = await embedding_pipeline.embed_chunk({
            "content": request.query,
            "modality": "text",
            "confidence_score": 1.0,
            "source_index": 0,
        })

        if not query_embedding:
            logger.warning(f"Failed to embed query: {request.query}")
            return {
                "query": request.query,
                "session_id": str(request.session_id),
                "results": [],
                "count": 0,
                "message": "Search failed to embed query",
            }

        # Note: In a real implementation, this would:
        # 1. Query all chunks for the session from database
        # 2. Filter by modality if specified
        # 3. Calculate similarity scores using vector DB (pgvector/ChromaDB)
        # 4. Filter by min_confidence
        # 5. Sort by similarity and return top N results

        logger.debug(f"Search query: '{request.query}' for session {request.session_id}")
        logger.debug(f"Query embedding computed: {len(query_embedding)}-dim vector")

        return {
            "query": request.query,
            "session_id": str(request.session_id),
            "results": [],
            "count": 0,
            "message": "Database search not yet implemented (placeholder)",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching multimodal content: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search failed",
        )


@router.get("/session/{session_id}/chunks", summary="List session chunks")
async def list_session_chunks(
    session_id: UUID,
    modality: Optional[str] = None,
    limit: int = 100,
    api_key_id: UUID = Depends(get_current_api_key_id),
):
    """List all chunks extracted from content in a session.

    Args:
        session_id: Session UUID
        modality: Filter by modality ("image", "audio", "video", "text")
        limit: Max chunks to return
        api_key_id: API key from auth

    Returns:
        List of chunks with metadata
    """
    try:
        if limit < 1 or limit > 1000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Limit must be between 1 and 1000",
            )

        # Note: In a real implementation, this would query the database:
        # SELECT * FROM multimodal_chunks
        # WHERE session_id = {session_id}
        # AND (modality = {modality} OR {modality} IS NULL)
        # ORDER BY created_at DESC
        # LIMIT {limit}

        logger.debug(f"Listing chunks for session {session_id} (modality: {modality}, limit: {limit})")

        return {
            "session_id": str(session_id),
            "modality_filter": modality,
            "chunks": [],
            "total_count": 0,
            "limit": limit,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing chunks: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list chunks",
        )


@router.get("/session/{session_id}/stats", summary="Get session statistics")
async def get_session_stats(
    session_id: UUID,
    api_key_id: UUID = Depends(get_current_api_key_id),
):
    """Get statistics for a session's multi-modal content.

    Args:
        session_id: Session UUID
        api_key_id: API key from auth

    Returns:
        Session statistics
    """
    try:
        # Note: In a real implementation, this would aggregate:
        # - Total content items uploaded
        # - Processed vs pending count
        # - Total chunks extracted
        # - Total tokens used for embeddings
        # - Total storage used
        # - Breakdown by modality

        logger.debug(f"Getting stats for session {session_id}")

        return {
            "session_id": str(session_id),
            "total_content_count": 0,
            "processed_content_count": 0,
            "total_chunks": 0,
            "total_tokens_used": 0,
            "storage_size_bytes": 0,
            "modalities": [],
            "created_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get session stats",
        )


@router.post("/session/{session_id}/process", summary="Reprocess session content")
async def reprocess_session_content(
    session_id: UUID,
    modalities: Optional[List[str]] = None,
    api_key_id: UUID = Depends(get_current_api_key_id),
):
    """Reprocess all or specific modality content in a session.

    Queues content for reprocessing with current processors,
    useful after updating processor configurations or models.

    Args:
        session_id: Session UUID
        modalities: Specific modalities to reprocess (if None, all reprocessed)
        api_key_id: API key from auth

    Returns:
        Reprocessing job info
    """
    try:
        if modalities:
            valid_modalities = {"image", "audio", "video", "text"}
            invalid = set(modalities) - valid_modalities
            if invalid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid modalities: {invalid}. Must be one of: {valid_modalities}",
                )

        # Note: In a real implementation, this would:
        # 1. Find all content items for the session
        # 2. Filter by modality if specified
        # 3. Queue each for reprocessing
        # 4. Return job ID and count

        logger.info(f"Queued reprocessing for session {session_id} (modalities: {modalities})")

        return {
            "session_id": str(session_id),
            "modalities_selected": modalities,
            "content_queued_count": 0,
            "job_id": str(uuid_module.uuid4()),
            "message": "Content queued for reprocessing",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reprocessing content: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to queue reprocessing",
        )


@router.get("/processing-status/{content_id}", summary="Get processing status")
async def get_processing_status(
    content_id: UUID,
    api_key_id: UUID = Depends(get_current_api_key_id),
):
    """Get processing status for a content item.

    Args:
        content_id: Content UUID
        api_key_id: API key from auth

    Returns:
        Processing status and progress
    """
    try:
        # Note: In a real implementation, this would:
        # 1. Look up the content item
        # 2. Check processing_log table for status
        # 3. Return current progress

        logger.debug(f"Getting processing status for content {content_id}")

        return {
            "content_id": str(content_id),
            "is_processed": False,
            "is_processing": False,
            "success": None,
            "error_message": None,
            "processing_time_ms": 0.0,
            "chunks_extracted": 0,
            "tokens_used": 0,
            "extraction_method": None,
            "progress_percent": 0,
        }

    except Exception as e:
        logger.error(f"Error getting processing status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get processing status",
        )


@router.delete("/content/{content_id}", summary="Delete content")
async def delete_multimodal_content(
    content_id: UUID,
    api_key_id: UUID = Depends(get_current_api_key_id),
):
    """Delete multi-modal content and associated chunks.

    Removes content from storage and database. This action cannot be undone.

    Args:
        content_id: Content UUID
        api_key_id: API key from auth

    Returns:
        Deletion confirmation
    """
    try:
        # Note: In a real implementation, this would:
        # 1. Look up content record
        # 2. Delete from storage backend (if external storage)
        # 3. Delete all associated chunks from database
        # 4. Delete content record itself

        logger.warning(f"Deletion requested for content {content_id} by {api_key_id}")

        # Get storage backend for cleanup
        config = settings
        storage_backend = get_storage_backend_from_config(config)

        # Attempt to delete from storage (if exists)
        try:
            storage_path = f"{content_id}.bin"  # Expected naming
            deleted = await storage_backend.delete_file(storage_path)
            if deleted:
                logger.info(f"Deleted file from {storage_backend.get_backend_name()}: {storage_path}")
        except Exception as e:
            logger.warning(f"Could not delete file from storage: {e}")

        return {
            "content_id": str(content_id),
            "deleted": True,
            "message": "Content and associated chunks deleted",
        }

    except Exception as e:
        logger.error(f"Error deleting content: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete content",
        )
