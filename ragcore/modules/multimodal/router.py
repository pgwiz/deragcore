"""HTTP router for multi-modal content operations."""

import logging
from typing import Optional, List
from uuid import UUID
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, status
from pydantic import BaseModel

from ragcore.auth.dependencies import get_current_api_key_id
from ragcore.modules.multimodal.models import (
    MultiModalContent,
    MultiModalChunk,
    ModuleType,
    ProcessingResult,
)

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
        valid_modalities = ["image", "audio", "video"]
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

        # Placeholder: would store content and queue for processing
        # For now, return success response
        import uuid

        content_id = uuid.uuid4()

        return {
            "id": str(content_id),
            "session_id": str(session_id),
            "modality": modality,
            "file_name": file.filename,
            "file_size_bytes": len(content_bytes),
            "is_processed": False,
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
        # Placeholder: would fetch from database
        return {
            "id": str(content_id),
            "is_processed": False,
            "chunks": [],
            "message": "Content information placeholder",
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

        # Placeholder: would search embeddings
        return {
            "query": request.query,
            "session_id": str(request.session_id),
            "results": [],
            "count": 0,
            "message": "Search results placeholder",
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
    """List all extracted chunks for a session.

    Can optionally filter by modality (image, audio, video, text).

    Args:
        session_id: Session UUID
        modality: Optional modality filter
        limit: Max chunks to return
        api_key_id: API key from auth

    Returns:
        List of chunks with metadata
    """
    try:
        if modality:
            valid = ["image", "audio", "video", "text"]
            if modality.lower() not in valid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid modality. Must be one of: {valid}",
                )

        # Placeholder: would fetch from database
        return {
            "session_id": str(session_id),
            "modality_filter": modality,
            "chunks": [],
            "count": 0,
            "message": "Chunks list placeholder",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing chunks: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list chunks",
        )


@router.get("/session/{session_id}/stats", summary="Get session multi-modal statistics")
async def get_session_stats(
    session_id: UUID,
    api_key_id: UUID = Depends(get_current_api_key_id),
):
    """Get statistics about multi-modal content in session.

    Includes counts by modality, total tokens, embedding coverage, etc.

    Args:
        session_id: Session UUID
        api_key_id: API key from auth

    Returns:
        Statistics dictionary
    """
    try:
        # Placeholder: would calculate from database
        return {
            "session_id": str(session_id),
            "total_content": 0,
            "total_chunks": 0,
            "by_modality": {
                "image": {"count": 0, "avg_confidence": 0},
                "audio": {"count": 0, "avg_confidence": 0},
                "video": {"count": 0, "avg_confidence": 0},
                "text": {"count": 0, "avg_confidence": 0},
            },
            "total_tokens": 0,
            "embedded_chunks": 0,
            "embedding_coverage": 0.0,
        }

    except Exception as e:
        logger.error(f"Error getting session stats: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get statistics",
        )


@router.post("/session/{session_id}/process", summary="Reprocess session content")
async def reprocess_session_content(
    session_id: UUID,
    modalities: Optional[List[str]] = None,
    api_key_id: UUID = Depends(get_current_api_key_id),
):
    """Manually trigger reprocessing of session content.

    Useful for retrying failed processing or using updated processors.

    Args:
        session_id: Session UUID
        modalities: Optional list of modalities to reprocess
        api_key_id: API key from auth

    Returns:
        Processing job status
    """
    try:
        # Placeholder: would queue reprocessing
        return {
            "session_id": str(session_id),
            "modalities": modalities,
            "queued_count": 0,
            "message": "Reprocessing queued",
        }

    except Exception as e:
        logger.error(f"Error reprocessing: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to queue reprocessing",
        )


@router.get("/processing-status/{content_id}", summary="Get content processing status")
async def get_processing_status(
    content_id: UUID,
    api_key_id: UUID = Depends(get_current_api_key_id),
):
    """Get detailed processing status for a content item.

    Args:
        content_id: Content UUID
        api_key_id: API key from auth

    Returns:
        Processing status with timing and results
    """
    try:
        # Placeholder: would fetch from database
        return {
            "content_id": str(content_id),
            "is_processed": False,
            "success": None,
            "processing_time_ms": 0,
            "chunks_extracted": 0,
            "tokens_used": 0,
            "extraction_method": None,
            "error_message": None,
        }

    except Exception as e:
        logger.error(f"Error getting processing status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get processing status",
        )


@router.delete("/content/{content_id}", summary="Delete multi-modal content")
async def delete_multimodal_content(
    content_id: UUID,
    api_key_id: UUID = Depends(get_current_api_key_id),
):
    """Delete multi-modal content and its chunks.

    Removes storage references (S3/Blob) and database records.

    Args:
        content_id: Content UUID
        api_key_id: API key from auth

    Returns:
        Deletion confirmation
    """
    try:
        # Placeholder: would delete from database and storage
        return {
            "content_id": str(content_id),
            "deleted": True,
            "message": "Content and chunks deleted",
        }

    except Exception as e:
        logger.error(f"Error deleting content: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete content",
        )
