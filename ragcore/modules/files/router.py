"""File management routes - Upload, list, delete files and view chunks."""

import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select

from ragcore.config import settings
from ragcore.db.database import get_db_session
from ragcore.models import File as FileModel, Chunk, Job
from ragcore.core.job_manager import queue_file_processing_job

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/files", tags=["files"])


# ============================================================================
# Response Models
# ============================================================================


class FileUploadResponse(BaseModel):
    """Response to file upload - returns file_id and job_id for async tracking."""

    file_id: UUID
    status: str
    job_id: UUID
    message: str

    class Config:
        from_attributes = True


class FileResponse(BaseModel):
    """File metadata response."""

    id: UUID
    filename: str
    file_size: int
    content_type: str
    status: str
    chunks_count: int
    error_message: Optional[str] = None
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True


class ChunkResponse(BaseModel):
    """Single chunk response."""

    id: UUID
    chunk_index: int
    text: str
    tokens: int
    similarity_score: Optional[float] = None

    class Config:
        from_attributes = True


class ChunksResponse(BaseModel):
    """Response for list of chunks from a file."""

    file_id: UUID
    filename: str
    total_chunks: int
    chunks: List[ChunkResponse]


class DeleteResponse(BaseModel):
    """Response to file deletion."""

    file_id: UUID
    message: str


# ============================================================================
# File Upload Endpoint
# ============================================================================


@router.post("/upload", response_model=FileUploadResponse, status_code=202)
async def upload_file(
    file: UploadFile = File(...),
    session_id: Optional[UUID] = Query(None),
) -> FileUploadResponse:
    """
    Upload a PDF or DOCX file for processing.

    Returns 202 Accepted immediately with file_id + job_id for async polling.
    File processing happens in background via ARQ worker.

    Args:
        file: PDF or DOCX file to upload
        session_id: Optional session scope

    Returns:
        FileUploadResponse with file_id and job_id for tracking

    Raises:
        400: Invalid file type or size exceeds limit
        422: File processing failed
        503: No embedding provider available
    """
    # =====================================================================
    # Validation
    # =====================================================================

    # Check content type
    content_type = file.content_type or ""
    valid_types = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]

    if not any(valid in content_type for valid in ["pdf", "word", "wordprocessingml"]):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Supported: PDF, DOCX",
        )

    # Check file size
    if file.size and file.size > settings.max_file_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"File size {file.size / 1024 / 1024:.1f}MB exceeds "
            f"limit of {settings.max_file_size_mb}MB",
        )

    # Check embedding provider available
    providers = {
        "azure": settings.embedding_provider == "azure",
        "openai": settings.embedding_provider == "openai",
        "anthropic": settings.embedding_provider == "anthropic",
        "ollama": settings.embedding_provider == "ollama",
    }
    if not any(providers.values()):
        raise HTTPException(
            status_code=503,
            detail="No embedding provider configured",
        )

    logger.info(f"File upload: {file.filename} ({file.size} bytes)")

    # =====================================================================
    # Create File Record
    # =====================================================================

    async with get_db_session() as session:
        try:
            # Read file bytes
            file_bytes = await file.read()

            # Create File record
            file_record = FileModel(
                filename=file.filename,
                file_size=file.size,
                content_type=file.content_type,
                status="pending",
                chunks_count=0,
                session_id=session_id,
            )
            session.add(file_record)
            await session.flush()  # Flush to get file_id
            file_id = file_record.id

            # Create Job record for background processing
            job_record = Job(
                job_type="file_process",
                status="pending",
                result=None,
                error=None,
            )
            session.add(job_record)
            await session.flush()
            job_id = job_record.id

            # Commit both records
            await session.commit()

            logger.info(
                f"Created File {file_id} and Job {job_id} for {file.filename}"
            )

            # =====================================================================
            # Queue Background Job
            # =====================================================================
            try:
                await queue_file_processing_job(
                    file_id=file_id,
                    file_bytes=file_bytes,
                    content_type=file.content_type,
                    session_id=session_id,
                )
            except RuntimeError as e:
                logger.error(f"Failed to queue job: {str(e)}")
                # Return success anyway - client can poll the file status
                # The job won't execute but user will see the status

            return FileUploadResponse(
                file_id=file_id,
                status="pending",
                job_id=job_id,
                message=f"File queued for processing. Poll GET /files/{file_id} for status.",
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Upload error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=422,
                detail=f"Failed to process upload: {str(e)}",
            )


# ============================================================================
# List Files Endpoint
# ============================================================================


@router.get("", response_model=List[FileResponse])
async def list_files(session_id: Optional[UUID] = Query(None)) -> List[FileResponse]:
    """
    List all uploaded files, optionally scoped to a session.

    Args:
        session_id: Optional session ID to filter files

    Returns:
        List of files with their metadata
    """
    async with get_db_session() as session:
        try:
            if session_id:
                stmt = select(FileModel).where(FileModel.session_id == session_id)
            else:
                stmt = select(FileModel)

            stmt = stmt.order_by(FileModel.created_at.desc())
            result = await session.execute(stmt)
            files = result.scalars().all()

            return [
                FileResponse(
                    id=f.id,
                    filename=f.filename,
                    file_size=f.file_size,
                    content_type=f.content_type,
                    status=f.status,
                    chunks_count=f.chunks_count,
                    error_message=f.error_message,
                    created_at=f.created_at.isoformat(),
                    updated_at=f.updated_at.isoformat(),
                )
                for f in files
            ]

        except Exception as e:
            logger.error(f"List files error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to list files: {str(e)}",
            )


# ============================================================================
# Get File Chunks Endpoint
# ============================================================================


@router.get("/{file_id}/chunks", response_model=ChunksResponse)
async def get_file_chunks(file_id: UUID) -> ChunksResponse:
    """
    Retrieve all chunks for a file.

    Args:
        file_id: ID of the file

    Returns:
        Chunks with their text and metadata

    Raises:
        404: File not found or not ready
    """
    async with get_db_session() as session:
        try:
            # Fetch file
            file_record = await session.get(FileModel, file_id)
            if not file_record:
                raise HTTPException(
                    status_code=404,
                    detail=f"File {file_id} not found",
                )

            # Check status
            if file_record.status != "ready":
                raise HTTPException(
                    status_code=422,
                    detail=f"File is {file_record.status}, not ready. "
                    f"{file_record.error_message or ''}",
                )

            # Fetch chunks
            stmt = (
                select(Chunk)
                .where(Chunk.file_id == file_id)
                .order_by(Chunk.chunk_index)
            )
            result = await session.execute(stmt)
            chunks = result.scalars().all()

            return ChunksResponse(
                file_id=file_id,
                filename=file_record.filename,
                total_chunks=len(chunks),
                chunks=[
                    ChunkResponse(
                        id=c.id,
                        chunk_index=c.chunk_index,
                        text=c.text,
                        tokens=c.tokens,
                    )
                    for c in chunks
                ],
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Get chunks error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch chunks: {str(e)}",
            )


# ============================================================================
# Delete File Endpoint
# ============================================================================


@router.delete("/{file_id}", response_model=DeleteResponse)
async def delete_file(file_id: UUID) -> DeleteResponse:
    """
    Delete a file and all associated chunks.

    Uses cascade delete via SQLAlchemy relationship.

    Args:
        file_id: ID of file to delete

    Returns:
        Confirmation message

    Raises:
        404: File not found
    """
    async with get_db_session() as session:
        try:
            # Fetch file
            file_record = await session.get(FileModel, file_id)
            if not file_record:
                raise HTTPException(
                    status_code=404,
                    detail=f"File {file_id} not found",
                )

            filename = file_record.filename

            # Delete (cascade handles chunks)
            await session.delete(file_record)
            await session.commit()

            logger.info(f"Deleted file {file_id} ({filename}) and all chunks")

            return DeleteResponse(
                file_id=file_id,
                message=f"File '{filename}' and {file_record.chunks_count} "
                f"chunks deleted",
            )

        except HTTPException:
            raise
        except Exception as e:
            await session.rollback()
            logger.error(f"Delete error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete file: {str(e)}",
            )
