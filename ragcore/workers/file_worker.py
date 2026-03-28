"""Background worker for async file processing using ARQ (Async Redis Queue)."""

import logging
from typing import Optional
from uuid import UUID

from ragcore.config import settings
from ragcore.db.database import get_db_session
from ragcore.models import File, Job
from ragcore.modules.files.pipeline import FileProcessingPipeline
from sqlalchemy import select
from datetime import datetime

logger = logging.getLogger(__name__)


async def process_file_job(
    ctx: dict,
    file_id: str,
    file_bytes: bytes,
    content_type: str,
    session_id: Optional[str] = None,
) -> dict:
    """
    ARQ job for processing uploaded files.

    Runs asynchronously in background via Redis queue.
    ARQ automatically retries on failure (max_retries from config).

    Args:
        ctx: ARQ job context (includes redis connection, etc.)
        file_id: UUID of File record (as string)
        file_bytes: Raw file content
        content_type: MIME type
        session_id: Optional session UUID (as string)

    Returns:
        dict with job result: {status: 'completed' | 'failed', chunks: int, error: str}
    """
    file_id_uuid = UUID(file_id)
    session_id_uuid = UUID(session_id) if session_id else None

    logger.info(
        f"process_file_job START: file_id={file_id}, "
        f"size={len(file_bytes)} bytes"
    )

    try:
        # =====================================================================
        # Fetch Job record and update to 'running'
        # =====================================================================
        async with get_db_session() as session:
            # Find Job record (there should only be one pending job for this file)
            stmt = select(Job).where(
                (Job.job_type == "file_process") & (Job.status == "pending")
            )
            result = await session.execute(stmt)
            job_record = result.scalar_one_or_none()

            if not job_record:
                logger.warning(f"No pending job found for file {file_id}")
                return {
                    "status": "failed",
                    "chunks": 0,
                    "error": "Job record not found",
                }

            job_id = job_record.id

            # Update job status to 'running'
            job_record.status = "running"
            job_record.started_at = datetime.utcnow()
            session.add(job_record)
            await session.commit()

            logger.info(f"Job {job_id} status updated to 'running'")

        # =====================================================================
        # Run file processing pipeline
        # =====================================================================
        pipeline = FileProcessingPipeline()

        await pipeline.process(
            file_id=file_id_uuid,
            file_bytes=file_bytes,
            content_type=content_type,
            session_id=session_id_uuid,
        )

        # =====================================================================
        # Fetch File record to get final chunk count
        # =====================================================================
        async with get_db_session() as session:
            file_record = await session.get(File, file_id_uuid)

            if not file_record:
                logger.error(f"File {file_id} not found after processing")
                return {
                    "status": "failed",
                    "chunks": 0,
                    "error": "File record not found after processing",
                }

            chunks_count = file_record.chunks_count
            status = file_record.status

            # Update job record with result
            stmt = select(Job).where(Job.job_type == "file_process")
            result = await session.execute(stmt)
            job_record = result.scalar_one_or_none()

            if job_record:
                job_record.status = "completed" if status == "ready" else "failed"
                job_record.completed_at = datetime.utcnow()
                job_record.result = {
                    "file_id": str(file_id),
                    "chunks_count": chunks_count,
                    "file_status": status,
                }
                job_record.error = file_record.error_message if status == "failed" else None
                session.add(job_record)
                await session.commit()

                logger.info(
                    f"Job {job_record.id} completed: "
                    f"file_status={status}, chunks={chunks_count}"
                )

        return {
            "status": "completed" if status == "ready" else "failed",
            "chunks": chunks_count,
            "error": file_record.error_message if status == "failed" else None,
        }

    except Exception as e:
        logger.error(
            f"process_file_job FAILED: {str(e)}",
            exc_info=True,
        )

        # Try to update job record with error
        try:
            async with get_db_session() as session:
                stmt = select(Job).where(Job.job_type == "file_process")
                result = await session.execute(stmt)
                job_record = result.scalar_one_or_none()

                if job_record:
                    job_record.status = "failed"
                    job_record.completed_at = datetime.utcnow()
                    job_record.error = str(e)
                    job_record.retry_count = job_record.retry_count + 1
                    session.add(job_record)
                    await session.commit()

                    # Try to update File record status
                    file_record = await session.get(File, UUID(file_id))
                    if file_record:
                        file_record.status = "failed"
                        file_record.error_message = f"Job error: {str(e)}"
                        session.add(file_record)
                        await session.commit()

        except Exception as db_error:
            logger.error(f"Failed to update error status: {str(db_error)}")

        return {
            "status": "failed",
            "chunks": 0,
            "error": str(e),
        }


# =====================================================================
# ARQ Worker Function List
# =====================================================================

async def startup(ctx):
    """ARQ startup hook - called when worker starts."""
    logger.info("File worker starting...")


async def shutdown(ctx):
    """ARQ shutdown hook - called when worker stops."""
    logger.info("File worker shutting down...")
