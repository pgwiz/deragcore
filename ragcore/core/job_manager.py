"""Job management - Queue async tasks via ARQ."""

import logging
from typing import Optional
from uuid import UUID

try:
    from arq import create_pool
    from arq.connections import RedisSettings
except ImportError:
    raise ImportError("arq package required - pip install arq")

from ragcore.config import settings

logger = logging.getLogger(__name__)

# Global redis pool for queueing
_redis_pool = None


async def get_redis_pool():
    """Get or create ARQ Redis connection pool."""
    global _redis_pool

    if _redis_pool is None:
        try:
            redis_settings = RedisSettings.from_dsn(settings.redis_url)
            _redis_pool = await create_pool(redis_settings)
            logger.info("Created ARQ redis pool")
        except Exception as e:
            logger.error(f"Failed to create redis pool: {str(e)}")
            raise

    return _redis_pool


async def close_redis_pool():
    """Close ARQ Redis connection pool."""
    global _redis_pool

    if _redis_pool is not None:
        await _redis_pool.close()
        _redis_pool = None
        logger.info("Closed ARQ redis pool")


async def queue_file_processing_job(
    file_id: UUID,
    file_bytes: bytes,
    content_type: str,
    session_id: Optional[UUID] = None,
) -> None:
    """
    Queue a file processing job for async execution.

    Args:
        file_id: UUID of File record
        file_bytes: Raw file content
        content_type: MIME type
        session_id: Optional session UUID

    Raises:
        RuntimeError: If redis pool not available
    """
    try:
        pool = await get_redis_pool()

        # Enqueue the job - arq will serialize/deserialize the file_bytes
        job = await pool.enqueue_job(
            "process_file_job",
            str(file_id),  # Convert UUID to string for serialization
            file_bytes,
            content_type,
            str(session_id) if session_id else None,
        )

        logger.info(f"Queued file processing job: file_id={file_id}, job_id={job.id}")

    except Exception as e:
        logger.error(f"Failed to queue job: {str(e)}")
        raise RuntimeError(f"Failed to queue job: {str(e)}") from e
