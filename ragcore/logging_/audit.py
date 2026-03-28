"""Structured logging and audit trail."""

import logging
import time
from typing import Optional
from uuid import UUID
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ragcore.db.database import async_session_factory
from ragcore.auth.models import AuditLog

logger = logging.getLogger(__name__)


class AuditLogger:
    """Track all API requests and responses for compliance/debugging."""

    @staticmethod
    async def log_request(
        method: str,
        path: str,
        ip_address: Optional[str],
        user_agent: Optional[str],
        user_id: Optional[UUID],
        api_key_id: Optional[UUID],
        status_code: int,
        response_time_ms: Optional[int],
        error: Optional[str] = None,
    ) -> None:
        """
        Log API request to audit trail.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: Request path
            ip_address: Client IP
            user_agent: User-Agent header
            user_id: Authenticated user ID
            api_key_id: Used API key ID
            status_code: Response HTTP status
            response_time_ms: Response time in milliseconds
            error: Error message if failed
        """
        from ragcore.config import settings

        if not settings.audit_logging_enabled:
            return

        try:
            async with async_session_factory() as session:
                audit_log = AuditLog(
                    user_id=user_id,
                    api_key_id=api_key_id,
                    method=method,
                    path=path,
                    status_code=status_code,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    response_time_ms=response_time_ms,
                    error=error[:512] if error else None,  # Truncate long errors
                    created_at=datetime.utcnow(),
                )
                session.add(audit_log)
                await session.commit()

                logger.debug(
                    f"Audit log: {method} {path} {status_code} "
                    f"({response_time_ms}ms) user={user_id}"
                )

        except Exception as e:
            logger.error(f"Failed to log audit trail: {str(e)}", exc_info=True)

    @staticmethod
    async def get_logs(
        user_id: Optional[UUID] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditLog]:
        """
        Retrieve audit logs (filtered by user if provided).

        Args:
            user_id: Optional user ID to filter by
            limit: Max results
            offset: Pagination offset

        Returns:
            List of audit log entries
        """
        async with async_session_factory() as session:
            stmt = select(AuditLog).order_by(AuditLog.created_at.desc())

            if user_id:
                stmt = stmt.where(AuditLog.user_id == user_id)

            stmt = stmt.limit(limit).offset(offset)

            result = await session.execute(stmt)
            return result.scalars().all()


# Global audit logger instance
audit_logger = AuditLogger()
