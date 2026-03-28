"""Health check and monitoring endpoints."""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from ragcore.config import settings
from ragcore.db.database import async_session_factory
from ragcore.auth.models import APIKey

logger = logging.getLogger(__name__)


class HealthCheck:
    """Monitor application health and dependencies."""

    @staticmethod
    async def check_database() -> tuple[bool, Optional[str]]:
        """Check database connectivity."""
        try:
            async with async_session_factory() as session:
                result = await session.execute(text("SELECT 1"))
                return True, None
        except Exception as e:
            error_msg = f"Database check failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    @staticmethod
    async def check_api_keys() -> tuple[bool, Optional[str]]:
        """Check API key store accessibility."""
        try:
            async with async_session_factory() as session:
                stmt = select(APIKey).limit(1)
                await session.execute(stmt)
                return True, None
        except Exception as e:
            error_msg = f"API key store check failed: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

    @staticmethod
    async def get_health_status() -> Dict[str, Any]:
        """
        Get comprehensive health status.

        Returns:
            Health status dict with all services
        """
        db_ok, db_error = await HealthCheck.check_database()
        keys_ok, keys_error = await HealthCheck.check_api_keys()

        return {
            "status": "healthy" if (db_ok and keys_ok) else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "database": {"status": "ok" if db_ok else "error", "error": db_error},
                "api_keys": {
                    "status": "ok" if keys_ok else "error",
                    "error": keys_error,
                },
                "auth": {"status": "enabled" if settings.auth_enabled else "disabled"},
                "rate_limiting": {
                    "status": "enabled" if settings.rate_limit_enabled else "disabled"
                },
                "audit_logging": {
                    "status": "enabled" if settings.audit_logging_enabled else "disabled"
                },
                "webhooks": {
                    "status": "enabled" if settings.webhook_enabled else "disabled"
                },
            },
            "config": {
                "environment": settings.env,
                "log_level": settings.log_level,
                "auth_required": settings.require_api_key,
                "quota_enabled": settings.quota_enabled,
            },
        }


health_check = HealthCheck()
