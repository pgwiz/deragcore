"""API Key management and validation."""

import logging
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Tuple
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ragcore.config import settings
from ragcore.db.database import async_session_factory
from ragcore.auth.models import APIKey, User

logger = logging.getLogger(__name__)


class APIKeyManager:
    """Manage API key generation, validation, and tracking."""

    @staticmethod
    async def generate_key(
        user_id: UUID,
        name: Optional[str] = None,
        expires_in_days: Optional[int] = None,
    ) -> Tuple[str, APIKey]:
        """
        Generate new API key for user.

        Args:
            user_id: User ID
            name: Optional key name/label
            expires_in_days: Days until expiration (None = no expiration)

        Returns:
            Tuple of (plaintext_key, api_key_record)
        """
        # Generate random key
        plaintext = f"rg_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(plaintext.encode()).hexdigest()

        async with async_session_factory() as session:
            api_key = APIKey(
                user_id=user_id,
                key=key_hash,
                name=name or f"Key-{datetime.utcnow().strftime('%Y%m%d')}",
                expires_at=(
                    datetime.utcnow() + timedelta(days=expires_in_days)
                    if expires_in_days
                    else None
                ),
            )
            session.add(api_key)
            await session.commit()
            await session.refresh(api_key)

            logger.info(f"Generated API key for user {user_id}: {api_key.id}")
            return plaintext, api_key

    @staticmethod
    async def validate_key(plaintext_key: str) -> Optional[Tuple[APIKey, User]]:
        """
        Validate API key and return associated user.

        Args:
            plaintext_key: Raw API key from request

        Returns:
            Tuple of (api_key, user) if valid, None otherwise
        """
        if not plaintext_key:
            return None

        key_hash = hashlib.sha256(plaintext_key.encode()).hexdigest()

        async with async_session_factory() as session:
            stmt = (
                select(APIKey)
                .where(APIKey.key == key_hash)
                .where(APIKey.is_active == True)
            )
            result = await session.execute(stmt)
            api_key = result.scalar_one_or_none()

            if not api_key:
                logger.debug("API key not found or inactive")
                return None

            # Check expiration
            if api_key.expires_at and datetime.utcnow() > api_key.expires_at:
                logger.warning(f"API key expired: {api_key.id}")
                return None

            # Get user
            stmt = select(User).where(User.id == api_key.user_id)
            result = await session.execute(stmt)
            user = result.scalar_one_or_none()

            if not user or not user.is_active:
                logger.warning(f"User not found or inactive for key {api_key.id}")
                return None

            # Update last_used_at
            api_key.last_used_at = datetime.utcnow()
            session.add(api_key)
            await session.commit()

            return api_key, user

    @staticmethod
    async def list_keys(user_id: UUID) -> list[APIKey]:
        """List all API keys for user."""
        async with async_session_factory() as session:
            stmt = select(APIKey).where(APIKey.user_id == user_id)
            result = await session.execute(stmt)
            return result.scalars().all()

    @staticmethod
    async def revoke_key(api_key_id: UUID) -> bool:
        """Revoke (disable) an API key."""
        async with async_session_factory() as session:
            stmt = select(APIKey).where(APIKey.id == api_key_id)
            result = await session.execute(stmt)
            api_key = result.scalar_one_or_none()

            if not api_key:
                return False

            api_key.is_active = False
            session.add(api_key)
            await session.commit()

            logger.info(f"Revoked API key: {api_key_id}")
            return True

    @staticmethod
    async def increment_request_count(api_key: APIKey) -> None:
        """Track request count for quota enforcement."""
        async with async_session_factory() as session:
            # Refresh to get latest counts
            stmt = select(APIKey).where(APIKey.id == api_key.id)
            result = await session.execute(stmt)
            current_key = result.scalar_one()

            # Reset if new day
            today = datetime.utcnow().date()
            if (
                not current_key.last_reset_date
                or current_key.last_reset_date.date() != today
            ):
                current_key.requests_today = 0
                current_key.last_reset_date = datetime.utcnow()

            current_key.requests_today += 1
            current_key.requests_month += 1

            session.add(current_key)
            await session.commit()
