"""FastAPI dependency injection for authentication."""

import logging
from typing import Optional
from fastapi import Depends, HTTPException, status, Header

from ragcore.config import settings
from ragcore.auth.manager import APIKeyManager
from ragcore.auth.models import APIKey, User

logger = logging.getLogger(__name__)


async def get_current_api_key(
    x_api_key: Optional[str] = Header(None),
) -> Optional[tuple[APIKey, User]]:
    """
    Dependency: Extract and validate API key from header.

    Args:
        x_api_key: API key from X-API-Key header

    Returns:
        Tuple of (api_key, user) if valid and auth enabled

    Raises:
        HTTPException: 401 if auth required but missing/invalid
    """
    if not settings.auth_enabled or not settings.require_api_key:
        return None

    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    result = await APIKeyManager.validate_key(x_api_key)
    if not result:
        logger.warning(f"Invalid API key attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return result


async def get_current_user(
    auth: Optional[tuple[APIKey, User]] = Depends(get_current_api_key),
) -> Optional[User]:
    """
    Dependency: Get current authenticated user.

    Args:
        auth: Auth tuple from get_current_api_key

    Returns:
        User object if authenticated
    """
    if auth is None:
        return None
    return auth[1]


async def get_current_api_key_id(
    auth: Optional[tuple[APIKey, User]] = Depends(get_current_api_key),
) -> Optional[str]:
    """
    Dependency: Get current API key ID for tracking.

    Args:
        auth: Auth tuple from get_current_api_key

    Returns:
        API key ID if authenticated
    """
    if auth is None:
        return None
    return str(auth[0].id)
