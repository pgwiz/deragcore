"""Auth endpoints - API key management."""

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, EmailStr

from ragcore.auth.manager import APIKeyManager
from ragcore.auth.dependencies import get_current_user, get_current_api_key
from ragcore.auth.models import User, APIKey

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


# ============================================================================
# Request/Response Models
# ============================================================================


class APIKeyResponse(BaseModel):
    """API key response (without plaintext)."""

    id: str
    name: str
    last_used_at: Optional[str] = None
    created_at: str
    expires_at: Optional[str] = None
    is_active: bool


class GenerateKeyRequest(BaseModel):
    """Request to generate new API key."""

    name: Optional[str] = None
    expires_in_days: Optional[int] = None


class GenerateKeyResponse(BaseModel):
    """Generate key response (includes plaintext key)."""

    key: str  # PLAINTEXT - shown only once
    id: str
    name: str
    created_at: str


class UserResponse(BaseModel):
    """User information response."""

    id: str
    email: str
    name: Optional[str] = None
    is_active: bool
    created_at: str


# ============================================================================
# Endpoints
# ============================================================================


@router.post("/keys", response_model=GenerateKeyResponse)
async def generate_api_key(
    req: GenerateKeyRequest,
    current_user: User = Depends(get_current_user),
) -> GenerateKeyResponse:
    """
    Generate new API key for current user.

    **Requires**: API authentication (X-API-Key header)
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    plaintext, api_key = await APIKeyManager.generate_key(
        user_id=current_user.id,
        name=req.name,
        expires_in_days=req.expires_in_days,
    )

    logger.info(f"Generated API key for user {current_user.id}")

    return GenerateKeyResponse(
        key=plaintext,  # Only shown once
        id=str(api_key.id),
        name=api_key.name,
        created_at=api_key.created_at.isoformat(),
    )


@router.get("/keys", response_model=list[APIKeyResponse])
async def list_api_keys(
    current_user: User = Depends(get_current_user),
) -> list[APIKeyResponse]:
    """
    List all API keys for current user.

    **Requires**: API authentication (X-API-Key header)
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    keys = await APIKeyManager.list_keys(current_user.id)

    return [
        APIKeyResponse(
            id=str(k.id),
            name=k.name,
            last_used_at=k.last_used_at.isoformat() if k.last_used_at else None,
            created_at=k.created_at.isoformat(),
            expires_at=k.expires_at.isoformat() if k.expires_at else None,
            is_active=k.is_active,
        )
        for k in keys
    ]


@router.delete("/keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    current_user: User = Depends(get_current_user),
) -> dict:
    """
    Revoke (disable) an API key.

    **Requires**: API authentication (X-API-Key header)
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    success = await APIKeyManager.revoke_key(UUID(key_id))
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )

    logger.info(f"User {current_user.id} revoked key {key_id}")
    return {"message": "API key revoked"}


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user),
) -> UserResponse:
    """
    Get current authenticated user info.

    **Requires**: API authentication (X-API-Key header)
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    return UserResponse(
        id=str(current_user.id),
        email=current_user.email,
        name=current_user.name,
        is_active=current_user.is_active,
        created_at=current_user.created_at.isoformat(),
    )
