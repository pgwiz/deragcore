"""Webhook management endpoints."""

import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, HttpUrl

from ragcore.auth.dependencies import get_current_user
from ragcore.auth.models import User
from ragcore.webhooks.manager import webhook_manager
from ragcore.webhooks.models import WebhookEventType, SUPPORTED_EVENTS, Webhook

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


# ============================================================================
# Request/Response Models
# ============================================================================


class WebhookResponse(BaseModel):
    """Webhook response."""

    id: str
    url: str
    events: List[str]
    is_active: bool
    created_at: str
    last_triggered_at: Optional[str]


class CreateWebhookRequest(BaseModel):
    """Create webhook request."""

    url: HttpUrl
    events: List[str]


class SupportedEventsResponse(BaseModel):
    """Supported events response."""

    events: List[str]


# ============================================================================
# Endpoints
# ============================================================================


@router.get("/events", response_model=SupportedEventsResponse)
async def get_supported_events() -> SupportedEventsResponse:
    """
    Get list of supported webhook event types.

    **Public** endpoint - no authentication required.
    """
    return SupportedEventsResponse(events=SUPPORTED_EVENTS)


@router.post("", response_model=WebhookResponse)
async def create_webhook(
    req: CreateWebhookRequest,
    current_user: User = Depends(get_current_user),
) -> WebhookResponse:
    """
    Register new webhook for current user.

    **Requires**: API authentication (X-API-Key header)
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    try:
        webhook = await webhook_manager.create_webhook(
            user_id=current_user.id,
            url=str(req.url),
            events=req.events,
        )

        return WebhookResponse(
            id=str(webhook.id),
            url=webhook.url,
            events=webhook.events,
            is_active=webhook.is_active,
            created_at=webhook.created_at.isoformat(),
            last_triggered_at=(
                webhook.last_triggered_at.isoformat()
                if webhook.last_triggered_at
                else None
            ),
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("", response_model=List[WebhookResponse])
async def list_webhooks(
    current_user: User = Depends(get_current_user),
) -> List[WebhookResponse]:
    """
    List all webhooks for current user.

    **Requires**: API authentication (X-API-Key header)
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    webhooks = await webhook_manager.list_webhooks(current_user.id)

    return [
        WebhookResponse(
            id=str(w.id),
            url=w.url,
            events=w.events,
            is_active=w.is_active,
            created_at=w.created_at.isoformat(),
            last_triggered_at=(
                w.last_triggered_at.isoformat() if w.last_triggered_at else None
            ),
        )
        for w in webhooks
    ]


@router.delete("/{webhook_id}")
async def delete_webhook(
    webhook_id: str,
    current_user: User = Depends(get_current_user),
) -> dict:
    """
    Delete webhook.

    **Requires**: API authentication (X-API-Key header)
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    success = await webhook_manager.delete_webhook(UUID(webhook_id))
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Webhook not found",
        )

    return {"message": "Webhook deleted"}
