"""Webhook management and dispatch."""

import logging
import secrets
import hmac
import hashlib
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime

import httpx
from sqlalchemy import select

from ragcore.config import settings
from ragcore.db.database import async_session_factory
from ragcore.webhooks.models import Webhook, WebhookEvent, SUPPORTED_EVENTS

logger = logging.getLogger(__name__)


class WebhookManager:
    """Manage webhook registration, delivery, and retry logic."""

    @staticmethod
    async def create_webhook(
        user_id: UUID,
        url: str,
        events: List[str],
    ) -> Webhook:
        """
        Register new webhook for user.

        Args:
            user_id: User ID
            url: HTTPS endpoint URL
            events: List of event types to subscribe to

        Returns:
            Created webhook record
        """
        # Validate URL
        if not url.startswith("https://"):
            raise ValueError("Webhook URL must use HTTPS")

        # Validate events
        invalid_events = set(events) - set(SUPPORTED_EVENTS)
        if invalid_events:
            raise ValueError(f"Invalid event types: {invalid_events}")

        # Generate secret for HMAC signing
        secret = secrets.token_urlsafe(32)

        async with async_session_factory() as session:
            webhook = Webhook(
                user_id=user_id,
                url=url,
                events=events,
                secret=secret,
            )
            session.add(webhook)
            await session.commit()
            await session.refresh(webhook)

            logger.info(f"Created webhook for user {user_id}: {webhook.id}")
            return webhook

    @staticmethod
    async def list_webhooks(user_id: UUID) -> List[Webhook]:
        """List all webhooks for user."""
        async with async_session_factory() as session:
            stmt = select(Webhook).where(Webhook.user_id == user_id)
            result = await session.execute(stmt)
            return result.scalars().all()

    @staticmethod
    async def delete_webhook(webhook_id: UUID) -> bool:
        """Delete webhook."""
        async with async_session_factory() as session:
            stmt = select(Webhook).where(Webhook.id == webhook_id)
            result = await session.execute(stmt)
            webhook = result.scalar_one_or_none()

            if not webhook:
                return False

            await session.delete(webhook)
            await session.commit()
            logger.info(f"Deleted webhook {webhook_id}")
            return True

    @staticmethod
    def _sign_payload(payload: str, secret: str) -> str:
        """
        Sign payload with HMAC-SHA256.

        Args:
            payload: JSON payload string
            secret: Webhook secret

        Returns:
            HMAC signature (hex)
        """
        return hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    async def dispatch_event(
        event_type: str,
        payload: Dict[str, Any],
        user_id: Optional[UUID] = None,
    ) -> None:
        """
        Send event to all subscribed webhooks.

        Args:
            event_type: Event type (from WebhookEventType)
            payload: Event payload data
            user_id: Optional user ID to filter webhooks
        """
        if not settings.webhook_enabled:
            return

        async with async_session_factory() as session:
            # Find subscribed webhooks
            stmt = select(Webhook).where(Webhook.is_active == True)
            if user_id:
                stmt = stmt.where(Webhook.user_id == user_id)

            result = await session.execute(stmt)
            webhooks = result.scalars().all()

            for webhook in webhooks:
                if event_type not in webhook.events:
                    continue

                # Create event record
                webhook_event = WebhookEvent(
                    webhook_id=webhook.id,
                    event_type=event_type,
                    payload={
                        "event": event_type,
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": payload,
                    },
                    status="pending",
                )
                session.add(webhook_event)
                await session.commit()

                # Schedule delivery
                asyncio.create_task(
                    WebhookManager._deliver_event(webhook, webhook_event)
                )

    @staticmethod
    async def _deliver_event(webhook: Webhook, event: WebhookEvent) -> None:
        """
        Deliver webhook event with retry logic.

        Args:
            webhook: Webhook to deliver to
            event: Event to send
        """
        import json

        payload_json = json.dumps(event.payload)
        signature = WebhookManager._sign_payload(payload_json, webhook.secret)

        async with async_session_factory() as session:
            # Refresh event
            stmt = select(WebhookEvent).where(WebhookEvent.id == event.id)
            result = await session.execute(stmt)
            event = result.scalar_one()

            for attempt in range(settings.webhook_max_retries):
                try:
                    async with httpx.AsyncClient(timeout=settings.webhook_timeout_seconds) as client:
                        response = await client.post(
                            webhook.url,
                            json=event.payload,
                            headers={
                                "X-Webhook-Signature": signature,
                                "X-Webhook-Event": event.event_type,
                                "Content-Type": "application/json",
                            },
                        )

                        if response.status_code in (200, 201, 202, 204):
                            event.status = "sent"
                            event.sent_at = datetime.utcnow()
                            event.attempts = attempt + 1
                            webhook.last_triggered_at = datetime.utcnow()
                            session.add(event)
                            session.add(webhook)
                            await session.commit()
                            logger.info(
                                f"Webhook delivered: {webhook.id} "
                                f"({event.event_type}) on attempt {attempt + 1}"
                            )
                            return

                        event.last_error = f"HTTP {response.status_code}"

                except Exception as e:
                    event.last_error = str(e)
                    logger.error(
                        f"Webhook delivery failed (attempt {attempt + 1}/"
                        f"{settings.webhook_max_retries}): {str(e)}"
                    )

                    # Retry with exponential backoff
                    if attempt < settings.webhook_max_retries - 1:
                        await asyncio.sleep(
                            settings.webhook_retry_backoff_seconds ** (attempt + 1)
                        )

            # All retries exhausted
            event.status = "failed"
            event.attempts = settings.webhook_max_retries
            session.add(event)
            await session.commit()

            logger.error(
                f"Webhook delivery permanently failed after {settings.webhook_max_retries} "
                f"attempts: {webhook.id} ({event.event_type})"
            )


webhook_manager = WebhookManager()
