"""Webhook system for event notifications."""

import logging
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any
from uuid import UUID
from sqlalchemy import Column, String, DateTime, Boolean, Integer, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
import uuid

from ragcore.db.database import Base

logger = logging.getLogger(__name__)


class Webhook(Base):
    """Registered webhook endpoint for event notifications."""

    __tablename__ = "webhooks"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)

    # Webhook configuration
    url = Column(String(512), nullable=False)  # HTTPS only
    events = Column(JSON, default=list)  # List of event types to subscribe to
    secret = Column(String(64), nullable=False)  # For HMAC signature verification
    is_active = Column(Boolean, default=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_triggered_at = Column(DateTime, nullable=True)


class WebhookEvent(Base):
    """Event delivery record."""

    __tablename__ = "webhook_events"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    webhook_id = Column(
        PG_UUID(as_uuid=True), ForeignKey("webhooks.id"), nullable=False
    )

    # Event details
    event_type = Column(String(64), nullable=False)  # research_complete, error, etc.
    payload = Column(JSON, nullable=False)  # Event data

    # Delivery tracking
    status = Column(String(32), default="pending")  # pending, sent, failed
    attempts = Column(Integer, default=0)
    last_error = Column(String(512), nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    sent_at = Column(DateTime, nullable=True)


# Webhook event types
class WebhookEventType:
    """Webhook event type constants."""

    RESEARCH_COMPLETE = "research.complete"
    RESEARCH_FAILED = "research.failed"
    CHAT_RESPONSE = "chat.response_sent"
    ERROR = "system.error"
    RATE_LIMIT = "system.rate_limit_exceeded"
    QUOTA_EXCEEDED = "system.quota_exceeded"


# Supported events
SUPPORTED_EVENTS = [
    WebhookEventType.RESEARCH_COMPLETE,
    WebhookEventType.RESEARCH_FAILED,
    WebhookEventType.CHAT_RESPONSE,
    WebhookEventType.ERROR,
    WebhookEventType.RATE_LIMIT,
    WebhookEventType.QUOTA_EXCEEDED,
]
