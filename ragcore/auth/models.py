"""Authentication data models."""

from datetime import datetime
from uuid import UUID
from typing import Optional
from sqlalchemy import Column, String, DateTime, Integer, Boolean, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
import uuid

from ragcore.db.database import Base


class User(Base):
    """API user account."""

    __tablename__ = "users"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class APIKey(Base):
    """API key for authentication."""

    __tablename__ = "api_keys"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    key = Column(String(64), unique=True, nullable=False, index=True)  # hashed key
    name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)

    # Rate limiting & quotas
    requests_today = Column(Integer, default=0)
    requests_month = Column(Integer, default=0)
    last_reset_date = Column(DateTime, nullable=True)

    # Metadata
    last_used_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)  # None = no expiration


class AuditLog(Base):
    """Audit trail for security & compliance."""

    __tablename__ = "audit_logs"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    api_key_id = Column(PG_UUID(as_uuid=True), ForeignKey("api_keys.id"), nullable=True)

    # Request details
    method = Column(String(10), nullable=False)  # GET, POST, etc.
    path = Column(String(512), nullable=False)
    status_code = Column(Integer, nullable=False)

    # Metadata
    ip_address = Column(String(45), nullable=True)  # IPv4 or IPv6
    user_agent = Column(String(512), nullable=True)
    response_time_ms = Column(Integer, nullable=True)

    # Error tracking
    error = Column(String(512), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)
