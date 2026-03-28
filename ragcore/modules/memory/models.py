"""Long-term memory models for multi-session persistence."""

from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from uuid import UUID
from sqlalchemy import Column, String, DateTime, Integer, Boolean, JSON, ForeignKey, Text, Index, Float
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, ARRAY
from pgvector.sqlalchemy import Vector
import uuid

from ragcore.db.database import Base


class LongTermMemory(Base):
    """Persistent memory entries across sessions."""

    __tablename__ = "long_term_memory"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(PG_UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False, index=True)
    user_id = Column(PG_UUID(as_uuid=True), nullable=True, index=True)

    # Memory content
    memory_type = Column(String(32), nullable=False)  # "finding" | "decision" | "insight" | "error"
    content = Column(Text(), nullable=False)
    summary = Column(Text(), nullable=True)  # Short summary for quick retrieval

    # Semantic embedding for similarity search
    embedding = Column(Vector(1536), nullable=True)  # Claude embeddings are 1536 dims

    # Metadata
    source = Column(String(255), nullable=True)  # Origin: "research", "analysis", "user_input"
    tags = Column(ARRAY(String), default=list)  # For categorization
    context_data = Column(JSON(), default=dict)  # Additional context

    # Importance and relevance
    importance_score = Column(Float(), default=0.5)  # 0.0-1.0 scale
    relevance_score = Column(Float(), default=0.5)  # Updated on retrieval
    access_count = Column(Integer(), default=0)  # Times retrieved

    # Lifecycle
    ttl_seconds = Column(Integer(), nullable=True)  # Time to live in seconds
    created_at = Column(DateTime(), default=datetime.utcnow)
    last_accessed_at = Column(DateTime(), default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at = Column(DateTime(), nullable=True, index=True)  # Auto-calculated on insert

    # Status
    is_active = Column(Boolean(), default=True, index=True)
    is_shared = Column(Boolean(), default=False)  # Shared across users

    __table_args__ = (
        Index("ix_long_term_memory_session", "session_id"),
        Index("ix_long_term_memory_type", "memory_type"),
        Index("ix_long_term_memory_user", "user_id"),
        Index("ix_long_term_memory_expires", "expires_at"),
        Index("ix_long_term_memory_active", "is_active"),
    )


class EpisodicSnapshot(Base):
    """Record of a research episode or conversation turn."""

    __tablename__ = "episodic_snapshots"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(PG_UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False, index=True)
    user_id = Column(PG_UUID(as_uuid=True), nullable=True)

    # Episode details
    episode_number = Column(Integer(), nullable=False)  # Sequence within session
    episode_type = Column(String(32), nullable=False)  # "research" | "chat" | "analysis"
    title = Column(String(255), nullable=False)
    description = Column(Text(), nullable=False)

    # Content from the episode
    input_query = Column(Text(), nullable=False)  # What was asked
    output_summary = Column(Text(), nullable=True)  # What was found
    key_findings = Column(JSON(), default=list)  # [{finding, confidence, source}]
    sources_used = Column(ARRAY(String), default=list)  # [url, filename, etc]

    # Tools/actions taken
    tools_invoked = Column(JSON(), default=list)  # [{tool_name, params, result_summary}]
    actions_taken = Column(JSON(), default=list)  # Free-form action log

    # Related memories
    related_memories = Column(ARRAY(PG_UUID(as_uuid=True)), default=list)  # FK to LongTermMemory

    # Metadata
    duration_ms = Column(Integer(), nullable=True)
    tokens_used = Column(Integer(), default=0)
    success = Column(Boolean(), default=True)
    tags = Column(ARRAY(String), default=list)

    # Lifecycle
    created_at = Column(DateTime(), default=datetime.utcnow)
    expires_at = Column(DateTime(), nullable=True, index=True)  # 1 year TTL typical

    __table_args__ = (
        Index("ix_episodic_snapshot_session", "session_id"),
        Index("ix_episodic_snapshot_user", "user_id"),
        Index("ix_episodic_snapshot_type", "episode_type"),
        Index("ix_episodic_snapshot_expires", "expires_at"),
    )


class MemoryAccessLog(Base):
    """Log of memory retrievals for learning access patterns."""

    __tablename__ = "memory_access_logs"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    memory_id = Column(PG_UUID(as_uuid=True), ForeignKey("long_term_memory.id"), nullable=False, index=True)
    session_id = Column(PG_UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False, index=True)
    user_id = Column(PG_UUID(as_uuid=True), nullable=True)

    # Access details
    access_type = Column(String(32), nullable=False)  # "retrieval" | "update" | "delete"
    query = Column(Text(), nullable=True)  # What was searched for
    similarity_score = Column(Float(), nullable=True)  # If similarity search

    # Context
    context = Column(JSON(), default=dict)  # {retrieved_at, used_in, result}
    timestamp = Column(DateTime(), default=datetime.utcnow)

    __table_args__ = (
        Index("ix_memory_access_log_memory", "memory_id"),
        Index("ix_memory_access_log_session", "session_id"),
        Index("ix_memory_access_log_timestamp", "timestamp"),
    )


class MemoryCleanupTask(Base):
    """Track memory cleanup jobs."""

    __tablename__ = "memory_cleanup_tasks"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_type = Column(String(32), nullable=False)  # "expire_old" | "consolidate" | "vacuum"
    status = Column(String(32), nullable=False)  # "pending" | "running" | "completed" | "failed"

    # Execution
    items_processed = Column(Integer(), default=0)
    items_deleted = Column(Integer(), default=0)
    duration_ms = Column(Integer(), nullable=True)
    error_message = Column(Text(), nullable=True)

    scheduled_at = Column(DateTime(), nullable=False)
    started_at = Column(DateTime(), nullable=True)
    completed_at = Column(DateTime(), nullable=True)

    __table_args__ = (
        Index("ix_memory_cleanup_task_status", "status"),
        Index("ix_memory_cleanup_task_scheduled", "scheduled_at"),
    )
