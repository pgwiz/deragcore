"""SQLAlchemy ORM models for RAGCORE."""

import uuid
from datetime import datetime
from sqlalchemy import String, Float, Integer, Boolean, DateTime, JSON, ForeignKey, Text, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID, BYTEA
from pgvector.sqlalchemy import Vector
from ragcore.db.database import Base
from typing import Optional, Dict, Any, List


class ModelConfig(Base):
    """Configuration for AI models (which provider, which model, params)."""

    __tablename__ = "model_config"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    """Human-readable name e.g. 'fast-claude', 'research-phi4'"""

    provider: Mapped[str] = mapped_column(
        String(50), index=True
    )
    """Provider name: 'anthropic' | 'azure' | 'openai' | 'ollama'"""

    model_id: Mapped[str] = mapped_column(String(255))
    """Exact model string sent to provider API"""

    temperature: Mapped[float] = mapped_column(Float, default=0.7)
    """Sampling temperature 0.0-2.0"""

    max_tokens: Mapped[int] = mapped_column(Integer, default=2048)
    """Maximum tokens to generate"""

    system_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    """Optional per-config system instruction"""

    is_default: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    """Fallback when session has no explicit config"""

    extra: Mapped[Dict[str, Any]] = mapped_column(JSON, default={})
    """Provider-specific extras e.g. api_version"""

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    sessions: Mapped[list["Session"]] = relationship("Session", back_populates="model_config")

    def __repr__(self) -> str:
        return f"<ModelConfig {self.name} ({self.provider}/{self.model_id})>"


class Session(Base):
    """Chat session with model configuration and history."""

    __tablename__ = "session"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    model_config_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("model_config.id"), nullable=True
    )
    """Foreign key to ModelConfig - if None, use default"""

    title: Mapped[str] = mapped_column(String(255), default="Unnamed Session")
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    model_config: Mapped[Optional[ModelConfig]] = relationship("ModelConfig", back_populates="sessions")
    files: Mapped[List["File"]] = relationship("File", back_populates="session")

    def __repr__(self) -> str:
        return f"<Session {self.id} - {self.title}>"


class Job(Base):
    """Background job tracking (file processing, research, etc)."""

    __tablename__ = "job"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    job_type: Mapped[str] = mapped_column(String(50), index=True)
    """Job type: 'file_process' | 'research' | 'embedding'"""

    status: Mapped[str] = mapped_column(
        String(50), default="pending", index=True
    )
    """Status: 'pending' | 'running' | 'completed' | 'failed'"""

    result: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    """Job result or final data"""

    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    """Error message if job failed"""

    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    def __repr__(self) -> str:
        return f"<Job {self.id} - {self.job_type}({self.status})>"


class File(Base):
    """Uploaded document metadata and processing status."""

    __tablename__ = "file"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    filename: Mapped[str] = mapped_column(String(512), index=True)
    """Original filename"""

    file_size: Mapped[int] = mapped_column(Integer)
    """File size in bytes"""

    content_type: Mapped[str] = mapped_column(String(64))
    """MIME type: 'application/pdf' | 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'"""

    status: Mapped[str] = mapped_column(
        String(50), default="pending", index=True
    )
    """Status: 'pending' | 'parsing' | 'chunking' | 'embedding' | 'ready' | 'failed'"""

    chunks_count: Mapped[int] = mapped_column(Integer, default=0)
    """Number of chunks created from this file"""

    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    """Error description if status is 'failed'"""

    session_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True), ForeignKey("session.id"), nullable=True, index=True
    )
    """Optional: session scope for this file"""

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    session: Mapped[Optional["Session"]] = relationship("Session", back_populates="files")
    chunks: Mapped[List["Chunk"]] = relationship("Chunk", back_populates="file", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<File {self.filename} ({self.status}) - {self.chunks_count} chunks>"


class Chunk(Base):
    """Document chunk with embedding vector for RAG retrieval."""

    __tablename__ = "chunk"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    file_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("file.id"), index=True
    )
    """Foreign key to parent File"""

    chunk_index: Mapped[int] = mapped_column(Integer)
    """Order of chunk within file (0-based)"""

    text: Mapped[str] = mapped_column(Text)
    """Extracted and cleaned text content"""

    embedding: Mapped[Optional[Vector]] = mapped_column(Vector(1536), nullable=True)
    """Vector embedding (1536 dims for OpenAI/Azure models, 768 for Anthropic)"""

    tokens: Mapped[int] = mapped_column(Integer, default=0)
    """Approximate token count for this chunk"""

    metadata_: Mapped[Dict[str, Any]] = mapped_column(JSON, default={}, name="metadata")
    """Source metadata: {'page': int, 'section': str, 'offset': int} etc."""

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    file: Mapped["File"] = relationship("File", back_populates="chunks")

    # Index for vector similarity search
    __table_args__ = (
        Index("ix_chunk_embedding", "embedding", postgresql_using="ivfflat", postgresql_with={"lists": 100}),
    )

    def __repr__(self) -> str:
        return f"<Chunk {self.id[:8]} from {self.file_id[:8]} - {self.tokens} tokens>"
