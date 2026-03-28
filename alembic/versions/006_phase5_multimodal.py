"""Sprint 5: Multi-Modal Support - Create tables for image, audio, video processing.

Revision ID: 006_phase5_multimodal
Revises: 005_phase5_chroma
Create Date: 2026-03-28 10:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# Revision identifiers
revision = "006_phase5_multimodal"
down_revision = "005_phase5_chroma"
branch_labels = None
depends_on = None


def upgrade():
    """Create multi-modal tables."""
    # MultiModalContent table - stores raw content and processing state
    op.create_table(
        "multimodal_content",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("session_id", sa.UUID(), nullable=False),
        sa.Column("modality", sa.String(10), nullable=False),  # image, audio, video, text
        sa.Column("raw_content", sa.LargeBinary(), nullable=True),  # Can be large
        sa.Column("text_content", sa.Text(), nullable=True),  # Extracted or original text
        sa.Column(
            "metadata",
            postgresql.JSON(),
            nullable=False,
            server_default=sa.literal("{}").compiled,
        ),
        sa.Column("is_processed", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("processing_error", sa.Text(), nullable=True),
        sa.Column("storage_path", sa.String(500), nullable=True),  # S3/Blob path
        sa.Column("is_base64_inline", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.id"],
            ondelete="CASCADE",
        ),
        sa.Index("idx_multimodal_session_modality", "session_id", "modality"),
        sa.Index("idx_multimodal_is_processed", "is_processed"),
    )

    # MultiModalChunk table - extracted chunks from content
    op.create_table(
        "multimodal_chunks",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("session_id", sa.UUID(), nullable=False),
        sa.Column("content_id", sa.UUID(), nullable=False),
        sa.Column("modality", sa.String(10), nullable=False),  # image, audio, video, text
        sa.Column("content", sa.Text(), nullable=False),  # Extracted text
        sa.Column(
            "embedding",
            postgresql.VECTOR(1536),
            nullable=True,
        ),  # Universal 1536-dim embedding
        sa.Column(
            "metadata",
            postgresql.JSON(),
            nullable=False,
            server_default=sa.literal("{}").compiled,
        ),
        sa.Column("source_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("confidence_score", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("is_critical", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["content_id"],
            ["multimodal_content.id"],
            ondelete="CASCADE",
        ),
        sa.Index("idx_multimodal_chunks_session", "session_id"),
        sa.Index("idx_multimodal_chunks_content", "content_id"),
        sa.Index("idx_multimodal_chunks_embedding", "embedding", postgresql_using="ivfflat"),
        sa.Index("idx_multimodal_chunks_confidence", "confidence_score"),
    )

    # MultiModalProcessingLog table - track processing history
    op.create_table(
        "multimodal_processing_log",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("content_id", sa.UUID(), nullable=False),
        sa.Column("session_id", sa.UUID(), nullable=False),
        sa.Column("processor_type", sa.String(50), nullable=False),  # ImageProcessor, AudioProcessor, etc
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("chunks_extracted", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("processing_time_ms", sa.Float(), nullable=False, server_default="0"),
        sa.Column("tokens_used", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "extraction_method",
            sa.String(50),
            nullable=True,
        ),  # claude_vision, azure_speech, whisper, etc
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["content_id"],
            ["multimodal_content.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["sessions.id"],
            ondelete="CASCADE",
        ),
        sa.Index("idx_processing_log_content", "content_id"),
        sa.Index("idx_processing_log_session", "session_id"),
        sa.Index("idx_processing_log_success", "success"),
    )


def downgrade():
    """Drop multi-modal tables."""
    op.drop_table("multimodal_processing_log")
    op.drop_table("multimodal_chunks")
    op.drop_table("multimodal_content")
