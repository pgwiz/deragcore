"""Database migration 005 - ChromaDB sync state and queue tables."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create ChromaDB tracking and sync tables."""
    # ChromaDB sync state table
    op.create_table(
        "chroma_sync_state",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("deployment_mode", sa.String(32), nullable=False),
        sa.Column("last_full_sync_at", sa.DateTime(), nullable=True),
        sa.Column("sync_queue_size", sa.Integer(), default=0),
        sa.Column("chroma_available", sa.Boolean(), default=True),
        sa.Column("last_chroma_check", sa.DateTime(), nullable=True),
        sa.Column("consecutive_failures", sa.Integer(), default=0),
        sa.Column("circuit_breaker_until", sa.DateTime(), nullable=True),
        sa.Column("avg_chroma_latency_ms", sa.Float(), nullable=True),
        sa.Column("avg_pgvector_latency_ms", sa.Float(), nullable=True),
        sa.Column("chroma_preferred", sa.Boolean(), default=False),
        sa.Column("created_at", sa.DateTime(), default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), default=sa.func.now(), onupdate=sa.func.now()),
        sa.ForeignKeyConstraint(["session_id"], ["session.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        "ix_chroma_sync_session",
        "chroma_sync_state",
        ["session_id"],
    )

    # ChromaDB sync queue table
    op.create_table(
        "chroma_sync_queue",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("memory_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("operation", sa.String(32), nullable=False),  # insert, update, delete
        sa.Column("retry_count", sa.Integer(), default=0),
        sa.Column("max_retries", sa.Integer(), default=3),
        sa.Column("last_retry_at", sa.DateTime(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), default=sa.func.now()),
        sa.Column("scheduled_for", sa.DateTime(), default=sa.func.now()),
        sa.ForeignKeyConstraint(["memory_id"], ["long_term_memory.id"]),
        sa.ForeignKeyConstraint(["session_id"], ["session.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        "ix_chroma_sync_queue_status",
        "chroma_sync_queue",
        ["memory_id", "operation"],
    )

    op.create_index(
        "ix_chroma_sync_queue_scheduled",
        "chroma_sync_queue",
        ["scheduled_for"],
    )


def downgrade() -> None:
    """Drop ChromaDB tracking and sync tables."""
    op.drop_index("ix_chroma_sync_queue_scheduled", table_name="chroma_sync_queue")
    op.drop_index("ix_chroma_sync_queue_status", table_name="chroma_sync_queue")
    op.drop_table("chroma_sync_queue")

    op.drop_index("ix_chroma_sync_session", table_name="chroma_sync_state")
    op.drop_table("chroma_sync_state")
