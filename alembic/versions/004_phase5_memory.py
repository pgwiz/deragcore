"""Sprint 2 Phase 5: Long-Term Memory System

Revision ID: 004
Revises: 003
Create Date: 2026-03-28 11:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision = '004'
down_revision = '003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create memory tables."""

    # Create long_term_memory table
    op.create_table(
        'long_term_memory',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.func.gen_random_uuid()),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True, index=True),
        sa.Column('memory_type', sa.String(32), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('summary', sa.Text(), nullable=True),
        sa.Column('embedding', Vector(1536), nullable=True),
        sa.Column('source', sa.String(255), nullable=True),
        sa.Column('tags', postgresql.ARRAY(sa.String()), server_default='{}'),
        sa.Column('context_data', sa.JSON(), server_default='{}'),
        sa.Column('importance_score', sa.Float(), server_default='0.5'),
        sa.Column('relevance_score', sa.Float(), server_default='0.5'),
        sa.Column('access_count', sa.Integer(), server_default='0'),
        sa.Column('ttl_seconds', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('last_accessed_at', sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('expires_at', sa.DateTime(), nullable=True, index=True),
        sa.Column('is_active', sa.Boolean(), server_default='true', index=True),
        sa.Column('is_shared', sa.Boolean(), server_default='false'),
        sa.ForeignKeyConstraint(['session_id'], ['sessions.id'], ),
        sa.PrimaryKeyConstraint('id'),
    )

    # Create episodic_snapshots table
    op.create_table(
        'episodic_snapshots',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.func.gen_random_uuid()),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('episode_number', sa.Integer(), nullable=False),
        sa.Column('episode_type', sa.String(32), nullable=False),
        sa.Column('title', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('input_query', sa.Text(), nullable=False),
        sa.Column('output_summary', sa.Text(), nullable=True),
        sa.Column('key_findings', sa.JSON(), server_default='[]'),
        sa.Column('sources_used', postgresql.ARRAY(sa.String()), server_default='{}'),
        sa.Column('tools_invoked', sa.JSON(), server_default='[]'),
        sa.Column('actions_taken', sa.JSON(), server_default='[]'),
        sa.Column('related_memories', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), server_default='{}'),
        sa.Column('duration_ms', sa.Integer(), nullable=True),
        sa.Column('tokens_used', sa.Integer(), server_default='0'),
        sa.Column('success', sa.Boolean(), server_default='true'),
        sa.Column('tags', postgresql.ARRAY(sa.String()), server_default='{}'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('expires_at', sa.DateTime(), nullable=True, index=True),
        sa.ForeignKeyConstraint(['session_id'], ['sessions.id'], ),
        sa.PrimaryKeyConstraint('id'),
    )

    # Create memory_access_logs table
    op.create_table(
        'memory_access_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.func.gen_random_uuid()),
        sa.Column('memory_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('access_type', sa.String(32), nullable=False),
        sa.Column('query', sa.Text(), nullable=True),
        sa.Column('similarity_score', sa.Float(), nullable=True),
        sa.Column('context', sa.JSON(), server_default='{}'),
        sa.Column('timestamp', sa.DateTime(), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['memory_id'], ['long_term_memory.id'], ),
        sa.ForeignKeyConstraint(['session_id'], ['sessions.id'], ),
        sa.PrimaryKeyConstraint('id'),
    )

    # Create memory_cleanup_tasks table
    op.create_table(
        'memory_cleanup_tasks',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.func.gen_random_uuid()),
        sa.Column('task_type', sa.String(32), nullable=False),
        sa.Column('status', sa.String(32), nullable=False, index=True),
        sa.Column('items_processed', sa.Integer(), server_default='0'),
        sa.Column('items_deleted', sa.Integer(), server_default='0'),
        sa.Column('duration_ms', sa.Integer(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('scheduled_at', sa.DateTime(), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )

    # Create indexes
    op.create_index('ix_long_term_memory_session', 'long_term_memory', ['session_id'])
    op.create_index('ix_long_term_memory_type', 'long_term_memory', ['memory_type'])
    op.create_index('ix_long_term_memory_user', 'long_term_memory', ['user_id'])
    op.create_index('ix_long_term_memory_expires', 'long_term_memory', ['expires_at'])
    op.create_index('ix_long_term_memory_active', 'long_term_memory', ['is_active'])

    op.create_index('ix_episodic_snapshot_session', 'episodic_snapshots', ['session_id'])
    op.create_index('ix_episodic_snapshot_user', 'episodic_snapshots', ['user_id'])
    op.create_index('ix_episodic_snapshot_type', 'episodic_snapshots', ['episode_type'])
    op.create_index('ix_episodic_snapshot_expires', 'episodic_snapshots', ['expires_at'])

    op.create_index('ix_memory_access_log_memory', 'memory_access_logs', ['memory_id'])
    op.create_index('ix_memory_access_log_session', 'memory_access_logs', ['session_id'])
    op.create_index('ix_memory_access_log_timestamp', 'memory_access_logs', ['timestamp'])

    op.create_index('ix_memory_cleanup_task_status', 'memory_cleanup_tasks', ['status'])
    op.create_index('ix_memory_cleanup_task_scheduled', 'memory_cleanup_tasks', ['scheduled_at'])


def downgrade() -> None:
    """Drop memory tables."""
    op.drop_index('ix_memory_cleanup_task_scheduled', 'memory_cleanup_tasks')
    op.drop_index('ix_memory_cleanup_task_status', 'memory_cleanup_tasks')

    op.drop_index('ix_memory_access_log_timestamp', 'memory_access_logs')
    op.drop_index('ix_memory_access_log_session', 'memory_access_logs')
    op.drop_index('ix_memory_access_log_memory', 'memory_access_logs')

    op.drop_index('ix_episodic_snapshot_expires', 'episodic_snapshots')
    op.drop_index('ix_episodic_snapshot_type', 'episodic_snapshots')
    op.drop_index('ix_episodic_snapshot_user', 'episodic_snapshots')
    op.drop_index('ix_episodic_snapshot_session', 'episodic_snapshots')

    op.drop_index('ix_long_term_memory_active', 'long_term_memory')
    op.drop_index('ix_long_term_memory_expires', 'long_term_memory')
    op.drop_index('ix_long_term_memory_user', 'long_term_memory')
    op.drop_index('ix_long_term_memory_type', 'long_term_memory')
    op.drop_index('ix_long_term_memory_session', 'long_term_memory')

    op.drop_table('memory_cleanup_tasks')
    op.drop_table('memory_access_logs')
    op.drop_table('episodic_snapshots')
    op.drop_table('long_term_memory')
