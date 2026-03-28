"""Initial schema with ModelConfig, Session, Job, File, Chunk

Revision ID: 001
Revises:
Create Date: 2026-03-27 16:20:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS pgvector')

    # Create model_config table
    op.create_table(
        'model_config',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('provider', sa.String(50), nullable=False),
        sa.Column('model_id', sa.String(255), nullable=False),
        sa.Column('temperature', sa.Float(), nullable=False, server_default='0.7'),
        sa.Column('max_tokens', sa.Integer(), nullable=False, server_default='2048'),
        sa.Column('system_prompt', sa.Text(), nullable=True),
        sa.Column('is_default', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('extra', postgresql.JSON(), nullable=False, server_default='{}'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_model_config')),
        sa.UniqueConstraint('name', name=op.f('uq_model_config_name')),
    )
    op.create_index(op.f('ix_model_config_name'), 'model_config', ['name'], unique=True)
    op.create_index(op.f('ix_model_config_provider'), 'model_config', ['provider'])
    op.create_index(op.f('ix_model_config_is_default'), 'model_config', ['is_default'])

    # Create session table
    op.create_table(
        'session',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('model_config_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('title', sa.String(255), nullable=False, server_default='Unnamed Session'),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['model_config_id'], ['model_config.id'], name=op.f('fk_session_model_config_id')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_session')),
    )
    op.create_index(op.f('ix_session_created_at'), 'session', ['created_at'])

    # Create job table
    op.create_table(
        'job',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('job_type', sa.String(50), nullable=False),
        sa.Column('status', sa.String(50), nullable=False, server_default='pending'),
        sa.Column('result', postgresql.JSON(), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('retry_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('max_retries', sa.Integer(), nullable=False, server_default='3'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_job')),
    )
    op.create_index(op.f('ix_job_job_type'), 'job', ['job_type'])
    op.create_index(op.f('ix_job_status'), 'job', ['status'])
    op.create_index(op.f('ix_job_created_at'), 'job', ['created_at'])

    # Create file table
    op.create_table(
        'file',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('filename', sa.String(512), nullable=False),
        sa.Column('file_size', sa.Integer(), nullable=False),
        sa.Column('content_type', sa.String(64), nullable=False),
        sa.Column('status', sa.String(50), nullable=False, server_default='pending'),
        sa.Column('chunks_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['session_id'], ['session.id'], name=op.f('fk_file_session_id')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_file')),
    )
    op.create_index(op.f('ix_file_filename'), 'file', ['filename'])
    op.create_index(op.f('ix_file_status'), 'file', ['status'])
    op.create_index(op.f('ix_file_session_id'), 'file', ['session_id'])
    op.create_index(op.f('ix_file_created_at'), 'file', ['created_at'])

    # Create chunk table
    op.create_table(
        'chunk',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('file_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('embedding', Vector(dim=1536), nullable=True),
        sa.Column('tokens', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('metadata', postgresql.JSON(), nullable=False, server_default='{}'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['file_id'], ['file.id'], name=op.f('fk_chunk_file_id')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_chunk')),
    )
    op.create_index(op.f('ix_chunk_file_id'), 'chunk', ['file_id'])
    # Create IVFFlat index for vector similarity search
    op.execute(
        """
        CREATE INDEX ix_chunk_embedding ON chunk USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
        """
    )


def downgrade() -> None:
    op.drop_table('chunk')
    op.drop_table('file')
    op.drop_table('job')
    op.drop_table('session')
    op.drop_table('model_config')
