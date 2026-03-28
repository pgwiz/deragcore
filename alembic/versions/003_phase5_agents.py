"""Sprint 1 Phase 5: Agent Chain Models

Revision ID: 003
Revises: 002
Create Date: 2026-03-28 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create agent and chain tables."""

    # Create agent_definitions table
    op.create_table(
        'agent_definitions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.func.gen_random_uuid()),
        sa.Column('name', sa.String(255), nullable=False, unique=True, index=True),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('tools', postgresql.ARRAY(sa.String()), server_default='{}'),
        sa.Column('model_config_id', sa.String(255), nullable=False),
        sa.Column('system_prompt', sa.Text(), nullable=False),
        sa.Column('temperature', sa.Integer(), server_default='7'),
        sa.Column('max_tokens', sa.Integer(), server_default='2048'),
        sa.Column('timeout_seconds', sa.Integer(), server_default='300'),
        sa.Column('is_active', sa.Boolean(), server_default='true'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
    )

    # Create chain_definitions table
    op.create_table(
        'chain_definitions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.func.gen_random_uuid()),
        sa.Column('name', sa.String(255), nullable=False, unique=True, index=True),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('chain_type', sa.String(32), nullable=False),
        sa.Column('agents', sa.JSON(), server_default='[]'),
        sa.Column('routing_rules', sa.JSON(), nullable=True),
        sa.Column('aggregation_strategy', sa.String(32), server_default='concat'),
        sa.Column('max_iterations', sa.Integer(), server_default='3'),
        sa.Column('timeout_seconds', sa.Integer(), server_default='900'),
        sa.Column('is_active', sa.Boolean(), server_default='true'),
        sa.Column('version', sa.Integer(), server_default='1'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
    )

    # Create chain_executions table
    op.create_table(
        'chain_executions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.func.gen_random_uuid()),
        sa.Column('chain_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('status', sa.String(32), server_default='pending'),
        sa.Column('steps_completed', sa.Integer(), server_default='0'),
        sa.Column('total_steps', sa.Integer(), nullable=True),
        sa.Column('input_query', sa.Text(), nullable=False),
        sa.Column('execution_trace', sa.JSON(), server_default='{}'),
        sa.Column('final_result', sa.JSON(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('total_tokens_used', sa.Integer(), server_default='0'),
        sa.Column('total_cost_units', sa.Integer(), server_default='0'),
        sa.Column('started_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['chain_id'], ['chain_definitions.id'], ),
        sa.ForeignKeyConstraint(['session_id'], ['sessions.id'], ),
        sa.PrimaryKeyConstraint('id'),
    )

    # Create execution_steps table
    op.create_table(
        'execution_steps',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.func.gen_random_uuid()),
        sa.Column('execution_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('step_number', sa.Integer(), nullable=False),
        sa.Column('agent_name', sa.String(255), nullable=False),
        sa.Column('input_data', sa.JSON(), nullable=False),
        sa.Column('output_data', sa.JSON(), nullable=True),
        sa.Column('status', sa.String(32), server_default='pending'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('tokens_used', sa.Integer(), server_default='0'),
        sa.Column('cost_units', sa.Integer(), server_default='0'),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('duration_ms', sa.Integer(), nullable=True),
        sa.Column('tool_calls', sa.JSON(), server_default='[]'),
        sa.ForeignKeyConstraint(['execution_id'], ['chain_executions.id'], ),
        sa.PrimaryKeyConstraint('id'),
    )

    # Create chain_templates table
    op.create_table(
        'chain_templates',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.func.gen_random_uuid()),
        sa.Column('name', sa.String(255), nullable=False, unique=True),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('category', sa.String(64), nullable=False),
        sa.Column('chain_type', sa.String(32), nullable=False),
        sa.Column('agents', sa.JSON(), nullable=False),
        sa.Column('routing_rules', sa.JSON(), nullable=True),
        sa.Column('aggregation_strategy', sa.String(32), server_default='concat'),
        sa.Column('usage_count', sa.Integer(), server_default='0'),
        sa.Column('avg_success_rate', sa.Integer(), server_default='0'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
    )

    # Create indexes
    op.create_index('ix_chain_execution_session', 'chain_executions', ['session_id'])
    op.create_index('ix_chain_execution_status', 'chain_executions', ['status'])


def downgrade() -> None:
    """Drop agent and chain tables."""
    op.drop_index('ix_chain_execution_status', 'chain_executions')
    op.drop_index('ix_chain_execution_session', 'chain_executions')
    op.drop_table('chain_templates')
    op.drop_table('execution_steps')
    op.drop_table('chain_executions')
    op.drop_table('chain_definitions')
    op.drop_table('agent_definitions')
