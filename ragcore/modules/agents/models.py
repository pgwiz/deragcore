"""Agent chain models and definitions."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID
from sqlalchemy import Column, String, DateTime, Integer, Boolean, JSON, ForeignKey, Text, Index
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, ARRAY
import uuid
import json

from ragcore.db.database import Base


class AgentDefinition(Base):
    """Defines a single agent in the system."""

    __tablename__ = "agent_definitions"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=False)

    # Agent capabilities
    tools = Column(ARRAY(String), default=list)  # Tool names: "search_web", "search_docs", etc.
    model_config_id = Column(String(255), nullable=False)  # "claude-3-5-sonnet"
    system_prompt = Column(Text, nullable=False)

    # Execution parameters
    temperature = Column(Integer, default=7)  # 0-10 scale (* 0.1 = actual temp)
    max_tokens = Column(Integer, default=2048)
    timeout_seconds = Column(Integer, default=300)

    # Metadata
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ChainDefinition(Base):
    """Defines a multi-agent workflow."""

    __tablename__ = "chain_definitions"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=False)

    # Chain structure
    chain_type = Column(String(32), nullable=False)  # "sequential" | "parallel" | "conditional" | "recursive"

    # Agent configuration in chain order
    agents = Column(JSON, default=list)  # [{"agent_name": "...", "order": 1, "config": {...}}]

    # Routing for conditional chains
    routing_rules = Column(JSON, nullable=True)  # {condition_logic: ...}

    # How to combine results from parallel agents
    aggregation_strategy = Column(String(32), default="concat")  # "concat" | "merge" | "vote"

    # Chain execution limits
    max_iterations = Column(Integer, default=3)
    timeout_seconds = Column(Integer, default=900)

    # Metadata
    is_active = Column(Boolean, default=True)
    version = Column(Integer, default=1)  # For chain versioning
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ChainExecution(Base):
    """Records chain execution history."""

    __tablename__ = "chain_executions"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chain_id = Column(PG_UUID(as_uuid=True), ForeignKey("chain_definitions.id"), nullable=False)
    session_id = Column(PG_UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False, index=True)
    user_id = Column(PG_UUID(as_uuid=True), nullable=True)

    # Execution state
    status = Column(String(32), default="pending")  # "pending" | "running" | "completed" | "failed"
    steps_completed = Column(Integer, default=0)
    total_steps = Column(Integer, nullable=True)

    # Execution details
    input_query = Column(Text, nullable=False)  # Original user query
    execution_trace = Column(JSON, default=dict)  # Each step: {agent, input, output, tokens, duration}
    final_result = Column(JSON, nullable=True)  # Final aggregated result

    # Metadata
    error_message = Column(Text, nullable=True)
    total_tokens_used = Column(Integer, default=0)
    total_cost_units = Column(Integer, default=0)

    # Timing
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Indexing
    __table_args__ = (
        Index("ix_chain_execution_session", "session_id"),
        Index("ix_chain_execution_status", "status"),
    )


class ExecutionStep(Base):
    """Individual step within a chain execution."""

    __tablename__ = "execution_steps"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    execution_id = Column(
        PG_UUID(as_uuid=True), ForeignKey("chain_executions.id"), nullable=False, index=True
    )

    # Step details
    step_number = Column(Integer, nullable=False)
    agent_name = Column(String(255), nullable=False)

    # I/O
    input_data = Column(JSON, nullable=False)  # What the agent received
    output_data = Column(JSON, nullable=True)  # What the agent produced

    # Metadata
    status = Column(String(32), default="pending")  # "pending" | "running" | "completed" | "failed"
    error_message = Column(Text, nullable=True)
    tokens_used = Column(Integer, default=0)
    cost_units = Column(Integer, default=0)

    # Timing
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    duration_ms = Column(Integer, nullable=True)

    # Tool invocations during this step
    tool_calls = Column(JSON, default=list)  # [{tool_name, input, output, status}]


class ChainTemplate(Base):
    """Pre-built chain templates for common scenarios."""

    __tablename__ = "chain_templates"

    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), unique=True, nullable=False)
    description = Column(Text, nullable=False)
    category = Column(String(64), nullable=False)  # "research", "analysis", "qa", "extraction"

    # Copy of chain definition structure
    chain_type = Column(String(32), nullable=False)
    agents = Column(JSON, nullable=False)
    routing_rules = Column(JSON, nullable=True)
    aggregation_strategy = Column(String(32), default="concat")

    # Usage stats
    usage_count = Column(Integer, default=0)
    avg_success_rate = Column(Integer, default=0)  # 0-100

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Pydantic request/response models imported in router
