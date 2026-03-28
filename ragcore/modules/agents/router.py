"""HTTP router for agent chain operations."""

import logging
from typing import Optional, List
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field

from ragcore.auth.dependencies import get_current_api_key_id
from ragcore.modules.agents.orchestrator import orchestrator
from ragcore.modules.agents.execution_planner import execution_planner
from ragcore.modules.agents.models import (
    ChainDefinition,
    ChainExecution,
    AgentDefinition,
)
from ragcore.db.database import async_session_factory
from sqlalchemy import select

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])


# Request/Response Models
class AgentDefinitionSchema(BaseModel):
    """Agent definition schema."""

    name: str
    description: str
    tools: List[str] = []
    model_config_id: str
    system_prompt: str
    temperature: int = 7
    max_tokens: int = 2048
    timeout_seconds: int = 300

    class Config:
        from_attributes = True


class ChainDefinitionSchema(BaseModel):
    """Chain definition schema."""

    name: str
    description: str
    chain_type: str  # "sequential" | "parallel" | "conditional" | "recursive"
    agents: List[dict]  # Agent configurations
    aggregation_strategy: str = "concat"
    max_iterations: int = 3
    timeout_seconds: int = 900

    class Config:
        from_attributes = True


class ChainExecutionSchema(BaseModel):
    """Chain execution schema."""

    id: UUID
    chain_id: Optional[UUID]
    session_id: UUID
    status: str
    input_query: str
    final_result: Optional[dict] = None
    error_message: Optional[str] = None
    total_tokens_used: int = 0
    completed_at: Optional[str] = None

    class Config:
        from_attributes = True


class ExecuteChainRequest(BaseModel):
    """Request to execute a chain."""

    chain_id: UUID
    query: str
    session_id: UUID
    user_id: Optional[UUID] = None
    override_chain: Optional[ChainDefinitionSchema] = None


class ExecuteChainResponse(BaseModel):
    """Response from chain execution."""

    execution_id: UUID
    status: str
    final_result: dict
    total_tokens_used: int
    error_message: Optional[str] = None


class ChainStatusResponse(BaseModel):
    """Response with chain status."""

    execution_id: UUID
    status: str
    steps_completed: int
    total_steps: int
    error_message: Optional[str] = None


# Endpoints
@router.get("/chains", summary="List all chains")
async def list_chains(api_key_id: UUID = Depends(get_current_api_key_id)):
    """List all available chain definitions."""
    try:
        async with async_session_factory() as session:
            stmt = select(ChainDefinition).where(ChainDefinition.is_active == True)
            result = await session.execute(stmt)
            chains = result.scalars().all()

            return {
                "total": len(chains),
                "chains": [
                    {
                        "id": str(chain.id),
                        "name": chain.name,
                        "description": chain.description,
                        "chain_type": chain.chain_type,
                        "agent_count": len(chain.agents),
                    }
                    for chain in chains
                ],
            }
    except Exception as e:
        logger.error(f"Error listing chains: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list chains",
        )


@router.get("/chains/{chain_id}", summary="Get chain details")
async def get_chain(
    chain_id: UUID,
    api_key_id: UUID = Depends(get_current_api_key_id),
):
    """Get details of a specific chain."""
    try:
        async with async_session_factory() as session:
            stmt = select(ChainDefinition).where(ChainDefinition.id == chain_id)
            result = await session.execute(stmt)
            chain = result.scalar_one_or_none()

            if not chain:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Chain {chain_id} not found",
                )

            return ChainDefinitionSchema.from_orm(chain)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chain {chain_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve chain",
        )


@router.post("/chains", summary="Create a new chain")
async def create_chain(
    chain: ChainDefinitionSchema,
    api_key_id: UUID = Depends(get_current_api_key_id),
):
    """Create a new chain definition."""
    try:
        # Verify agents exist
        async with async_session_factory() as session:
            for agent_config in chain.agents:
                agent_name = agent_config.get("agent_name")
                stmt = select(AgentDefinition).where(
                    AgentDefinition.name == agent_name
                )
                result = await session.execute(stmt)
                if not result.scalar_one_or_none():
                    raise ValueError(f"Agent '{agent_name}' not found")

            # Create chain
            new_chain = ChainDefinition(
                name=chain.name,
                description=chain.description,
                chain_type=chain.chain_type,
                agents=chain.agents,
                aggregation_strategy=chain.aggregation_strategy,
                max_iterations=chain.max_iterations,
                timeout_seconds=chain.timeout_seconds,
                is_active=True,
            )
            session.add(new_chain)
            await session.commit()
            await session.refresh(new_chain)

            logger.info(f"Created chain: {new_chain.name} ({new_chain.id})")

            return {
                "id": str(new_chain.id),
                "name": new_chain.name,
                "message": "Chain created successfully",
            }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error creating chain: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create chain",
        )


@router.post("/execute", summary="Execute a chain")
async def execute_chain(
    request: ExecuteChainRequest,
    api_key_id: UUID = Depends(get_current_api_key_id),
):
    """Execute a chain with the given query.

    This endpoint orchestrates a multi-agent chain execution and returns the result.
    """
    try:
        logger.info(f"Executing chain {request.chain_id} with query: {request.query[:100]}")

        # Convert override_chain if provided
        override_chain = None
        if request.override_chain:
            override_chain = ChainDefinition(
                name=request.override_chain.name,
                description=request.override_chain.description,
                chain_type=request.override_chain.chain_type,
                agents=request.override_chain.agents,
                aggregation_strategy=request.override_chain.aggregation_strategy,
                max_iterations=request.override_chain.max_iterations,
                timeout_seconds=request.override_chain.timeout_seconds,
            )

        # Execute chain
        execution = await orchestrator.execute_chain(
            chain_id=request.chain_id,
            query=request.query,
            session_id=request.session_id,
            user_id=request.user_id,
            override_chain=override_chain,
        )

        response = ExecuteChainResponse(
            execution_id=execution.id,
            status=execution.status,
            final_result=execution.final_result or {},
            total_tokens_used=execution.total_tokens_used,
            error_message=execution.error_message,
        )

        logger.info(f"Chain execution {execution.id} completed with status: {execution.status}")

        return response

    except ValueError as e:
        logger.warning(f"Validation error in chain execution: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Error executing chain: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chain execution failed",
        )


@router.get("/executions/{execution_id}", summary="Get execution status")
async def get_execution_status(
    execution_id: UUID,
    api_key_id: UUID = Depends(get_current_api_key_id),
):
    """Get status of a chain execution."""
    try:
        async with async_session_factory() as session:
            stmt = select(ChainExecution).where(ChainExecution.id == execution_id)
            result = await session.execute(stmt)
            execution = result.scalar_one_or_none()

            if not execution:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Execution {execution_id} not found",
                )

            return ChainExecutionSchema.from_orm(execution)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting execution {execution_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve execution",
        )


@router.get("/executions", summary="List recent executions")
async def list_executions(
    session_id: Optional[UUID] = None,
    status_filter: Optional[str] = None,
    limit: int = 10,
    api_key_id: UUID = Depends(get_current_api_key_id),
):
    """List recent chain executions with optional filtering."""
    try:
        async with async_session_factory() as session:
            stmt = select(ChainExecution)

            if session_id:
                stmt = stmt.where(ChainExecution.session_id == session_id)

            if status_filter:
                stmt = stmt.where(ChainExecution.status == status_filter)

            stmt = stmt.order_by(ChainExecution.started_at.desc()).limit(limit)

            result = await session.execute(stmt)
            executions = result.scalars().all()

            return {
                "total": len(executions),
                "executions": [
                    {
                        "id": str(exec.id),
                        "status": exec.status,
                        "chain_id": str(exec.chain_id) if exec.chain_id else None,
                        "session_id": str(exec.session_id),
                        "steps_completed": exec.steps_completed,
                        "total_steps": exec.total_steps,
                        "total_tokens_used": exec.total_tokens_used,
                        "started_at": exec.started_at.isoformat(),
                    }
                    for exec in executions
                ],
            }

    except Exception as e:
        logger.error(f"Error listing executions: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list executions",
        )


@router.get("/agents", summary="List available agents")
async def list_agents(
    api_key_id: UUID = Depends(get_current_api_key_id),
):
    """List all available agent definitions."""
    try:
        async with async_session_factory() as session:
            stmt = select(AgentDefinition).where(AgentDefinition.is_active == True)
            result = await session.execute(stmt)
            agents = result.scalars().all()

            return {
                "total": len(agents),
                "agents": [
                    {
                        "name": agent.name,
                        "description": agent.description,
                        "tools": agent.tools,
                        "model_config_id": agent.model_config_id,
                    }
                    for agent in agents
                ],
            }

    except Exception as e:
        logger.error(f"Error listing agents: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list agents",
        )
