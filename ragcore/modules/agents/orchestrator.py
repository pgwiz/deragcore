"""Agent chain orchestrator - executes multi-agent workflows."""

import logging
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ragcore.config import settings
from ragcore.db.database import async_session_factory
from ragcore.modules.agents.models import (
    ChainDefinition,
    ChainExecution,
    ExecutionStep,
    AgentDefinition,
)
from ragcore.core.ai_controller import AIController
from ragcore.core.schemas import ORION_DEFAULT_PROMPT

logger = logging.getLogger(__name__)


class AgentChainOrchestrator:
    """Orchestrates multi-agent chain execution."""

    def __init__(self):
        """Initialize orchestrator."""
        self.ai_controller = AIController
        self.max_retries = 3

    async def execute_chain(
        self,
        chain_id: UUID,
        query: str,
        session_id: UUID,
        user_id: Optional[UUID] = None,
        override_chain: Optional[ChainDefinition] = None,
    ) -> ChainExecution:
        """
        Execute a multi-agent chain.

        Args:
            chain_id: Chain to execute
            query: User query/input
            session_id: Session context
            user_id: Optional user ID
            override_chain: Optional chain definition to override

        Returns:
            ChainExecution with results
        """
        async with async_session_factory() as session:
            # Get chain definition
            if override_chain:
                chain = override_chain
            else:
                stmt = select(ChainDefinition).where(ChainDefinition.id == chain_id)
                result = await session.execute(stmt)
                chain = result.scalar_one_or_none()

            if not chain:
                raise ValueError(f"Chain {chain_id} not found")

            # Create execution record
            execution = ChainExecution(
                chain_id=chain.id if not override_chain else None,
                session_id=session_id,
                user_id=user_id,
                input_query=query,
                status="running",
                total_steps=len(chain.agents),
            )
            session.add(execution)
            await session.commit()
            await session.refresh(execution)

            logger.info(f"Starting chain execution {execution.id}: {chain.name}")

        # Execute chain based on type
        try:
            if chain.chain_type == "sequential":
                result = await self._execute_sequential(chain, query, execution)
            elif chain.chain_type == "parallel":
                result = await self._execute_parallel(chain, query, execution)
            elif chain.chain_type == "conditional":
                result = await self._execute_conditional(chain, query, execution)
            elif chain.chain_type == "recursive":
                result = await self._execute_recursive(chain, query, execution, chain.max_iterations)
            else:
                raise ValueError(f"Unknown chain type: {chain.chain_type}")

            # Mark as completed
            execution.status = "completed"
            execution.final_result = result
            execution.completed_at = datetime.utcnow()

            logger.info(f"Chain execution {execution.id} completed successfully")

        except Exception as e:
            logger.error(f"Chain execution {execution.id} failed: {str(e)}", exc_info=True)
            execution.status = "failed"
            execution.error_message = str(e)
            execution.completed_at = datetime.utcnow()

        # Save execution
        async with async_session_factory() as session:
            stmt = select(ChainExecution).where(ChainExecution.id == execution.id)
            result = await session.execute(stmt)
            saved_execution = result.scalar_one()
            saved_execution.status = execution.status
            saved_execution.final_result = execution.final_result
            saved_execution.error_message = execution.error_message
            saved_execution.completed_at = execution.completed_at
            session.add(saved_execution)
            await session.commit()

        return execution

    async def _execute_sequential(
        self,
        chain: ChainDefinition,
        initial_input: str,
        execution: ChainExecution,
    ) -> Dict[str, Any]:
        """
        Execute agents one after another, passing output to next.

        Flow: Agent1(input) → Agent2(Agent1.output) → Agent3(Agent2.output) → Final result
        """
        logger.info(f"Executing sequential chain with {len(chain.agents)} agents")

        current_input = initial_input
        results = {}

        async with async_session_factory() as session:
            for step_number, agent_config in enumerate(chain.agents, 1):
                agent_name = agent_config.get("agent_name")
                logger.info(f"Step {step_number}: Executing agent '{agent_name}'")

                try:
                    # Get agent definition
                    stmt = select(AgentDefinition).where(
                        AgentDefinition.name == agent_name
                    )
                    result = await session.execute(stmt)
                    agent = result.scalar_one_or_none()

                    if not agent:
                        raise ValueError(f"Agent '{agent_name}' not found")

                    # Execute agent
                    start_time = datetime.utcnow()
                    output = await self._run_agent(
                        agent=agent,
                        input_data=current_input,
                        context=results,
                        session_id=execution.session_id,
                    )
                    duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)

                    # Record step
                    step = ExecutionStep(
                        execution_id=execution.id,
                        step_number=step_number,
                        agent_name=agent_name,
                        input_data={"text": current_input},
                        output_data=output,
                        status="completed",
                        duration_ms=duration_ms,
                        tokens_used=output.get("tokens_used", 0),
                    )
                    session.add(step)
                    await session.commit()

                    # Store result for next agent
                    results[agent_name] = output
                    current_input = output.get("response", "")

                    # Update execution
                    execution.steps_completed = step_number
                    execution.total_tokens_used += output.get("tokens_used", 0)

                    logger.info(
                        f"Step {step_number} completed in {duration_ms}ms: {agent_name}"
                    )

                except Exception as e:
                    logger.error(f"Step {step_number} failed: {str(e)}", exc_info=True)
                    step = ExecutionStep(
                        execution_id=execution.id,
                        step_number=step_number,
                        agent_name=agent_name,
                        input_data={"text": current_input},
                        status="failed",
                        error_message=str(e),
                    )
                    session.add(step)
                    await session.commit()
                    raise

        return {
            "stage": "completed",
            "final_response": current_input,
            "all_results": results,
            "total_tokens": execution.total_tokens_used,
        }

    async def _execute_parallel(
        self,
        chain: ChainDefinition,
        initial_input: str,
        execution: ChainExecution,
    ) -> Dict[str, Any]:
        """
        Execute agents in parallel, aggregate results.

        Flow: Agent1(input) ─┐
              Agent2(input) ─┤→ Aggregator(all results) → Final result
              Agent3(input) ─┘
        """
        logger.info(f"Executing parallel chain with {len(chain.agents)} agents")

        tasks = []
        async with async_session_factory() as session:
            for step_number, agent_config in enumerate(chain.agents, 1):
                agent_name = agent_config.get("agent_name")

                # Get agent definition
                stmt = select(AgentDefinition).where(
                    AgentDefinition.name == agent_name
                )
                result = await session.execute(stmt)
                agent = result.scalar_one_or_none()

                if not agent:
                    raise ValueError(f"Agent '{agent_name}' not found")

                # Create task
                tasks.append(
                    self._run_agent(
                        agent=agent,
                        input_data=initial_input,
                        context={},
                        session_id=execution.session_id,
                    )
                )

        # Run all in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        agent_results = {}
        total_tokens = 0

        for i, (agent_config, result) in enumerate(
            zip(chain.agents, results), 1
        ):
            agent_name = agent_config.get("agent_name")
            if isinstance(result, Exception):
                logger.error(f"Agent {agent_name} failed: {str(result)}")
                agent_results[agent_name] = {"error": str(result)}
            else:
                agent_results[agent_name] = result
                total_tokens += result.get("tokens_used", 0)

        # Aggregate using strategy
        if chain.aggregation_strategy == "concat":
            aggregated = " ".join(
                [result.get("response", "") for result in agent_results.values()
                 if "response" in result]
            )
        elif chain.aggregation_strategy == "merge":
            aggregated = {agent: result for agent, result in agent_results.items()}
        else:  # vote
            aggregated = agent_results

        execution.total_tokens_used = total_tokens

        return {
            "stage": "completed",
            "agent_results": agent_results,
            "aggregated": aggregated,
            "total_tokens": total_tokens,
        }

    async def _execute_conditional(
        self,
        chain: ChainDefinition,
        initial_input: str,
        execution: ChainExecution,
    ) -> Dict[str, Any]:
        """
        Execute based on conditional routing.

        Flow: Evaluator(input) → Check condition → Route to AgentA or AgentB
        """
        raise NotImplementedError("Conditional chains coming in next sprint")

    async def _execute_recursive(
        self,
        chain: ChainDefinition,
        initial_input: str,
        execution: ChainExecution,
        max_iterations: int,
    ) -> Dict[str, Any]:
        """
        Execute recursively until condition met or max iterations reached.
        """
        raise NotImplementedError("Recursive chains coming in next sprint")

    async def _run_agent(
        self,
        agent: AgentDefinition,
        input_data: Any,
        context: Dict[str, Any],
        session_id: UUID,
    ) -> Dict[str, Any]:
        """
        Run a single agent.

        Args:
            agent: Agent definition
            input_data: Input to agent
            context: Results from previous steps
            session_id: Session context

        Returns:
            {response, tokens_used, tool_calls}
        """
        logger.debug(f"Running agent: {agent.name}")

        # Build system prompt with context
        system_prompt = agent.system_prompt
        if context:
            context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
            system_prompt += f"\n\nPrevious results:\n{context_str}"

        # Call LLM
        response = self.ai_controller.complete(
            provider_name="anthropic",
            model_id=agent.model_config_id,
            messages=[{"role": "user", "content": str(input_data)}],
            temperature=agent.temperature / 10.0,
            max_tokens=agent.max_tokens,
            system_prompt=system_prompt,
        )

        return {
            "response": response.text,
            "tokens_used": response.input_tokens + response.output_tokens,
            "model": response.model,
            "provider": response.provider,
            "tool_calls": [],  # Will be extended in Sprint 1 Phase 2
        }


# Global orchestrator instance
orchestrator = AgentChainOrchestrator()
