"""Execution planning and routing for agent chains."""

import logging
from typing import Optional, Dict, Any, List
from uuid import UUID

from ragcore.modules.agents.models import ChainDefinition, AgentDefinition

logger = logging.getLogger(__name__)


class ExecutionPlan:
    """Represents a planned execution of a chain."""

    def __init__(
        self,
        chain_id: UUID,
        chain_definition: ChainDefinition,
        initial_query: str,
    ):
        """Initialize execution plan.

        Args:
            chain_id: ID of chain to execute
            chain_definition: Chain definition
            initial_query: User's initial query
        """
        self.chain_id = chain_id
        self.chain_definition = chain_definition
        self.initial_query = initial_query
        self.steps: List[Dict[str, Any]] = []
        self.routing_decisions: Dict[str, Any] = {}

    def add_step(
        self,
        step_number: int,
        agent_name: str,
        tools_required: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a step to the plan.

        Args:
            step_number: Order in execution
            agent_name: Agent to execute
            tools_required: Tools needed
            metadata: Optional additional data
        """
        step = {
            "step_number": step_number,
            "agent_name": agent_name,
            "tools_required": tools_required,
            "metadata": metadata or {},
        }
        self.steps.append(step)

    def set_routing_decision(self, key: str, value: Any) -> None:
        """Record a routing decision.

        Args:
            key: Decision key
            value: Decision value
        """
        self.routing_decisions[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dict."""
        return {
            "chain_id": str(self.chain_id),
            "chain_type": self.chain_definition.chain_type,
            "initial_query": self.initial_query,
            "steps": self.steps,
            "routing_decisions": self.routing_decisions,
            "total_steps": len(self.steps),
        }


class ExecutionPlanner:
    """Plan execution of agent chains."""

    def __init__(self):
        """Initialize execution planner."""
        logger.info("ExecutionPlanner initialized")

    async def plan_sequential_execution(
        self,
        chain: ChainDefinition,
        initial_query: str,
        chain_id: UUID,
    ) -> ExecutionPlan:
        """Plan sequential chain execution.

        Args:
            chain: Chain definition
            initial_query: User query
            chain_id: Chain ID

        Returns:
            ExecutionPlan with ordered steps
        """
        plan = ExecutionPlan(chain_id, chain, initial_query)

        logger.debug(f"Planning sequential execution for {len(chain.agents)} agents")

        # Sequential: add steps in order
        for step_number, agent_config in enumerate(chain.agents, 1):
            agent_name = agent_config.get("agent_name")
            tools_required = agent_config.get("tools", [])

            plan.add_step(
                step_number=step_number,
                agent_name=agent_name,
                tools_required=tools_required,
                metadata={"config": agent_config},
            )

        logger.info(f"Sequential execution plan created: {len(plan.steps)} steps")
        return plan

    async def plan_parallel_execution(
        self,
        chain: ChainDefinition,
        initial_query: str,
        chain_id: UUID,
    ) -> ExecutionPlan:
        """Plan parallel chain execution.

        Args:
            chain: Chain definition
            initial_query: User query
            chain_id: Chain ID

        Returns:
            ExecutionPlan with parallel steps + aggregation
        """
        plan = ExecutionPlan(chain_id, chain, initial_query)

        logger.debug(f"Planning parallel execution for {len(chain.agents)} agents")

        # Parallel: all steps get same number (they run concurrently)
        for agent_config in chain.agents:
            agent_name = agent_config.get("agent_name")
            tools_required = agent_config.get("tools", [])

            plan.add_step(
                step_number=1,  # All run in parallel
                agent_name=agent_name,
                tools_required=tools_required,
                metadata={
                    "config": agent_config,
                    "parallel": True,
                },
            )

        # Record aggregation strategy
        plan.set_routing_decision(
            "aggregation_strategy", chain.aggregation_strategy
        )

        logger.info(
            f"Parallel execution plan created: {len(plan.steps)} agents, "
            f"aggregation={chain.aggregation_strategy}"
        )
        return plan

    async def plan_conditional_execution(
        self,
        chain: ChainDefinition,
        initial_query: str,
        chain_id: UUID,
    ) -> ExecutionPlan:
        """Plan conditional chain execution.

        Args:
            chain: Chain definition
            initial_query: User query
            chain_id: Chain ID

        Returns:
            ExecutionPlan with conditional routing

        Note:
            Full implementation in next sprint
        """
        raise NotImplementedError("Conditional chains coming in next sprint")

    async def plan_recursive_execution(
        self,
        chain: ChainDefinition,
        initial_query: str,
        chain_id: UUID,
    ) -> ExecutionPlan:
        """Plan recursive chain execution.

        Args:
            chain: Chain definition
            initial_query: User query
            chain_id: Chain ID

        Returns:
            ExecutionPlan with recursive loop

        Note:
            Full implementation in next sprint
        """
        raise NotImplementedError("Recursive chains coming in next sprint")

    async def plan_execution(
        self,
        chain: ChainDefinition,
        initial_query: str,
        chain_id: UUID,
    ) -> ExecutionPlan:
        """Plan execution based on chain type.

        Args:
            chain: Chain definition
            initial_query: User query
            chain_id: Chain ID

        Returns:
            ExecutionPlan

        Raises:
            ValueError: If chain type not supported
        """
        chain_type = chain.chain_type

        logger.info(f"Planning execution for chain type: {chain_type}")

        if chain_type == "sequential":
            return await self.plan_sequential_execution(chain, initial_query, chain_id)
        elif chain_type == "parallel":
            return await self.plan_parallel_execution(chain, initial_query, chain_id)
        elif chain_type == "conditional":
            return await self.plan_conditional_execution(
                chain, initial_query, chain_id
            )
        elif chain_type == "recursive":
            return await self.plan_recursive_execution(chain, initial_query, chain_id)
        else:
            raise ValueError(f"Unknown chain type: {chain_type}")

    def validate_plan(self, plan: ExecutionPlan) -> tuple[bool, Optional[str]]:
        """Validate an execution plan.

        Args:
            plan: ExecutionPlan to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not plan.steps:
            return False, "Plan has no steps"

        # Validate agents exist (in real implementation, query database)
        agent_names = set()
        for step in plan.steps:
            agent_names.add(step["agent_name"])

        if not agent_names:
            return False, "Plan has no agents"

        logger.debug(f"Plan validation passed: {len(plan.steps)} steps, {len(agent_names)} agents")
        return True, None

    def estimate_execution_time_ms(self, plan: ExecutionPlan) -> int:
        """Estimate execution time for a plan.

        Args:
            plan: ExecutionPlan to estimate

        Returns:
            Estimated time in milliseconds
        """
        # Simple estimation:
        # - Sequential: sum of 1000ms per step
        # - Parallel: max of 1000ms per step (all contribute equally)
        # Real implementation would use agent configs

        if plan.chain_definition.chain_type == "parallel":
            # Parallel agents run concurrently
            return 1000 * len(plan.steps)
        else:
            # Sequential agents run one after another
            return 1000 * len(plan.steps)

    def estimate_token_usage(self, plan: ExecutionPlan) -> int:
        """Estimate token usage for a plan.

        Args:
            plan: ExecutionPlan to estimate

        Returns:
            Estimated token count
        """
        # Simple estimation: ~2000 tokens per agent
        # Real implementation would use agent configs and query complexity
        return 2000 * len(plan.steps)


# Global execution planner instance
execution_planner = ExecutionPlanner()
