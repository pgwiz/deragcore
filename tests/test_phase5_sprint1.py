"""Phase 5 Sprint 1: Agent Chain Orchestration - Test Suite."""

import pytest
import asyncio
from uuid import uuid4
from datetime import datetime

# Test imports
def test_agent_orchestrator_imports():
    """Test orchestrator module imports."""
    from ragcore.modules.agents.orchestrator import (
        AgentChainOrchestrator,
        orchestrator,
    )

    assert AgentChainOrchestrator is not None
    assert orchestrator is not None
    assert hasattr(orchestrator, "execute_chain")


def test_tool_composer_imports():
    """Test tool composer module imports."""
    from ragcore.modules.agents.tool_composer import (
        ToolComposer,
        ToolDefinition,
        ToolResult,
        tool_composer,
    )

    assert ToolComposer is not None
    assert ToolDefinition is not None
    assert ToolResult is not None
    assert tool_composer is not None


def test_execution_planner_imports():
    """Test execution planner module imports."""
    from ragcore.modules.agents.execution_planner import (
        ExecutionPlanner,
        ExecutionPlan,
        execution_planner,
    )

    assert ExecutionPlanner is not None
    assert ExecutionPlan is not None
    assert execution_planner is not None


def test_agent_models_imports():
    """Test agent models."""
    from ragcore.modules.agents.models import (
        AgentDefinition,
        ChainDefinition,
        ChainExecution,
        ExecutionStep,
        ChainTemplate,
    )

    assert AgentDefinition is not None
    assert ChainDefinition is not None
    assert ChainExecution is not None
    assert ExecutionStep is not None
    assert ChainTemplate is not None


def test_router_imports():
    """Test router module imports."""
    from ragcore.modules.agents import router

    assert router is not None


# Tool Composer Tests
def test_tool_definition_creation():
    """Test creating a tool definition."""
    from ragcore.modules.agents.tool_composer import ToolDefinition

    async def mock_search(params, session_id=None):
        return [{"title": "Result 1", "url": "http://example.com"}]

    tool = ToolDefinition(
        name="web_search",
        description="Search the web",
        category="search",
        execute_func=mock_search,
        required_params=["query"],
        optional_params={"max_results": 5},
    )

    assert tool.name == "web_search"
    assert tool.description == "Search the web"
    assert tool.category == "search"
    assert tool.required_params == ["query"]


def test_tool_composer_registration():
    """Test registering tools with composer."""
    from ragcore.modules.agents.tool_composer import ToolComposer, ToolDefinition

    async def mock_tool(params, session_id=None):
        return {"result": "success"}

    composer = ToolComposer()

    tool1 = ToolDefinition(
        name="tool1",
        description="Tool 1",
        category="search",
        execute_func=mock_tool,
        required_params=["input"],
        optional_params={},
    )

    tool2 = ToolDefinition(
        name="tool2",
        description="Tool 2",
        category="analysis",
        execute_func=mock_tool,
        required_params=["input"],
        optional_params={},
    )

    composer.register_tool(tool1)
    composer.register_tool(tool2)

    assert composer.get_tool("tool1") is not None
    assert composer.get_tool("tool2") is not None
    assert len(composer.list_tools()) == 2


def test_tool_result_serialization():
    """Test tool result serialization."""
    from ragcore.modules.agents.tool_composer import ToolResult

    result = ToolResult(
        tool_name="web_search",
        status="success",
        result=[{"title": "Result"}],
        execution_time_ms=150,
    )

    result_dict = result.to_dict()
    assert result_dict["tool_name"] == "web_search"
    assert result_dict["status"] == "success"
    assert result_dict["execution_time_ms"] == 150


@pytest.mark.asyncio
async def test_tool_execution_missing_required_param():
    """Test tool execution with missing required parameters."""
    from ragcore.modules.agents.tool_composer import ToolComposer, ToolDefinition

    async def mock_tool(params, session_id=None):
        return {"result": "success"}

    composer = ToolComposer()
    tool = ToolDefinition(
        name="test_tool",
        description="Test",
        category="test",
        execute_func=mock_tool,
        required_params=["required_param"],
        optional_params={},
    )
    composer.register_tool(tool)

    # Try to execute with missing required parameter
    result = await composer.execute_tool("test_tool", {})

    assert result.status == "error"
    assert "Missing required parameters" in result.error_message


@pytest.mark.asyncio
async def test_tool_execution_success():
    """Test successful tool execution."""
    from ragcore.modules.agents.tool_composer import ToolComposer, ToolDefinition
    import asyncio

    async def mock_search(params, session_id=None):
        await asyncio.sleep(0.01)  # 10ms delay
        return [
            {"title": "Result 1", "url": "http://example.com/1"},
            {"title": "Result 2", "url": "http://example.com/2"},
        ]

    composer = ToolComposer()
    tool = ToolDefinition(
        name="web_search",
        description="Search the web",
        category="search",
        execute_func=mock_search,
        required_params=["query"],
        optional_params={},
    )
    composer.register_tool(tool)

    result = await composer.execute_tool(
        "web_search",
        {"query": "test query"},
    )

    assert result.status == "success"
    assert len(result.result) == 2
    assert result.execution_time_ms >= 10  # At least 10ms from our sleep


def test_tool_result_formatting():
    """Test formatting tool results for context."""
    from ragcore.modules.agents.tool_composer import ToolComposer, ToolResult

    composer = ToolComposer()

    results = [
        ToolResult(
            tool_name="web_search",
            status="success",
            result=[
                {"title": "Result 1"},
                {"title": "Result 2"},
            ],
            execution_time_ms=100,
        ),
    ]

    formatted = composer.format_tool_results_for_context(results)

    assert "web_search" in formatted
    assert "Result 1" in formatted
    assert "100ms" in formatted


# Execution Planner Tests
@pytest.mark.asyncio
async def test_execution_plan_creation():
    """Test creating an execution plan."""
    from ragcore.modules.agents.execution_planner import ExecutionPlan
    from ragcore.modules.agents.models import ChainDefinition

    chain = ChainDefinition(
        name="test_chain",
        description="Test",
        chain_type="sequential",
        agents=[],
    )

    plan = ExecutionPlan(
        chain_id=uuid4(),
        chain_definition=chain,
        initial_query="Test query",
    )

    assert plan.initial_query == "Test query"
    assert len(plan.steps) == 0


@pytest.mark.asyncio
async def test_sequential_execution_planning():
    """Test planning sequential chain execution."""
    from ragcore.modules.agents.execution_planner import execution_planner
    from ragcore.modules.agents.models import ChainDefinition

    chain = ChainDefinition(
        name="seq_chain",
        description="Sequential chain",
        chain_type="sequential",
        agents=[
            {"agent_name": "agent1", "tools": ["web_search"]},
            {"agent_name": "agent2", "tools": ["summarize"]},
        ],
    )

    plan = await execution_planner.plan_sequential_execution(
        chain=chain,
        initial_query="Test query",
        chain_id=uuid4(),
    )

    assert len(plan.steps) == 2
    assert plan.steps[0]["step_number"] == 1
    assert plan.steps[0]["agent_name"] == "agent1"
    assert plan.steps[1]["step_number"] == 2
    assert plan.steps[1]["agent_name"] == "agent2"


@pytest.mark.asyncio
async def test_parallel_execution_planning():
    """Test planning parallel chain execution."""
    from ragcore.modules.agents.execution_planner import execution_planner
    from ragcore.modules.agents.models import ChainDefinition

    chain = ChainDefinition(
        name="par_chain",
        description="Parallel chain",
        chain_type="parallel",
        agents=[
            {"agent_name": "agent1", "tools": ["web_search"]},
            {"agent_name": "agent2", "tools": ["web_search"]},
            {"agent_name": "agent3", "tools": ["web_search"]},
        ],
        aggregation_strategy="merge",
    )

    plan = await execution_planner.plan_parallel_execution(
        chain=chain,
        initial_query="Test query",
        chain_id=uuid4(),
    )

    assert len(plan.steps) == 3
    # All steps should run in parallel (same step number)
    assert all(step["step_number"] == 1 for step in plan.steps)
    assert plan.routing_decisions["aggregation_strategy"] == "merge"


@pytest.mark.asyncio
async def test_plan_validation():
    """Test execution plan validation."""
    from ragcore.modules.agents.execution_planner import ExecutionPlan, execution_planner
    from ragcore.modules.agents.models import ChainDefinition

    chain = ChainDefinition(
        name="test",
        description="Test",
        chain_type="sequential",
        agents=[],
    )

    plan = ExecutionPlan(
        chain_id=uuid4(),
        chain_definition=chain,
        initial_query="Query",
    )

    # Should be invalid with no steps
    is_valid, error = execution_planner.validate_plan(plan)
    assert not is_valid
    assert error is not None

    # Add a step and validate
    plan.add_step(1, "test_agent", [])
    is_valid, error = execution_planner.validate_plan(plan)
    assert is_valid


def test_execution_time_estimation():
    """Test execution time estimation."""
    from ragcore.modules.agents.execution_planner import ExecutionPlan, execution_planner
    from ragcore.modules.agents.models import ChainDefinition

    chain = ChainDefinition(
        name="test",
        description="Test",
        chain_type="sequential",
        agents=[
            {"agent_name": "agent1"},
            {"agent_name": "agent2"},
        ],
    )

    plan = ExecutionPlan(uuid4(), chain, "query")
    plan.add_step(1, "agent1", [])
    plan.add_step(2, "agent2", [])

    estimated_ms = execution_planner.estimate_execution_time_ms(plan)
    # Should estimate ~2000ms for 2 agents
    assert estimated_ms > 0


# Integration tests
def test_app_has_agents_routes():
    """Test app includes agent routes."""
    from ragcore.main import create_app

    app = create_app()
    routes = [str(route.path) for route in app.routes]

    assert any("/agents" in r for r in routes), f"Agent routes not found in {routes}"


@pytest.mark.asyncio
async def test_agent_module_initialization():
    """Test agent module can be fully initialized."""
    from ragcore.modules.agents import (
        orchestrator,
        tool_composer,
        execution_planner,
        router,
    )

    assert orchestrator is not None
    assert tool_composer is not None
    assert execution_planner is not None
    assert router is not None


def test_phase5_sprint1_complete():
    """Overall Sprint 1 completion check."""
    # Test all core components are present
    from ragcore.modules.agents.orchestrator import AgentChainOrchestrator
    from ragcore.modules.agents.tool_composer import ToolComposer
    from ragcore.modules.agents.execution_planner import ExecutionPlanner
    from ragcore.modules.agents.models import (
        AgentDefinition,
        ChainDefinition,
        ChainExecution,
        ExecutionStep,
    )

    assert AgentChainOrchestrator is not None
    assert ToolComposer is not None
    assert ExecutionPlanner is not None
    assert AgentDefinition is not None
    assert ChainDefinition is not None
    assert ChainExecution is not None
    assert ExecutionStep is not None

    print("✓ Phase 5 Sprint 1: Agent Chain Core Infrastructure - All Components Present")
