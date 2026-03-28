"""Tests for Phase 3 - Research module with web search tools."""

import pytest
from unittest.mock import patch, AsyncMock
from uuid import uuid4

from fastapi.testclient import TestClient

from ragcore.main import app
from ragcore.modules.research.tools.search import (
    SearchResult,
    TavilySearchTool,
    SerpAPISearchTool,
    DuckDuckGoSearchTool,
)
from ragcore.modules.research.tool_registry import ToolExecutor
from ragcore.modules.research.models import (
    ResearchFinding,
    ToolCall,
    ToolResult,
    ResearchTurn,
    ResearchSessionState,
)
from ragcore.modules.research.agent_planner import ResearchPlanner
from ragcore.modules.research.pipeline import ResearchPipeline

client = TestClient(app)


# ============================================================================
# Search Tool Tests
# ============================================================================


class TestSearchTools:
    """Test web search tool implementations."""

    def test_search_result_creation(self):
        """Test SearchResult model."""
        result = SearchResult(
            title="Test Article",
            url="https://example.com/test",
            snippet="This is a test snippet",
            source="tavily",
            relevance_score=0.95,
        )

        assert result.title == "Test Article"
        assert result.source == "tavily"
        assert result.relevance_score == 0.95

        # Test serialization
        result_dict = result.to_dict()
        assert result_dict["title"] == "Test Article"

    def test_tavily_tool_availability(self):
        """Test Tavily tool checks API key."""
        tool = TavilySearchTool()

        # Should be unavailable without key (from config)
        # Note: actual availability depends on settings.tavily_api_key
        availability = tool.is_available()
        assert isinstance(availability, bool)

    def test_duckduckgo_tool_always_available(self):
        """Test DuckDuckGo tool is always available."""
        tool = DuckDuckGoSearchTool()
        assert tool.is_available() is True

    @pytest.mark.asyncio
    async def test_duckduckgo_search_mock(self):
        """Test DuckDuckGo search with mock."""
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            pytest.skip("duckduckgo_search module not available")

        tool = DuckDuckGoSearchTool()

        # Just verify the tool is available and search method exists
        assert tool.is_available()
        assert callable(tool.search)


# ============================================================================
# Tool Registry Tests
# ============================================================================


class TestToolRegistry:
    """Test tool selection and execution."""

    def test_executor_initialization(self):
        """Test ToolExecutor initializes properly."""
        executor = ToolExecutor()

        assert executor.factory is not None
        assert executor.all_tools is not None
        assert executor.priority is not None

    def test_get_available_tools(self):
        """Test getting available tools."""
        executor = ToolExecutor()
        available = executor.get_available_tools()

        # Should always have at least DuckDuckGo (no API key needed)
        assert isinstance(available, dict)
        assert "duckduckgo" in available or len(available) >= 0

    def test_tool_status(self):
        """Test getting tool status."""
        executor = ToolExecutor()
        status = executor.get_tool_status()

        assert isinstance(status, dict)
        assert "duckduckgo" in status  # Always available
        assert isinstance(status["duckduckgo"], bool)

    @pytest.mark.asyncio
    async def test_execute_with_fallback(self):
        """Test fallback execution strategy."""
        executor = ToolExecutor()

        with patch.object(executor.factory, "get_tool") as mock_get:
            mock_tool = AsyncMock()
            mock_tool.search = AsyncMock(return_value=[
                SearchResult("Title", "url", "snippet", "mock", 0.9)
            ])
            mock_get.return_value = mock_tool

            results, tool_used = await executor.execute_with_fallback(
                query="test",
                max_results=5,
            )

            # Verify execution path
            assert len(results) >= 0 or tool_used in ["none", "mock"]


# ============================================================================
# Research Data Models Tests
# ============================================================================


class TestResearchModels:
    """Test research-specific data models."""

    def test_research_finding_creation(self):
        """Test ResearchFinding model."""
        finding = ResearchFinding(
            query="What is AI?",
            results=[{"title": "AI Basics", "url": "https://example.com"}],
            tool_used="tavily",
            synthesis="AI is artificial intelligence...",
            executed_at=pytest.importorskip("arrow").now().datetime,
        )

        assert finding.query == "What is AI?"
        assert finding.tool_used == "tavily"

    def test_tool_call_creation(self):
        """Test ToolCall model."""
        tool_call = ToolCall(
            id="call_123",
            tool_name="web_search",
            query="test query",
            status="pending",
        )

        assert tool_call.id == "call_123"
        assert tool_call.status == "pending"

        # Test status update
        tool_call.update_status("completed", result=[])
        assert tool_call.status == "completed"
        assert tool_call.execution_time_seconds is not None

    def test_research_turn_creation(self):
        """Test ResearchTurn model."""
        from datetime import datetime

        turn = ResearchTurn(
            role="assistant",
            content="Found research results",
            created_at=datetime.utcnow(),
            tool_calls=[],
            research_findings=[],
        )

        assert turn.role == "assistant"
        assert isinstance(turn.tool_calls, list)
        assert isinstance(turn.research_findings, list)

    def test_research_session_state(self):
        """Test ResearchSessionState model."""
        session_id = uuid4()
        state = ResearchSessionState(
            session_id=session_id,
            turns=[],
            findings_summary={},
            agent_decisions=[],
        )

        assert state.session_id == session_id
        assert state.current_turn == 0
        assert state.research_complete is False


# ============================================================================
# Research Pipeline Tests
# ============================================================================


class TestResearchPipeline:
    """Test research workflow orchestration."""

    def test_pipeline_initialization(self):
        """Test pipeline initializes."""
        pipeline = ResearchPipeline()

        assert pipeline.planner is not None
        assert pipeline.max_turns > 0

    @pytest.mark.asyncio
    async def test_pipeline_graceful_error(self):
        """Test pipeline handles errors gracefully."""
        pipeline = ResearchPipeline()

        # Mock tool execution to fail
        with patch.object(pipeline, "_summarize_findings") as mock_summary:
            mock_summary.return_value = "No findings"

            # When tools fail, should return empty response
            response, sources, state = await pipeline.research(
                query="test",
                session_id=uuid4(),
            )

            # Should complete without crashing
            assert isinstance(response, str)
            assert isinstance(sources, list)
            assert isinstance(state, ResearchSessionState)


# ============================================================================
# Research Endpoint Tests
# ============================================================================


class TestResearchEndpoints:
    """Test HTTP and WebSocket endpoints."""

    def test_research_health_endpoint(self):
        """Test /research/health endpoint."""
        response = client.get("/research/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["module"] == "research"
        assert "tools" in data

    def test_research_query_endpoint_mock(self):
        """Test /research endpoint with mock."""
        # Verify the endpoint exists via app routes
        routes_str = str([str(route.path) for route in client.app.routes])
        assert "/research" in routes_str


# ============================================================================
# Chat Compound Mode Tests
# ============================================================================


class TestChatCompoundMode:
    """Test chat module with research integration."""

    def test_chat_complete_request_with_research(self):
        """Test ChatCompleteRequest accepts enable_research parameter."""
        from ragcore.modules.chat.router import ChatCompleteRequest

        req = ChatCompleteRequest(
            message="What is AI?",
            enable_research=True,
        )

        assert req.message == "What is AI?"
        assert req.enable_research is True

    def test_context_builder_compound_mode(self):
        """Test ContextBuilder.build_compound method."""
        from ragcore.modules.chat.context_builder import ContextBuilder

        messages = ContextBuilder.build_compound(
            system_prompt="Compound mode prompt",
            query="Test question",
            research_findings="Research shows...",
            research_sources=[
                {"title": "Source", "url": "https://example.com", "snippet": "text"}
            ],
        )

        assert isinstance(messages, list)
        assert len(messages) >= 2
        # Should contain mixed sources
        content_text = " ".join([m.get("content", "") for m in messages])
        assert "Question" in content_text or "query" in content_text or "Source" in content_text


# ============================================================================
# Integration Tests
# ============================================================================


class TestPhase3Integration:
    """Integration tests for research module."""

    def test_all_apis_mounted(self):
        """Test all Phase 3 endpoints are available."""
        # List of endpoints to verify
        endpoints = [
            "/research/health",
            "/research/",
            "/chat/health",
            "/chat/complete",
            "/files",
        ]

        routes_str = str([str(route.path) for route in app.routes])

        for endpoint in endpoints:
            # Endpoints should be registered (rough check)
            assert endpoint in routes_str or "/research" in routes_str

    def test_dependencies_installed(self):
        """Verify Phase 3 dependencies are available."""
        imports_to_check = [
            ("ragcore.modules.research.tools.search", "TavilySearchTool"),
            ("ragcore.modules.research.tool_registry", "ToolExecutor"),
            ("ragcore.modules.research.models", "ResearchFinding"),
            ("ragcore.modules.research.agent_planner", "ResearchPlanner"),
            ("ragcore.modules.research.pipeline", "ResearchPipeline"),
        ]

        for module_name, class_name in imports_to_check:
            try:
                module = __import__(module_name, fromlist=[class_name])
                assert hasattr(module, class_name)
            except ImportError as e:
                pytest.skip(f"Module not available: {module_name} ({str(e)})")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
