"""Tool composition and binding for agents."""

import logging
from typing import Optional, Dict, Any, List, Callable
from uuid import UUID

logger = logging.getLogger(__name__)


class ToolDefinition:
    """Definition of an available tool."""

    def __init__(
        self,
        name: str,
        description: str,
        category: str,  # "search" | "retrieval" | "analysis" | "extraction"
        execute_func: Callable,
        required_params: List[str],
        optional_params: Dict[str, Any],
    ):
        """Initialize tool definition.

        Args:
            name: Tool identifier (e.g., "web_search")
            description: What the tool does
            category: Tool category for classification
            execute_func: Async callable that executes the tool
            required_params: Required parameters for execution
            optional_params: Optional parameters with defaults
        """
        self.name = name
        self.description = description
        self.category = category
        self.execute_func = execute_func
        self.required_params = required_params
        self.optional_params = optional_params

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "required_params": self.required_params,
            "optional_params": self.optional_params,
        }


class ToolResult:
    """Result from executing a tool."""

    def __init__(
        self,
        tool_name: str,
        status: str,  # "success" | "error" | "timeout"
        result: Any,
        error_message: Optional[str] = None,
        execution_time_ms: int = 0,
    ):
        """Initialize tool result.

        Args:
            tool_name: Name of executed tool
            status: Execution status
            result: Tool output (if successful)
            error_message: Error details (if failed)
            execution_time_ms: Time to execute
        """
        self.tool_name = tool_name
        self.status = status
        self.result = result
        self.error_message = error_message
        self.execution_time_ms = execution_time_ms

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for serialization."""
        return {
            "tool_name": self.tool_name,
            "status": self.status,
            "result": self.result,
            "error_message": self.error_message,
            "execution_time_ms": self.execution_time_ms,
        }


class ToolComposer:
    """Compose and execute tools for agents."""

    def __init__(self):
        """Initialize tool composer."""
        self.tools: Dict[str, ToolDefinition] = {}
        self.execution_timeout_seconds = 30
        logger.info("ToolComposer initialized")

    def register_tool(self, tool: ToolDefinition) -> None:
        """Register an available tool.

        Args:
            tool: ToolDefinition to register
        """
        self.tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def register_tools_batch(self, tools: List[ToolDefinition]) -> None:
        """Register multiple tools.

        Args:
            tools: List of ToolDefinitions
        """
        for tool in tools:
            self.register_tool(tool)
        logger.info(f"Registered {len(tools)} tools")

    def get_tool(self, tool_name: str) -> Optional[ToolDefinition]:
        """Get a tool by name.

        Args:
            tool_name: Name of tool

        Returns:
            ToolDefinition or None
        """
        return self.tools.get(tool_name)

    def get_tools_for_agent(self, agent_tools: List[str]) -> List[ToolDefinition]:
        """Get tool definitions for an agent.

        Args:
            agent_tools: List of tool names from agent config

        Returns:
            List of ToolDefinitions available to agent
        """
        available = []
        for tool_name in agent_tools:
            tool = self.get_tool(tool_name)
            if tool:
                available.append(tool)
            else:
                logger.warning(f"Tool '{tool_name}' not found in registry")

        return available

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools.

        Returns:
            List of tool definitions as dicts
        """
        return [tool.to_dict() for tool in self.tools.values()]

    def list_tools_by_category(self, category: str) -> List[ToolDefinition]:
        """List tools by category.

        Args:
            category: Tool category to filter by

        Returns:
            List of tools in category
        """
        return [tool for tool in self.tools.values() if tool.category == category]

    async def execute_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
        session_id: Optional[UUID] = None,
    ) -> ToolResult:
        """Execute a tool.

        Args:
            tool_name: Name of tool to execute
            params: Input parameters for tool
            session_id: Optional session context

        Returns:
            ToolResult with execution status and output
        """
        import time

        tool = self.get_tool(tool_name)
        if not tool:
            return ToolResult(
                tool_name=tool_name,
                status="error",
                result=None,
                error_message=f"Tool '{tool_name}' not found",
            )

        # Validate required parameters
        missing_params = [p for p in tool.required_params if p not in params]
        if missing_params:
            return ToolResult(
                tool_name=tool_name,
                status="error",
                result=None,
                error_message=f"Missing required parameters: {missing_params}",
            )

        # Execute tool
        start_time = time.time()
        try:
            logger.debug(f"Executing tool: {tool_name} with params: {params}")

            # Call tool's execute function
            result = await tool.execute_func(params=params, session_id=session_id)

            execution_time_ms = int((time.time() - start_time) * 1000)

            logger.info(
                f"Tool {tool_name} executed successfully in {execution_time_ms}ms"
            )

            return ToolResult(
                tool_name=tool_name,
                status="success",
                result=result,
                execution_time_ms=execution_time_ms,
            )

        except asyncio.TimeoutError:
            execution_time_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Tool {tool_name} timed out after {execution_time_ms}ms")
            return ToolResult(
                tool_name=tool_name,
                status="timeout",
                result=None,
                error_message=f"Tool execution timed out after {self.execution_timeout_seconds}s",
                execution_time_ms=execution_time_ms,
            )

        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            logger.error(
                f"Tool {tool_name} failed: {str(e)}", exc_info=True
            )
            return ToolResult(
                tool_name=tool_name,
                status="error",
                result=None,
                error_message=str(e),
                execution_time_ms=execution_time_ms,
            )

    async def execute_tools(
        self,
        tool_calls: List[Dict[str, Any]],
        session_id: Optional[UUID] = None,
    ) -> List[ToolResult]:
        """Execute multiple tool calls.

        Args:
            tool_calls: List of {tool_name, params} dicts
            session_id: Optional session context

        Returns:
            List of ToolResults
        """
        results = []
        for call in tool_calls:
            tool_name = call.get("tool_name")
            params = call.get("params", {})

            result = await self.execute_tool(tool_name, params, session_id)
            results.append(result)

        return results

    def format_tool_results_for_context(
        self, tool_results: List[ToolResult]
    ) -> str:
        """Format tool results for inclusion in agent context.

        Args:
            tool_results: Results from tool execution

        Returns:
            Formatted string for inclusion in system prompt
        """
        if not tool_results:
            return ""

        lines = ["## Tool Execution Results\n"]

        for result in tool_results:
            lines.append(f"### {result.tool_name}")
            lines.append(f"Status: {result.status}")

            if result.status == "success":
                # Format result based on type
                if isinstance(result.result, list):
                    lines.append(f"Results ({len(result.result)} items):")
                    for item in result.result[:5]:  # Limit to first 5
                        if isinstance(item, dict):
                            lines.append(f"  - {item.get('title', item)}")
                        else:
                            lines.append(f"  - {str(item)[:100]}")
                    if len(result.result) > 5:
                        lines.append(f"  ... and {len(result.result) - 5} more")
                elif isinstance(result.result, dict):
                    for key, value in list(result.result.items())[:5]:
                        lines.append(f"  {key}: {str(value)[:100]}")
                else:
                    lines.append(f"Result: {str(result.result)[:200]}")

            elif result.status == "error":
                lines.append(f"Error: {result.error_message}")
            elif result.status == "timeout":
                lines.append(f"Timeout: {result.error_message}")

            lines.append(f"Execution time: {result.execution_time_ms}ms\n")

        return "\n".join(lines)


# Import asyncio for timeout handling
import asyncio

# Global tool composer instance
tool_composer = ToolComposer()
