"""Tool registry and selection - Manages tool availability and fallback."""

import logging
from typing import List, Optional, Dict, Any

from ragcore.config import settings
from ragcore.modules.research.tools.search import (
    BaseSearchTool,
    SearchResult,
    SearchToolFactory,
)

logger = logging.getLogger(__name__)


class ToolExecutor:
    """Executes search tools with fallback strategy."""

    def __init__(self):
        """Initialize tool executor."""
        self.factory = SearchToolFactory()
        self.all_tools = self.factory.get_all_tools()
        self.available_tools = self.factory.get_available_tools()
        self.priority = settings.research_tool_priority

    def get_available_tools(self) -> Dict[str, BaseSearchTool]:
        """Get all available tools with credentials."""
        return self.available_tools

    def get_tool_status(self) -> Dict[str, bool]:
        """Get availability status for all tools."""
        return {
            name: tool.is_available() for name, tool in self.all_tools.items()
        }

    def select_tool_for_query(self, query: str) -> Optional[str]:
        """
        Select best tool for query using priority order.

        Returns tool name or None if no tools available.
        """
        # Try tools in priority order
        for tool_name in self.priority:
            tool = self.factory.get_tool(tool_name)
            if tool:
                logger.debug(f"Selected tool for query: {tool_name}")
                return tool_name

        logger.warning("No search tools available")
        return None

    async def execute_with_fallback(
        self,
        query: str,
        max_results: int = 5,
        max_attempts: int = 3,
    ) -> tuple[List[SearchResult], str]:
        """
        Execute search with automatic fallback to next tool.

        Tries tools in priority order until success.

        Args:
            query: Search query
            max_results: Max results per search
            max_attempts: Max number of tools to try before giving up

        Returns:
            Tuple of (results, tool_used_name)
        """
        attempts = 0

        for tool_name in self.priority:
            if attempts >= max_attempts:
                logger.warning(f"Reached max attempts ({max_attempts}) for query: {query[:50]}")
                break

            tool = self.factory.get_tool(tool_name)
            if not tool:
                logger.debug(f"Tool '{tool_name}' not available, trying next")
                continue

            attempts += 1

            try:
                logger.info(f"Executing search with {tool_name}: query='{query[:50]}'")

                results = await tool.search(query=query, max_results=max_results)

                if results:
                    logger.info(
                        f"Success with {tool_name}: Got {len(results)} results "
                        f"(attempt {attempts}/{max_attempts})"
                    )
                    return results, tool_name

                logger.debug(f"{tool_name} returned empty results, trying next tool")

            except Exception as e:
                logger.error(
                    f"Tool '{tool_name}' failed (attempt {attempts}/{max_attempts}): {str(e)}"
                )
                continue

        logger.error(f"All search tools failed for query: {query[:50]}")
        return [], "none"

    async def execute_specific_tool(
        self,
        tool_name: str,
        query: str,
        max_results: int = 5,
    ) -> tuple[List[SearchResult], str]:
        """
        Execute specific tool, no fallback.

        Args:
            tool_name: Tool to use ("tavily", "serpapi", "duckduckgo", "gpt-researcher")
            query: Search query
            max_results: Max results

        Returns:
            Tuple of (results, tool_used_name)
        """
        tool = self.factory.get_tool(tool_name)

        if not tool:
            logger.warning(f"Tool '{tool_name}' not available")
            return [], "none"

        try:
            logger.info(f"Executing {tool_name}: query='{query[:50]}'")
            results = await tool.search(query=query, max_results=max_results)
            logger.info(f"{tool_name} returned {len(results)} results")
            return results, tool_name

        except Exception as e:
            logger.error(f"Tool '{tool_name}' failed: {str(e)}")
            return [], "none"

    async def aggregate_results(
        self,
        results_list: List[tuple[List[SearchResult], str]],
        deduplicate: bool = True,
        max_total: int = 10,
    ) -> List[SearchResult]:
        """
        Aggregate results from multiple tool executions.

        Args:
            results_list: List of (results, tool_name) tuples
            deduplicate: Remove duplicate URLs
            max_total: Max results to return

        Returns:
            Aggregated and sorted results
        """
        aggregated = []
        seen_urls = set()

        for results, tool_name in results_list:
            for result in results:
                # Deduplicate by URL
                if deduplicate and result.url in seen_urls:
                    logger.debug(f"Skipping duplicate: {result.url}")
                    continue

                seen_urls.add(result.url)
                aggregated.append(result)

                if len(aggregated) >= max_total:
                    break

            if len(aggregated) >= max_total:
                break

        # Sort by relevance score (descending)
        aggregated.sort(key=lambda x: x.relevance_score, reverse=True)

        logger.info(
            f"Aggregated {len(aggregated)} results from {len(results_list)} tool executions"
        )
        return aggregated[:max_total]

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Get tool definitions for agent planning.

        Returns structured tool metadata for agent to understand capabilities.
        """
        return [
            {
                "name": "web_search_tavily",
                "description": "Search the web using Tavily - optimized for RAG, structured results",
                "available": self.factory.get_tool("tavily") is not None,
                "priority": 1,
            },
            {
                "name": "web_search_serpapi",
                "description": "Search via SerpAPI (Google Search) - broad coverage",
                "available": self.factory.get_tool("serpapi") is not None,
                "priority": 2,
            },
            {
                "name": "web_search_duckduckgo",
                "description": "Search via DuckDuckGo - free, no API key required",
                "available": self.factory.get_tool("duckduckgo") is not None,
                "priority": 3,
            },
            {
                "name": "deep_research_gpt",
                "description": "Multi-source deep research and report generation",
                "available": self.factory.get_tool("gpt-researcher") is not None,
                "priority": 4,
            },
        ]


# Global executor instance
executor = ToolExecutor()
