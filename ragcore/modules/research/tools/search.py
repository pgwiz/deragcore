"""Web search tools - Tavily, SerpAPI, DuckDuckGo, GPT Researcher."""

import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from datetime import datetime

from ragcore.config import settings

logger = logging.getLogger(__name__)


# ============================================================================
# Search Result Model
# ============================================================================


class SearchResult:
    """Single search result from any search provider."""

    def __init__(
        self,
        title: str,
        url: str,
        snippet: str,
        source: str,
        relevance_score: float = 0.5,
        published_date: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.title = title
        self.url = url
        self.snippet = snippet
        self.source = source  # "tavily" | "serpapi" | "duckduckgo" | "gpt-researcher"
        self.relevance_score = relevance_score
        self.published_date = published_date
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "relevance_score": self.relevance_score,
            "published_date": self.published_date,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return f"<SearchResult [{self.source}] '{self.title[:40]}...' score={self.relevance_score:.2f}>"


# ============================================================================
# Base Search Tool Interface
# ============================================================================


class BaseSearchTool(ABC):
    """Abstract base class for search tools."""

    @abstractmethod
    async def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> List[SearchResult]:
        """
        Search for query and return results.

        Args:
            query: Search query string
            max_results: Maximum results to return

        Returns:
            List of SearchResult objects
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if tool has required API key/config."""
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"


# ============================================================================
# Tavily Search Tool (Primary)
# ============================================================================


class TavilySearchTool(BaseSearchTool):
    """Tavily web search - optimized for RAG, structured results."""

    def __init__(self):
        """Initialize Tavily tool."""
        self.api_key = settings.tavily_api_key
        self.base_url = "https://api.tavily.com"

    def is_available(self) -> bool:
        """Check if Tavily API key is configured."""
        return self.api_key is not None

    async def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> List[SearchResult]:
        """
        Search using Tavily API.

        Args:
            query: Search query
            max_results: Max results to return

        Returns:
            List of SearchResult objects
        """
        if not self.is_available():
            logger.warning("Tavily API key not configured")
            return []

        try:
            from tavily import AsyncTavily

            client = AsyncTavily(api_key=self.api_key)

            # Tavily search with RAG optimization
            response = await client.search(
                query=query,
                max_results=max_results,
                include_answer=True,
            )

            results = []

            # Process Tavily results
            for item in response.get("results", []):
                result = SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    source="tavily",
                    relevance_score=float(item.get("score", 0.5)),
                    metadata={"published_date": item.get("published_date")},
                )
                results.append(result)

            logger.info(f"Tavily: Retrieved {len(results)} results for '{query[:50]}'")
            return results

        except Exception as e:
            logger.error(f"Tavily search error: {str(e)}")
            return []


# ============================================================================
# SerpAPI Search Tool (Fallback 1)
# ============================================================================


class SerpAPISearchTool(BaseSearchTool):
    """SerpAPI search - broader coverage via Google, Bing, Baidu."""

    def __init__(self):
        """Initialize SerpAPI tool."""
        self.api_key = settings.serpapi_api_key

    def is_available(self) -> bool:
        """Check if SerpAPI key is configured."""
        return self.api_key is not None

    async def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> List[SearchResult]:
        """
        Search using SerpAPI.

        Args:
            query: Search query
            max_results: Max results to return

        Returns:
            List of SearchResult objects
        """
        if not self.is_available():
            logger.warning("SerpAPI key not configured")
            return []

        try:
            from serpapi import GoogleSearch

            # SerpAPI Google search
            params = {
                "q": query,
                "api_key": self.api_key,
                "num": max_results,
            }

            search = GoogleSearch(params)
            data = search.get_dict()

            results = []

            # Process Google organic results
            for item in data.get("organic_results", []):
                result = SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    source="serpapi",
                    relevance_score=1.0 / (item.get("position", 10)),  # Rank-based score
                    metadata={
                        "position": item.get("position"),
                        "date": item.get("date"),
                    },
                )
                results.append(result)

            logger.info(f"SerpAPI: Retrieved {len(results)} results for '{query[:50]}'")
            return results

        except Exception as e:
            logger.error(f"SerpAPI search error: {str(e)}")
            return []


# ============================================================================
# DuckDuckGo Search Tool (Fallback 2 - Free)
# ============================================================================


class DuckDuckGoSearchTool(BaseSearchTool):
    """DuckDuckGo search - free fallback, no API key needed."""

    def is_available(self) -> bool:
        """DuckDuckGo always available (no API key required)."""
        return True

    async def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> List[SearchResult]:
        """
        Search using DuckDuckGo.

        Args:
            query: Search query
            max_results: Max results to return

        Returns:
            List of SearchResult objects
        """
        try:
            from duckduckgo_search import DDGS

            results = []

            with DDGS() as ddgs:
                # DuckDuckGo instant search
                search_results = ddgs.text(
                    keywords=query,
                    max_results=max_results,
                )

                for i, item in enumerate(search_results, 1):
                    result = SearchResult(
                        title=item.get("title", ""),
                        url=item.get("href", ""),
                        snippet=item.get("body", ""),
                        source="duckduckgo",
                        relevance_score=1.0 / i,  # Position-based score
                        metadata={},
                    )
                    results.append(result)

            logger.info(f"DuckDuckGo: Retrieved {len(results)} results for '{query[:50]}'")
            return results

        except Exception as e:
            logger.error(f"DuckDuckGo search error: {str(e)}")
            return []


# ============================================================================
# GPT Researcher Tool (Deep Research)
# ============================================================================


class GPTResearcherTool(BaseSearchTool):
    """GPT Researcher - multi-source deep research and report generation."""

    def __init__(self):
        """Initialize GPT Researcher tool."""
        self.enabled = settings.gpt_researcher_enabled

    def is_available(self) -> bool:
        """Check if GPT Researcher is enabled."""
        return self.enabled

    async def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> List[SearchResult]:
        """
        Perform deep research using GPT Researcher.

        Args:
            query: Research query
            max_results: Max unique sources to return

        Returns:
            List of SearchResult objects with synthesized findings
        """
        if not self.is_available():
            logger.warning("GPT Researcher not enabled")
            return []

        try:
            from gpt_researcher import ResearchAgent

            # Initialize research agent
            researcher = ResearchAgent(
                query=query,
                report_type="research_report",
                tone="objective",
                max_iterations=3,
            )

            # Run research
            research_result = await researcher.conduct_research()

            # Parse findings into SearchResult format
            results = []

            # Extract sources from research report
            if hasattr(research_result, "sources"):
                for i, source in enumerate(
                    research_result.sources[:max_results],
                    1,
                ):
                    result = SearchResult(
                        title=source.get("title", "Research Finding"),
                        url=source.get("url", f"research://{i}"),
                        snippet=source.get("snippet", research_result.report[:200]),
                        source="gpt-researcher",
                        relevance_score=0.95,  # High confidence from research agent
                        metadata={
                            "research_type": "deep",
                            "report_snippet": research_result.report[:500],
                        },
                    )
                    results.append(result)

            logger.info(
                f"GPT Researcher: Completed research for '{query[:50]}' "
                f"with {len(results)} sources"
            )
            return results

        except Exception as e:
            logger.error(f"GPT Researcher error: {str(e)}")
            return []


# ============================================================================
# Tool Factory
# ============================================================================


class SearchToolFactory:
    """Factory for creating and managing search tools."""

    @staticmethod
    def get_all_tools() -> Dict[str, BaseSearchTool]:
        """Get all available search tools."""
        return {
            "tavily": TavilySearchTool(),
            "serpapi": SerpAPISearchTool(),
            "duckduckgo": DuckDuckGoSearchTool(),
            "gpt-researcher": GPTResearcherTool(),
        }

    @staticmethod
    def get_available_tools() -> Dict[str, BaseSearchTool]:
        """Get tools that have required credentials/config."""
        all_tools = SearchToolFactory.get_all_tools()
        return {
            name: tool for name, tool in all_tools.items() if tool.is_available()
        }

    @staticmethod
    def get_tool(tool_name: str) -> Optional[BaseSearchTool]:
        """Get specific tool by name if available."""
        all_tools = SearchToolFactory.get_all_tools()
        tool = all_tools.get(tool_name)

        if tool and tool.is_available():
            return tool

        logger.warning(f"Tool '{tool_name}' not available")
        return None
