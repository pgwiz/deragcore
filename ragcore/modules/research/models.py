"""Research data models - SearchResult, ResearchFinding, ToolCall."""

import logging
from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime
from dataclasses import dataclass, asdict

from ragcore.modules.chat.history import ChatTurn

logger = logging.getLogger(__name__)


# ============================================================================
# Research Finding Model
# ============================================================================


@dataclass
class ResearchFinding:
    """Results from a single research query execution."""

    query: str
    """Original search query"""

    results: List[Dict[str, Any]]
    """Search results from tool"""

    tool_used: str
    """Which tool executed: 'tavily' | 'serpapi' | 'duckduckgo' | 'gpt-researcher'"""

    synthesis: str
    """Agent's synthesis of findings"""

    executed_at: datetime
    """When this search was executed"""

    confidence_score: float = 0.7
    """How confident agent is in these results (0-1)"""

    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return {
            "query": self.query,
            "results": self.results,
            "tool_used": self.tool_used,
            "synthesis": self.synthesis,
            "executed_at": self.executed_at.isoformat(),
            "confidence_score": self.confidence_score,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"<ResearchFinding query='{self.query[:40]}' "
            f"tool={self.tool_used} results={len(self.results)}>"
        )


# ============================================================================
# Tool Call Model
# ============================================================================


@dataclass
class ToolCall:
    """Track a single tool invocation in research workflow."""

    id: str  # Unique ID for this tool call
    """Unique identifier for this tool call"""

    tool_name: str
    """Which tool: 'web_search' | 'deep_research' | etc."""

    query: str
    """What was queried"""

    status: str
    """'pending' | 'executing' | 'completed' | 'failed'"""

    result: Optional[List[Dict[str, Any]]] = None
    """Tool execution result"""

    error: Optional[str] = None
    """Error message if failed"""

    created_at: datetime = None
    """When tool call was created"""

    completed_at: Optional[datetime] = None
    """When tool call finished"""

    execution_time_seconds: Optional[float] = None
    """How long execution took"""

    metadata: Dict[str, Any] = None
    """Additional metadata"""

    def __post_init__(self):
        """Initialize defaults."""
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}

    def update_status(
        self,
        status: str,
        result: Optional[List[Dict[str, Any]]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Update tool call status."""
        self.status = status

        if result is not None:
            self.result = result
        if error is not None:
            self.error = error

        if status == "completed":
            self.completed_at = datetime.utcnow()
            if self.created_at:
                self.execution_time_seconds = (
                    self.completed_at - self.created_at
                ).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return {
            "id": self.id,
            "tool_name": self.tool_name,
            "query": self.query,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "execution_time_seconds": self.execution_time_seconds,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"<ToolCall id={self.id[:8]} tool={self.tool_name} "
            f"status={self.status} query='{self.query[:30]}'>"
        )


# ============================================================================
# Tool Result Model
# ============================================================================


@dataclass
class ToolResult:
    """Result returned from tool execution."""

    tool_call_id: str
    """Reference to ToolCall.id"""

    content: str
    """Text result from tool"""

    sources: List[Dict[str, str]]
    """Source references: {title, url, snippet}"""

    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return {
            "tool_call_id": self.tool_call_id,
            "content": self.content,
            "sources": self.sources,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"<ToolResult tool_call={self.tool_call_id[:8]} "
            f"sources={len(self.sources)}>"
        )


# ============================================================================
# Research Turn Model (Extends ChatTurn)
# ============================================================================


class ResearchTurn(ChatTurn):
    """Chat turn with research-specific fields (tool calls, findings)."""

    def __init__(
        self,
        role: str,
        content: str,
        created_at: datetime,
        sources: Optional[List[UUID]] = None,
        tool_calls: Optional[List[ToolCall]] = None,
        tool_results: Optional[List[ToolResult]] = None,
        research_findings: Optional[List[ResearchFinding]] = None,
        agent_reasoning: Optional[str] = None,
    ):
        """Initialize research turn.

        Args:
            role: 'user' or 'assistant'
            content: Message content
            created_at: Timestamp
            sources: File IDs used
            tool_calls: Tool calls during this turn
            tool_results: Results from tools
            research_findings: Research finding summaries
            agent_reasoning: Agent's reasoning about next steps
        """
        super().__init__(role, content, created_at, sources)

        self.tool_calls = tool_calls or []
        self.tool_results = tool_results or []
        self.research_findings = research_findings or []
        self.agent_reasoning = agent_reasoning

    def add_tool_call(self, tool_call: ToolCall) -> None:
        """Add a tool call to this turn."""
        self.tool_calls.append(tool_call)

    def add_tool_result(self, tool_result: ToolResult) -> None:
        """Add a tool result to this turn."""
        self.tool_results.append(tool_result)

    def add_finding(self, finding: ResearchFinding) -> None:
        """Add a research finding to this turn."""
        self.research_findings.append(finding)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        base = super().to_dict()

        base.update(
            {
                "tool_calls": [tc.to_dict() for tc in self.tool_calls],
                "tool_results": [tr.to_dict() for tr in self.tool_results],
                "research_findings": [rf.to_dict() for rf in self.research_findings],
                "agent_reasoning": self.agent_reasoning,
            }
        )

        return base

    def __repr__(self) -> str:
        return (
            f"<ResearchTurn role={self.role} "
            f"tool_calls={len(self.tool_calls)} findings={len(self.research_findings)}>"
        )


# ============================================================================
# Research Session State Model
# ============================================================================


@dataclass
class ResearchSessionState:
    """Maintains state across multi-turn research workflow."""

    session_id: UUID
    """Session identifier"""

    turns: List[ResearchTurn]
    """All research turns in order"""

    findings_summary: Dict[str, ResearchFinding]
    """Map of query -> findings for deduplication"""

    agent_decisions: List[Dict[str, Any]]
    """Agent decision history: what it decided to do at each turn"""

    total_tool_calls: int = 0
    """Cumulative tool calls across all turns"""

    current_turn: int = 0
    """Current turn number (0-indexed)"""

    research_complete: bool = False
    """Whether research workflow is complete"""

    def add_turn(self, turn: ResearchTurn) -> None:
        """Add a turn to the research session."""
        self.turns.append(turn)
        self.current_turn += 1

        # Track findings
        for finding in turn.research_findings:
            key = f"{finding.query}_{finding.tool_used}"
            self.findings_summary[key] = finding

        # Track tool calls
        self.total_tool_calls += len(turn.tool_calls)

    def record_decision(
        self,
        decision: str,
        reasoning: str,
        metadata: Dict[str, Any] = None,
    ) -> None:
        """Record an agent decision."""
        decision_record = {
            "turn": self.current_turn,
            "decision": decision,  # "search_more", "search_different", "finalize"
            "reasoning": reasoning,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }
        self.agent_decisions.append(decision_record)

    def should_continue(self, max_turns: int) -> bool:
        """Check if research should continue."""
        # Continue if: not at max turns AND research not complete
        return self.current_turn < max_turns and not self.research_complete

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return {
            "session_id": str(self.session_id),
            "turns": [t.to_dict() for t in self.turns],
            "findings_summary": {
                k: v.to_dict() for k, v in self.findings_summary.items()
            },
            "agent_decisions": self.agent_decisions,
            "total_tool_calls": self.total_tool_calls,
            "current_turn": self.current_turn,
            "research_complete": self.research_complete,
        }

    def __repr__(self) -> str:
        return (
            f"<ResearchSessionState session={str(self.session_id)[:8]} "
            f"turns={len(self.turns)} findings={len(self.findings_summary)}>"
        )
