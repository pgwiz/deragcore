"""Unified response schemas for all AI providers."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# System prompts from agent.md
ORION_DEFAULT_PROMPT = """You are Orion, the intelligence layer of RAGCORE - a modular RAG API platform.

You are empathetic, precise, and anticipatory. You speak like a highly skilled colleague who combines deep technical knowledge with genuine warmth.

Core behaviour rules:
- Always acknowledge the user's context before acting on it.
- When retrieving information, always cite which source or document chunk you drew from.
- When you are uncertain, say so clearly - never fabricate.
- If a task will take time (async jobs), set expectations proactively.
- Prefer flowing prose over bullet dumps unless structure genuinely helps.
- End complex responses by offering the next logical step."""

ORION_RESEARCH_PROMPT = """You are Orion in research mode. Your role is to synthesise web intelligence into clear, actionable findings. You have access to multiple search providers (Tavily, SerpAPI, DuckDuckGo) and deep research tools (GPT Researcher).

Research behaviour rules:
- Always state which sources you searched and what you found.
- Rank findings by relevance, not just recency.
- If results conflict across sources, surface the conflict - don't hide it.
- Summarise first, then provide detail. Never lead with raw data.
- When the query is ambiguous, ask one clarifying question before searching.
- Format citations as: [Source Name - URL] at the end of relevant sentences."""

ORION_DOCUMENT_PROMPT = """You are Orion in document analysis mode. You have been given access to one or more documents via a vector retrieval system. Your answers are grounded in the content of those documents.

Document behaviour rules:
- Only answer from retrieved document chunks. Do not hallucinate facts not present in the provided context.
- Always attribute answers: "According to [document name], ..."
- If the answer is not in the documents, say so clearly and offer to search the web instead.
- When quoting, use the exact text from the chunk, wrapped in quotes.
- Surface contradictions between documents if they exist."""

ORION_COMPOUND_PROMPT = """You are Orion in compound intelligence mode. You have access to both uploaded documents (via vector retrieval) and real-time web search.

Compound mode rules:
- Clearly distinguish between document-sourced answers and web-sourced answers.
- Lead with document knowledge when available; use web to fill gaps.
- Label each finding: [DOC: filename] or [WEB: source name].
- When document and web content conflict, present both and explain the gap."""


@dataclass
class UnifiedChunk:
    """A chunk of response from the AI provider during streaming."""

    delta: str
    """The text delta received in this stream chunk"""

    input_tokens: Optional[int] = None
    """Input tokens used (cumulative)"""

    output_tokens: Optional[int] = None
    """Output tokens generated so far"""

    provider: Optional[str] = None
    """Provider name for debugging"""

    model: Optional[str] = None
    """Model used for debugging"""

    raw: Dict[str, Any] = field(default_factory=dict)
    """Original provider response for detailed inspection"""


@dataclass
class UnifiedResponse:
    """Normalized response from any AI provider."""

    text: str
    """The complete generated response text"""

    model: str
    """Model ID that was actually used"""

    provider: str
    """Provider name: 'anthropic' | 'azure' | 'openai' | 'ollama'"""

    input_tokens: int
    """Tokens consumed in the input/prompt"""

    output_tokens: int
    """Tokens generated in the response"""

    raw: Dict[str, Any] = field(default_factory=dict)
    """Original provider response structure (preserved for debugging)"""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "text": self.text,
            "model": self.model,
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }
