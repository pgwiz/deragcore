"""Token counting utility - Accurate per-message and per-component token counting."""

import logging
from typing import List, Dict, Any, Optional
import tiktoken

logger = logging.getLogger(__name__)


class TokenCounter:
    """Accurate token counting using cl100k_base tokenizer (GPT-3.5/4)."""

    def __init__(self, model: str = "gpt-3.5-turbo"):
        """Initialize token counter.

        Args:
            model: Model to use for tokenizer selection (default: gpt-3.5-turbo for cl100k_base)
        """
        self.model = model
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        logger.debug(f"TokenCounter initialized with {model}")

    def count_tokens(self, text: str) -> int:
        """Count exact number of tokens in text.

        Args:
            text: Text to count

        Returns:
            Number of tokens
        """
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Count total tokens in a message list.

        This follows OpenAI's token counting: ~4 tokens per message overhead.

        Args:
            messages: List of messages with 'role' and 'content'

        Returns:
            Total token count
        """
        if not messages:
            return 0

        total = 0
        for message in messages:
            # Each message has ~4 tokens overhead (role, separators, etc)
            total += 4
            if "content" in message:
                total += self.count_tokens(message["content"])

        # Add ~2 tokens for message list framing
        total += 2
        return total

    def get_component_breakdown(
        self,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        chunks: Optional[List[str]] = None,
        query: Optional[str] = None,
    ) -> Dict[str, int]:
        """Get token count breakdown by component.

        Useful for debugging and prioritization.

        Args:
            system_prompt: System instruction
            history: List of prior messages
            chunks: List of retrieved document chunks
            query: Current user query

        Returns:
            Dictionary with token counts: {system_prompt, history, chunks, query, total}
        """
        breakdown = {
            "system_prompt": 0,
            "history": 0,
            "chunks": 0,
            "query": 0,
            "total": 0,
        }

        if system_prompt:
            breakdown["system_prompt"] = self.count_tokens(system_prompt)

        if history:
            breakdown["history"] = self.count_messages_tokens(history)

        if chunks:
            breakdown["chunks"] = sum(self.count_tokens(chunk) for chunk in chunks)

        if query:
            breakdown["query"] = self.count_tokens(query)

        breakdown["total"] = sum(
            breakdown[k] for k in ["system_prompt", "history", "chunks", "query"]
        )

        return breakdown

    def estimate_text_tokens(self, text: str) -> int:
        """Fast estimate of tokens (fallback for very large texts).

        Uses heuristic: ~1 token per 4 characters (cl100k_base average).

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        if not text:
            return 0
        return max(1, len(text) // 4)
