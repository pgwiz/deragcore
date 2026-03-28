"""Token budget tracking - Track and enforce context window limits."""

import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class TokenBudget:
    """Track token usage and enforce context window limits."""

    def __init__(
        self,
        context_window_size: int = 200000,
        output_buffer_percentage: float = 0.15,
        compression_threshold: float = 0.85,
    ):
        """Initialize token budget.

        Args:
            context_window_size: Total context window in tokens (e.g., 200000 for Claude 3.5)
            output_buffer_percentage: Fraction of context reserved for output (e.g., 0.15 = 15%)
            compression_threshold: Trigger compression when usage exceeds this fraction (e.g., 0.85 = 85%)
        """
        self.context_window_size = context_window_size
        self.output_buffer_percentage = output_buffer_percentage
        self.compression_threshold = compression_threshold

        # Calculate derived values
        self.buffer_tokens = int(context_window_size * output_buffer_percentage)
        self.available_tokens = context_window_size - self.buffer_tokens
        self.compression_trigger_tokens = int(self.available_tokens * compression_threshold)

        # Track current usage
        self.current_usage = 0
        self.last_updated = datetime.utcnow()

        logger.debug(
            f"TokenBudget: window={context_window_size}, "
            f"buffer={self.buffer_tokens}, "
            f"available={self.available_tokens}, "
            f"compression_at={self.compression_trigger_tokens}"
        )

    def add_tokens(self, tokens: int) -> None:
        """Track token usage.

        Args:
            tokens: Tokens to add to current usage
        """
        self.current_usage += tokens
        self.last_updated = datetime.utcnow()

    def reset(self) -> None:
        """Reset token counter for new request."""
        self.current_usage = 0
        self.last_updated = datetime.utcnow()

    def remaining_tokens(self) -> int:
        """Get remaining tokens available for input.

        Returns:
            Remaining tokens (can be negative if over budget)
        """
        return self.available_tokens - self.current_usage

    def is_over_budget(self) -> bool:
        """Check if current usage exceeds available tokens.

        Returns:
            True if over budget
        """
        return self.current_usage > self.available_tokens

    def is_under_pressure(self) -> bool:
        """Check if under compression threshold (85%+ full).

        Returns:
            True if compression should be triggered
        """
        return self.current_usage >= self.compression_trigger_tokens

    def get_usage_percentage(self) -> float:
        """Get current usage as percentage of available tokens.

        Returns:
            Percentage (0.0 to 1.0+)
        """
        if self.available_tokens == 0:
            return 0.0
        return self.current_usage / self.available_tokens

    def get_status(self) -> dict:
        """Get detailed status breakdown.

        Returns:
            Dict with status information
        """
        return {
            "context_window": self.context_window_size,
            "output_buffer": self.buffer_tokens,
            "available_for_input": self.available_tokens,
            "compression_threshold": self.compression_trigger_tokens,
            "current_usage": self.current_usage,
            "remaining": self.remaining_tokens(),
            "usage_percentage": round(self.get_usage_percentage() * 100, 1),
            "is_over_budget": self.is_over_budget(),
            "is_under_pressure": self.is_under_pressure(),
            "last_updated": self.last_updated.isoformat(),
        }
