"""Anthropic Claude provider adapter."""

import logging
from typing import AsyncGenerator, List, Dict, Any, Optional
import anthropic
from ragcore.core.providers.base import BaseProvider
from ragcore.core.schemas import UnifiedResponse, UnifiedChunk
from ragcore.config import settings

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseProvider):
    """Adapter for Anthropic's Claude models."""

    def __init__(self):
        """Initialize Anthropic client from API key."""
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not configured")
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
    ) -> UnifiedResponse:
        """Generate a single completion using Claude."""
        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt or "You are a helpful assistant.",
                messages=messages,
            )

            text = "".join(block.text for block in response.content if hasattr(block, "text"))

            return UnifiedResponse(
                text=text,
                model=model,
                provider="anthropic",
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                raw=response.model_dump(),
            )
        except Exception as e:
            logger.error(f"Anthropic completion error: {e}")
            raise

    async def stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[UnifiedChunk, None]:
        """Stream tokens from Claude as they arrive."""
        try:
            with self.client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt or "You are a helpful assistant.",
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    yield UnifiedChunk(
                        delta=text,
                        provider="anthropic",
                        model=model,
                    )
        except Exception as e:
            logger.error(f"Anthropic stream error: {e}")
            raise

    def embed(self, text: str, model: str) -> List[float]:
        """Anthropic does not provide embedding models."""
        raise NotImplementedError("Anthropic provider does not support embeddings")

    def list_models(self) -> List[str]:
        """Return list of available Claude models."""
        return [
            "claude-opus-4-1",
            "claude-sonnet-4",
            "claude-haiku-4",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
        ]
