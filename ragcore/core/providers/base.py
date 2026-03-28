"""Base provider abstract class - all AI providers implement this interface."""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Dict, Any, Optional
from ragcore.core.schemas import UnifiedResponse, UnifiedChunk


class BaseProvider(ABC):
    """
    Abstract base class for all AI provider adapters.

    Every provider (Anthropic, Azure, OpenAI, Ollama) must implement
    these four methods to be compatible with the AIController.
    """

    @abstractmethod
    def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
    ) -> UnifiedResponse:
        """
        Generate a single completion from a list of messages.

        Args:
            messages: List of dicts with 'role' and 'content'
            model: Model ID to use
            temperature: Sampling temperature 0.0-2.0
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system instruction

        Returns:
            UnifiedResponse with text, tokens, and metadata
        """
        pass

    @abstractmethod
    async def stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[UnifiedChunk, None]:
        """
        Stream tokens from a completion as they arrive.

        Args:
            messages: List of dicts with 'role' and 'content'
            model: Model ID to use
            temperature: Sampling temperature 0.0-2.0
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system instruction

        Yields:
            UnifiedChunk objects for each token/event
        """
        pass

    @abstractmethod
    def embed(
        self,
        text: str,
        model: str,
    ) -> List[float]:
        """
        Generate embedding vector for text.

        Args:
            text: Text to embed
            model: Embedding model to use

        Returns:
            List of floats representing the embedding vector
        """
        pass

    @abstractmethod
    def list_models(self) -> List[str]:
        """
        List all available models for this provider.

        Returns:
            List of model IDs supported by this provider
        """
        pass

    def get_name(self) -> str:
        """Get the provider name for logging/identification."""
        return self.__class__.__name__.replace("Provider", "").lower()
