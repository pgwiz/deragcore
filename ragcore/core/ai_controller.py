"""AI Controller - routes requests to correct provider and normalizes responses."""

import logging
from typing import AsyncGenerator, List, Dict, Any, Optional
from ragcore.core.provider_registry import registry
from ragcore.core.schemas import UnifiedResponse, UnifiedChunk

logger = logging.getLogger(__name__)


class AIController:
    """
    Central controller for all AI requests.

    Routes each request to the correct provider based on ModelConfig,
    normalizes all responses, and handles errors consistently.
    """

    @staticmethod
    async def stream(
        provider_name: str,
        model_id: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[UnifiedChunk, None]:
        """
        Stream tokens from an AI provider.

        Args:
            provider_name: 'anthropic', 'azure', 'openai', 'ollama'
            model_id: Model identifier for that provider
            messages: Conversation messages
            temperature: Sampling temperature
            max_tokens: Max generation length
            system_prompt: Optional system instruction

        Yields:
            UnifiedChunk objects with streaming tokens
        """
        provider = registry.get_provider(provider_name)
        if provider is None:
            raise ValueError(f"Provider '{provider_name}' not available")

        logger.debug(
            f"Streaming from {provider_name}/{model_id} with {len(messages)} messages"
        )

        async for chunk in provider.stream(
            messages=messages,
            model=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        ):
            yield chunk

    @staticmethod
    def complete(
        provider_name: str,
        model_id: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
    ) -> UnifiedResponse:
        """
        Generate a single completion from an AI provider.

        Args:
            provider_name: 'anthropic', 'azure', 'openai', 'ollama'
            model_id: Model identifier for that provider
            messages: Conversation messages
            temperature: Sampling temperature
            max_tokens: Max generation length
            system_prompt: Optional system instruction

        Returns:
            UnifiedResponse with text, tokens, and metadata
        """
        provider = registry.get_provider(provider_name)
        if provider is None:
            raise ValueError(f"Provider '{provider_name}' not available")

        logger.debug(
            f"Completing from {provider_name}/{model_id} with {len(messages)} messages"
        )

        response = provider.complete(
            messages=messages,
            model=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )

        logger.debug(
            f"Response: {response.model} ({response.provider}) - "
            f"{response.input_tokens}in/{response.output_tokens}out"
        )

        return response

    @staticmethod
    def embed(provider_name: str, model_id: str, text: str) -> List[float]:
        """
        Generate an embedding vector from an AI provider.

        Args:
            provider_name: 'anthropic', 'azure', 'openai', 'ollama'
            model_id: Embedding model identifier
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        provider = registry.get_provider(provider_name)
        if provider is None:
            raise ValueError(f"Provider '{provider_name}' not available")

        logger.debug(f"Embedding from {provider_name}/{model_id}: {len(text)} chars")

        embedding = provider.embed(text=text, model=model_id)
        return embedding

    @staticmethod
    def get_available_providers() -> Dict[str, bool]:
        """Get status of all providers."""
        return registry.list_available_providers()
