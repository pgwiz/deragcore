"""OpenAI provider adapter."""

import logging
from typing import AsyncGenerator, List, Dict, Any, Optional
import openai
from ragcore.core.providers.base import BaseProvider
from ragcore.core.schemas import UnifiedResponse, UnifiedChunk
from ragcore.config import settings

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """Adapter for OpenAI's GPT models."""

    def __init__(self):
        """Initialize OpenAI client from API key."""
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY not configured")
        self.client = openai.OpenAI(api_key=settings.openai_api_key)

    def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
    ) -> UnifiedResponse:
        """Generate a single completion using GPT."""
        try:
            if system_prompt:
                messages = [{"role": "system", "content": system_prompt}] + messages

            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            text = response.choices[0].message.content or ""

            return UnifiedResponse(
                text=text,
                model=model,
                provider="openai",
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                raw={
                    "finish_reason": response.choices[0].finish_reason,
                    "id": response.id,
                },
            )
        except Exception as e:
            logger.error(f"OpenAI completion error: {e}")
            raise

    async def stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[UnifiedChunk, None]:
        """Stream tokens from GPT as they arrive."""
        try:
            if system_prompt:
                messages = [{"role": "system", "content": system_prompt}] + messages

            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield UnifiedChunk(
                        delta=chunk.choices[0].delta.content,
                        provider="openai",
                        model=model,
                    )
        except Exception as e:
            logger.error(f"OpenAI stream error: {e}")
            raise

    def embed(self, text: str, model: str) -> List[float]:
        """Generate embedding using OpenAI."""
        try:
            response = self.client.embeddings.create(input=text, model=model)
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise

    def list_models(self) -> List[str]:
        """Return list of available OpenAI models."""
        return [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
            "text-embedding-3-small",
            "text-embedding-3-large",
        ]
