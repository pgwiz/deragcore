"""Azure AI Foundry provider adapter."""

import logging
import asyncio
import json
from typing import AsyncGenerator, List, Dict, Any, Optional
from ragcore.core.providers.base import BaseProvider
from ragcore.core.schemas import UnifiedResponse, UnifiedChunk
from ragcore.config import settings

logger = logging.getLogger(__name__)


class AzureProvider(BaseProvider):
    """Adapter for Azure AI Foundry models."""

    def __init__(self):
        """Initialize Azure client from credentials."""
        if not settings.azure_api_key or not settings.azure_endpoint:
            raise ValueError("AZURE_API_KEY and AZURE_ENDPOINT not configured")

        # Import here to avoid dependency if not used
        try:
            from azure.ai.inference import ChatCompletionsClient
            from azure.core.credentials import AzureKeyCredential

            self.client = ChatCompletionsClient(
                endpoint=settings.azure_endpoint,
                credential=AzureKeyCredential(settings.azure_api_key),
            )
        except ImportError:
            raise ImportError("azure-ai-inference package required for Azure provider")

    def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
    ) -> UnifiedResponse:
        """Generate a single completion using Azure model."""
        try:
            from azure.ai.inference import ChatCompletionsFunctionToolCall

            if system_prompt:
                messages = [{"role": "system", "content": system_prompt}] + messages

            response = self.client.complete(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            text = response.choices[0].message.content or ""

            return UnifiedResponse(
                text=text,
                model=model,
                provider="azure",
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                raw={"finish_reason": response.choices[0].finish_reason},
            )
        except Exception as e:
            logger.error(f"Azure completion error: {e}")
            raise

    async def stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[UnifiedChunk, None]:
        """Stream tokens from Azure model using async execution."""
        try:
            if system_prompt:
                messages = [{"role": "system", "content": system_prompt}] + messages

            # Use asyncio.to_thread to run the blocking stream API in a thread pool
            def _run_stream():
                """Run the blocking stream API in a separate thread."""
                return self.client.complete_stream(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            stream_response = await asyncio.to_thread(_run_stream)

            # Iterate through the stream and yield chunks
            input_tokens = 0
            output_tokens = 0
            for chunk in stream_response:
                if hasattr(chunk, 'choices') and chunk.choices:
                    choice = chunk.choices[0]
                    if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                        content = choice.delta.content
                        if content:
                            yield UnifiedChunk(
                                delta=content,
                                provider="azure",
                                model=model,
                            )

                # Track usage if available
                if hasattr(chunk, 'usage') and chunk.usage:
                    input_tokens = chunk.usage.prompt_tokens or input_tokens
                    output_tokens = chunk.usage.completion_tokens or output_tokens

            # Final chunk with token counts
            if output_tokens > 0:
                yield UnifiedChunk(
                    delta="",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    provider="azure",
                    model=model,
                )

        except Exception as e:
            logger.error(f"Azure stream error: {e}")
            # Fallback to non-streaming if stream API fails
            logger.warning("Azure streaming failed, falling back to complete()")
            response = self.complete(messages, model, temperature, max_tokens, system_prompt)
            yield UnifiedChunk(
                delta=response.text,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                provider="azure",
                model=model,
            )

    def embed(self, text: str, model: str) -> List[float]:
        """Generate embedding using Azure."""
        try:
            from azure.ai.inference import EmbeddingsClient

            client = EmbeddingsClient(
                endpoint=settings.azure_endpoint,
                credential=__import__("azure.core.credentials", fromlist=["AzureKeyCredential"])
                .AzureKeyCredential(settings.azure_api_key),
            )

            response = client.embed(input=[text], model=model)
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Azure embedding error: {e}")
            raise NotImplementedError("Azure embedding not properly configured")

    def list_models(self) -> List[str]:
        """Return list of available Azure models."""
        return [
            "phi-4",
            "llama-3.3",
            "qwen-2.5",
            "text-embedding-3-small",
            "text-embedding-3-large",
        ]
