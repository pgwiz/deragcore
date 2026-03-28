"""Ollama local model provider adapter."""

import logging
from typing import AsyncGenerator, List, Dict, Any, Optional
from ragcore.core.providers.base import BaseProvider
from ragcore.core.schemas import UnifiedResponse, UnifiedChunk
from ragcore.config import settings

logger = logging.getLogger(__name__)


class OllamaProvider(BaseProvider):
    """Adapter for Ollama local models (fallback / offline)."""

    def __init__(self):
        """Initialize Ollama client."""
        if not settings.enable_ollama:
            raise ValueError("Ollama not enabled in configuration")
        self.base_url = settings.ollama_base_url
        logger.info(f"Ollama provider initialized at {self.base_url}")

    def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
    ) -> UnifiedResponse:
        """Generate completion using local Ollama model."""
        try:
            import requests

            # Format message for Ollama
            text_content = "\n".join(m["content"] for m in messages)

            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": text_content,
                    "temperature": temperature,
                    "stream": False,
                },
                timeout=60,
            )

            if response.status_code != 200:
                raise RuntimeError(f"Ollama error: {response.text}")

            result = response.json()
            text = result.get("response", "")

            return UnifiedResponse(
                text=text,
                model=model,
                provider="ollama",
                input_tokens=0,  # Ollama doesn't track tokens
                output_tokens=0,
                raw=result,
            )
        except Exception as e:
            logger.error(f"Ollama completion error: {e}")
            raise

    async def stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[UnifiedChunk, None]:
        """Stream tokens from local Ollama model."""
        try:
            import requests

            text_content = "\n".join(m["content"] for m in messages)

            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": text_content,
                    "temperature": temperature,
                    "stream": True,
                },
                timeout=60,
                stream=True,
            )

            if response.status_code != 200:
                raise RuntimeError(f"Ollama error: {response.text}")

            for line in response.iter_lines():
                if line:
                    import json

                    chunk = json.loads(line)
                    text = chunk.get("response", "")
                    if text:
                        yield UnifiedChunk(
                            delta=text,
                            provider="ollama",
                            model=model,
                        )
        except Exception as e:
            logger.error(f"Ollama stream error: {e}")
            raise

    def embed(self, text: str, model: str) -> List[float]:
        """Ollama embedding support (if model supports it)."""
        try:
            import requests

            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": model, "input": text},
                timeout=30,
            )

            if response.status_code != 200:
                raise RuntimeError(f"Ollama embed error: {response.text}")

            result = response.json()
            return result.get("embeddings", [[]])[0]
        except Exception as e:
            logger.error(f"Ollama embed error: {e}")
            raise NotImplementedError("Ollama embedding not available")

    def list_models(self) -> List[str]:
        """Get list of available local Ollama models."""
        try:
            import requests

            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                result = response.json()
                return [m["name"] for m in result.get("models", [])]
        except Exception as e:
            logger.warning(f"Could not fetch Ollama models: {e}")
        return ["llama2", "mistral", "phi"]  # Defaults
