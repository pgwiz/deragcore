"""Embedding provider adapter - abstracts embedding generation across providers."""

from typing import Optional, List, Any, Dict
import logging
import aiohttp
import json

from ragcore.core.model_provider_registry import ProviderType, ModelCapability
from ragcore.modules.multimodal.providers.base_adapter import BaseProviderAdapter

logger = logging.getLogger(__name__)


class EmbeddingProviderAdapter(BaseProviderAdapter):
    """Adapter for embedding/vectorization capabilities.

    Supports:
    - OpenAI text-embedding-3-large (primary) - 1536 dimensions
    - Azure OpenAI embeddings
    - Azure Foundry embeddings (serverless)
    - Google Vertex AI embeddings
    - Anthropic (falls back to OpenAI)
    """

    def __init__(
        self,
        registry=None,
        embedding_dimension: int = 1536,
        model_id: Optional[str] = None,
    ):
        """Initialize embedding adapter.

        Args:
            registry: ModelProviderRegistry
            embedding_dimension: Target embedding dimension (default 1536)
            model_id: Optional specific embedding model to use
        """
        super().__init__(
            registry=registry,
            primary_provider=ProviderType.OPENAI,
            fallback_provider=ProviderType.AZURE_OPENAI,
        )
        self.embedding_dimension = embedding_dimension
        self.model_id = model_id or "text-embedding-3-large"

    async def embed_text(self, text: str) -> Optional[List[float]]:
        """Generate embedding for single text.

        Args:
            text: Text to embed

        Returns:
            List of floats (1536-dim) or None
        """
        if not text or not text.strip():
            return None

        embeddings = await self.embed_texts([text])
        return embeddings[0] if embeddings else None

    async def embed_texts(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors or None
        """
        if not texts:
            return None

        # Filter out empty texts
        non_empty = [t for t in texts if t and t.strip()]
        if not non_empty:
            return None

        provider = self.get_available_provider()
        if not provider:
            logger.error("No embedding provider available")
            return None

        self.last_used_provider = provider

        try:
            if provider == ProviderType.OPENAI:
                return await self._embed_with_openai(non_empty)
            elif provider == ProviderType.AZURE_OPENAI:
                return await self._embed_with_azure_openai(non_empty)
            elif provider == ProviderType.AZURE_FOUNDRY:
                return await self._embed_with_azure_foundry(non_empty)
            elif provider == ProviderType.VERTEX_AI:
                return await self._embed_with_vertex(non_empty)
            elif provider == ProviderType.ANTHROPIC:
                return await self._embed_with_anthropic(non_empty)
            else:
                logger.warning(f"Embeddings not supported for {provider.value}")
                return None

        except Exception as e:
            logger.error(f"Error embedding with {provider.value}: {e}")
            self.record_provider_health(provider, False)
            if self.fallback_provider and self.fallback_provider != provider:
                logger.info(f"Retrying with fallback provider {self.fallback_provider.value}")
                return await self.embed_texts(texts)
            return None

    async def _embed_with_openai(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Call OpenAI embedding API.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings (1536-dim)
        """
        config = self.get_provider_config(ProviderType.OPENAI)
        if not config or not config.api_key:
            logger.error("OpenAI API key not configured")
            return None

        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "input": texts,
            "model": self.model_id or "text-embedding-3-large",
        }

        try:
            logger.debug(f"OpenAI: embedding {len(texts)} texts with model {self.model_id}")
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"OpenAI API error {response.status}: {error_text}")
                        return None

                    data = await response.json()
                    embeddings = [item["embedding"] for item in data.get("data", [])]

                    # Validate dimensions
                    for emb in embeddings:
                        if len(emb) != self.embedding_dimension:
                            logger.warning(f"OpenAI returned {len(emb)}-dim embedding, expected {self.embedding_dimension}")
                            return None

                    return embeddings

        except aiohttp.ClientError as e:
            logger.error(f"OpenAI HTTP error: {e}")
            return None
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            return None

    async def _embed_with_azure_openai(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Call Azure OpenAI embedding API.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings (1536-dim)
        """
        config = self.get_provider_config(ProviderType.AZURE_OPENAI)
        if not config or not config.api_key or not config.endpoint:
            logger.error("Azure OpenAI credentials not configured")
            return None

        # Extract deployment name from config or use default
        deployment_name = getattr(config, "deployment_name", "text-embedding-3-large")

        # Construct endpoint: https://{resource}.openai.azure.com/openai/deployments/{deployment-id}/embeddings
        url = f"{config.endpoint}/openai/deployments/{deployment_name}/embeddings?api-version=2024-02-15-preview"

        headers = {
            "api-key": config.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "input": texts,
            "model": self.model_id or "text-embedding-3-large",
        }

        try:
            logger.debug(f"Azure OpenAI: embedding {len(texts)} texts with deployment {deployment_name}")
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Azure OpenAI API error {response.status}: {error_text}")
                        return None

                    data = await response.json()
                    embeddings = [item["embedding"] for item in data.get("data", [])]

                    # Validate dimensions
                    for emb in embeddings:
                        if len(emb) != self.embedding_dimension:
                            logger.warning(f"Azure OpenAI returned {len(emb)}-dim embedding, expected {self.embedding_dimension}")
                            return None

                    return embeddings

        except aiohttp.ClientError as e:
            logger.error(f"Azure OpenAI HTTP error: {e}")
            return None
        except Exception as e:
            logger.error(f"Azure OpenAI embedding failed: {e}")
            return None

    async def _embed_with_azure_foundry(
        self, texts: List[str]
    ) -> Optional[List[List[float]]]:
        """Call Azure Foundry embedding endpoint.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings (1536-dim)
        """
        config = self.get_provider_config(ProviderType.AZURE_FOUNDRY)
        if not config or not config.api_key or not config.endpoint:
            logger.error("Azure Foundry credentials not configured")
            return None

        # Construct serverless endpoint: https://{project}.{region}.models.ai.azure.com/serverless/v1/embeddings/completions
        url = f"{config.endpoint}/serverless/v1/embeddings/completions"

        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "input": texts,
            "model": self.model_id or "text-embedding-3-large",
        }

        try:
            logger.debug(f"Azure Foundry: embedding {len(texts)} texts")
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Azure Foundry API error {response.status}: {error_text}")
                        return None

                    data = await response.json()
                    embeddings = [item["embedding"] for item in data.get("data", [])]

                    # Validate dimensions
                    for emb in embeddings:
                        if len(emb) != self.embedding_dimension:
                            logger.warning(f"Azure Foundry returned {len(emb)}-dim embedding, expected {self.embedding_dimension}")
                            return None

                    return embeddings

        except aiohttp.ClientError as e:
            logger.error(f"Azure Foundry HTTP error: {e}")
            return None
        except Exception as e:
            logger.error(f"Azure Foundry embedding failed: {e}")
            return None

    async def _embed_with_vertex(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Call Google Vertex AI embedding API.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings (1536-dim, resized if needed)
        """
        config = self.get_provider_config(ProviderType.VERTEX_AI)
        if not config or not config.api_key or not config.endpoint:
            logger.error("Vertex AI credentials not configured")
            return None

        # Extract project and region from endpoint or config
        # Expected format: https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/textembedding-gecko@latest:predict
        project_id = getattr(config, "project_id", None)
        region = getattr(config, "region", "us-central1")

        if not project_id:
            logger.error("Vertex AI project_id not configured")
            return None

        url = f"https://{region}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{region}/publishers/google/models/textembedding-gecko@latest:predict"

        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "instances": [{"content": text} for text in texts],
        }

        try:
            logger.debug(f"Vertex AI: embedding {len(texts)} texts")
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Vertex AI API error {response.status}: {error_text}")
                        return None

                    data = await response.json()
                    embeddings = [pred["embeddings"]["values"] for pred in data.get("predictions", [])]

                    # Validate and resize dimensions if needed
                    # Vertex AI embeddings are typically 768-dim, resize to 1536
                    resized_embeddings = []
                    for emb in embeddings:
                        if len(emb) != self.embedding_dimension:
                            # Simple resize: duplicate values or truncate
                            if len(emb) < self.embedding_dimension:
                                # Pad by repeating values
                                repeat_factor = self.embedding_dimension // len(emb) + 1
                                emb = (emb * repeat_factor)[:self.embedding_dimension]
                            else:
                                # Truncate
                                emb = emb[:self.embedding_dimension]
                            logger.debug(f"Vertex AI embedding resized to {self.embedding_dimension} dimensions")
                        resized_embeddings.append(emb)

                    return resized_embeddings

        except aiohttp.ClientError as e:
            logger.error(f"Vertex AI HTTP error: {e}")
            return None
        except Exception as e:
            logger.error(f"Vertex AI embedding failed: {e}")
            return None

    async def _embed_with_anthropic(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Use Anthropic provider (fallback to OpenAI).

        Note: Anthropic's Claude doesn't have a native embeddings API.
        This method falls back to OpenAI embeddings as recommended by Anthropic.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings (1536-dim)
        """
        logger.debug(f"Anthropic: no native embeddings API, falling back to OpenAI")

        # Check if OpenAI is available
        openai_config = self.get_provider_config(ProviderType.OPENAI)
        if openai_config and openai_config.api_key:
            logger.info("Using OpenAI embeddings as fallback for Anthropic request")
            return await self._embed_with_openai(texts)

        # If OpenAI not available, try Azure OpenAI
        azure_config = self.get_provider_config(ProviderType.AZURE_OPENAI)
        if azure_config and azure_config.api_key:
            logger.info("Using Azure OpenAI embeddings as fallback for Anthropic request")
            return await self._embed_with_azure_openai(texts)

        logger.error("Anthropic embeddings requested but no embedding fallback provider configured")
        return None

    async def execute(self, texts: List[str]) -> Any:
        """Execute embedding (implements BaseProviderAdapter interface).

        Args:
            texts: Texts to embed

        Returns:
            List of embeddings
        """
        return await self.embed_texts(texts)

    def get_supported_models(self, provider: ProviderType) -> Optional[List[str]]:
        """Get embedding models supported by provider.

        Args:
            provider: Provider type

        Returns:
            List of model IDs or None
        """
        models = self.registry.list_models(provider)
        embedding_models = [
            m.provider_model_id
            for m in models
            if ModelCapability.EMBEDDINGS in m.capabilities
        ]
        return embedding_models if embedding_models else None

    def estimate_embedding_tokens(self, text: str) -> int:
        """Estimate tokens in text before embedding.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        # Rough heuristic: 1 token ≈ 4 characters
        return max(1, len(text) // 4)

    def validate_embedding_dimension(self, embedding: List[float]) -> bool:
        """Validate embedding has correct dimension.

        Args:
            embedding: Embedding vector

        Returns:
            True if dimension matches target
        """
        if not embedding:
            return False
        return len(embedding) == self.embedding_dimension
