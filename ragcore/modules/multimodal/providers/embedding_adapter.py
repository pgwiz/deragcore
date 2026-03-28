"""Embedding provider adapter - abstracts embedding generation across providers."""

from typing import Optional, List, Any
import logging

from ragcore.core.model_provider_registry import ProviderType, ModelCapability
from ragcore.modules.multimodal.providers.base_adapter import BaseProviderAdapter

logger = logging.getLogger(__name__)


class EmbeddingProviderAdapter(BaseProviderAdapter):
    """Adapter for embedding/vectorization capabilities.

    Supports:
    - OpenAI text-embedding-3-large (primary) - 1536 dimensions
    - Anthropic embeddings (via Claude with extraction)
    - Azure OpenAI embeddings
    - Google Vertex AI PaLM embeddings
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
        # TODO: Implement OpenAI Embeddings API call
        # 1. Get OpenAI provider config
        # 2. Batch texts if needed (max 2048 tokens per request)
        # 3. Call /v1/embeddings with model="text-embedding-3-large"
        # 4. Extract embedding vectors from response
        # 5. Validate dimension (should be 1536)

        logger.debug(f"OpenAI: embedding {len(texts)} texts")

        # Placeholder: return deterministic embeddings for testing
        embeddings = []
        for text in texts:
            seed_value = len(text) % 100
            embedding = [0.1 * (seed_value % 10) / 10.0 + 0.001 * i for i in range(self.embedding_dimension)]
            embeddings.append(embedding)
        return embeddings

    async def _embed_with_azure_openai(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Call Azure OpenAI embedding API.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings (1536-dim)
        """
        # TODO: Implement Azure OpenAI Embeddings API call
        # 1. Get Azure OpenAI provider config
        # 2. Call /openai/deployments/{deployment}/embeddings
        # 3. Extract embeddings from response

        logger.debug(f"Azure OpenAI: embedding {len(texts)} texts")
        embeddings = []
        for text in texts:
            embedding = [0.2 + 0.001 * i for i in range(self.embedding_dimension)]
            embeddings.append(embedding)
        return embeddings

    async def _embed_with_azure_foundry(
        self, texts: List[str]
    ) -> Optional[List[List[float]]]:
        """Call Azure Foundry embedding endpoint.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings (1536-dim)
        """
        # TODO: Implement Azure Foundry Embeddings API call
        # 1. Get Azure Foundry provider config
        # 2. Call serverless endpoint: /serverless/v1/embeddings/completions
        # 3. Extract embeddings from response

        logger.debug(f"Azure Foundry: embedding {len(texts)} texts")
        embeddings = []
        for text in texts:
            embedding = [0.3 + 0.001 * i for i in range(self.embedding_dimension)]
            embeddings.append(embedding)
        return embeddings

    async def _embed_with_vertex(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Call Google Vertex AI embedding API.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings (1536-dim, or provider's native dimension)
        """
        # TODO: Implement Vertex AI Embeddings API call
        # 1. Get Vertex AI provider config
        # 2. Call /v1/projects/{project}/locations/{region}/publishers/google/models/embedding-001:predict
        # 3. Extract embeddings, potentially resize to 1536-dim if needed

        logger.debug(f"Vertex AI: embedding {len(texts)} texts")
        embeddings = []
        for text in texts:
            embedding = [0.4 + 0.001 * i for i in range(self.embedding_dimension)]
            embeddings.append(embedding)
        return embeddings

    async def _embed_with_anthropic(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Use Claude for embeddings (non-standard, extraction-based).

        Note: Anthropic's Claude doesn't have native embeddings API.
        This extracts semantic information from Claude's responses.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings (1536-dim approximation)
        """
        # TODO: Implement Anthropic embedding via Claude analysis
        # Note: This is non-standard; Anthropic recommends using OpenAI embeddings
        # Placeholder for now

        logger.debug(f"Anthropic (via Claude): embedding {len(texts)} texts (non-native)")
        embeddings = []
        for text in texts:
            embedding = [0.5 + 0.001 * i for i in range(self.embedding_dimension)]
            embeddings.append(embedding)
        return embeddings

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
