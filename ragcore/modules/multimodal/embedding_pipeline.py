"""Multi-modal unified embedding pipeline - all modalities → 1536-dim vectors."""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID
import time

from ragcore.modules.multimodal.models import (
    MultiModalChunk,
    ModuleType,
    ProcessingResult,
)
from ragcore.modules.multimodal.providers.embedding_adapter import EmbeddingProviderAdapter

logger = logging.getLogger(__name__)


class MultiModalEmbeddingPipeline:
    """Generate unified 1536-dimensional embeddings for all modalities.

    Converts text-based chunks from any modality (image OCR, audio transcription,
    video narration, etc.) into a shared embedding space for cross-modal search.
    """

    def __init__(
        self,
        embedding_client=None,
        embedding_model: str = "text-embedding-3-large",
        embedding_dimension: int = 1536,
        batch_size: int = 10,
        cache_enabled: bool = True,
        embedding_adapter: Optional[EmbeddingProviderAdapter] = None,
    ):
        """Initialize embedding pipeline.

        Args:
            embedding_client: Deprecated - use embedding_adapter instead
            embedding_model: Model name (default: text-embedding-3-large)
            embedding_dimension: Output dimension (default: 1536)
            batch_size: Batch processing size
            cache_enabled: Cache embeddings for identical text
            embedding_adapter: EmbeddingProviderAdapter for provider selection (optional)
        """
        self.embedding_client = embedding_client
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.batch_size = batch_size
        self.cache_enabled = cache_enabled
        self.embedding_cache: Dict[str, List[float]] = {}

        # Use provided adapter or create default
        if embedding_adapter:
            self.adapter = embedding_adapter
        else:
            self.adapter = EmbeddingProviderAdapter(
                embedding_dimension=embedding_dimension,
                model_id=embedding_model,
            )

        logger.info(
            f"Embedding pipeline initialized: provider={self.adapter.primary_provider.value if self.adapter.primary_provider else 'auto'}, "
            f"dimension={embedding_dimension}, batch_size={batch_size}"
        )

    async def embed_chunk(self, chunk: MultiModalChunk) -> MultiModalChunk:
        """Generate embedding for single chunk.

        Args:
            chunk: Chunk to embed (content field used)

        Returns:
            Chunk with embedding populated
        """
        try:
            if not chunk.content:
                logger.warning(f"Empty chunk content for chunk {chunk.id}, skipping embedding")
                return chunk

            # Check cache first
            if self.cache_enabled:
                cache_key = f"{chunk.modality.value}:{chunk.content[:100]}"
                if cache_key in self.embedding_cache:
                    chunk.embedding = self.embedding_cache[cache_key]
                    return chunk

            # Generate embedding
            embedding = await self._get_embedding(chunk.content)

            if embedding:
                chunk.embedding = embedding

                # Cache result
                if self.cache_enabled:
                    cache_key = f"{chunk.modality.value}:{chunk.content[:100]}"
                    self.embedding_cache[cache_key] = embedding

            return chunk

        except Exception as e:
            logger.error(f"Error embedding chunk {chunk.id}: {e}")
            # Don't fail - chunk proceeds without embedding
            return chunk

    async def embed_chunks_batch(
        self,
        chunks: List[MultiModalChunk],
        show_progress: bool = False,
    ) -> List[MultiModalChunk]:
        """Generate embeddings for multiple chunks in batches.

        Args:
            chunks: List of chunks to embed
            show_progress: Log progress

        Returns:
            Chunks with embeddings populated
        """
        start_time = time.time()
        embedded_chunks = []
        total = len(chunks)

        for i in range(0, total, self.batch_size):
            batch = chunks[i : i + self.batch_size]

            if show_progress:
                logger.info(f"Embedding batch {i // self.batch_size + 1} ({i}-{min(i + self.batch_size, total)}/{total})")

            # Process batch
            batch_texts = [chunk.content for chunk in batch if chunk.content]

            if batch_texts:
                embeddings = await self._get_embeddings_batch(batch_texts)

                # Assign embeddings to chunks
                embedding_idx = 0
                for chunk in batch:
                    if chunk.content and embedding_idx < len(embeddings):
                        chunk.embedding = embeddings[embedding_idx]
                        embedding_idx += 1

            embedded_chunks.extend(batch)

        elapsed = time.time() - start_time
        elapsed = max(elapsed, 0.01)  # Avoid division by zero for very fast batches
        logger.info(
            f"Embedded {total} chunks in {elapsed:.2f}s "
            f"({total / elapsed:.1f} chunks/sec, "
            f"{len(embedded_chunks)} succeeded)"
        )

        return embedded_chunks

    async def embed_processing_result(
        self,
        result: ProcessingResult,
    ) -> ProcessingResult:
        """Embed all chunks in a processing result.

        Args:
            result: Processing result containing chunks

        Returns:
            Result with all chunks embedded
        """
        if not result.chunks:
            return result

        result.chunks = await self.embed_chunks_batch(result.chunks)
        return result

    async def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for single text using provider adapter.

        Args:
            text: Text to embed

        Returns:
            1536-dimensional embedding vector or None
        """
        try:
            if not text or not text.strip():
                return None

            # Use provider adapter for actual embedding call
            embedding = await self.adapter.embed_text(text)

            if embedding and self.adapter.validate_embedding_dimension(embedding):
                return embedding

            if embedding:
                logger.warning(
                    f"Embedding dimension mismatch: got {len(embedding)}, expected {self.embedding_dimension}"
                )

            return embedding

        except Exception as e:
            logger.error(f"Error calling embedding API: {e}")
            return None

    async def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for batch of texts using provider adapter.

        Args:
            texts: List of texts to embed

        Returns:
            List of 1536-dimensional embeddings
        """
        if not texts:
            return []

        try:
            # Use provider adapter for batch embedding
            embeddings = await self.adapter.embed_texts(texts)

            if embeddings:
                # Validate dimensions
                valid_count = 0
                for embedding in embeddings:
                    if self.adapter.validate_embedding_dimension(embedding):
                        valid_count += 1

                if valid_count < len(embeddings):
                    logger.warning(
                        f"Dimension validation: {valid_count}/{len(embeddings)} embeddings valid"
                    )

            return embeddings or []

        except Exception as e:
            logger.error(f"Error in batch embedding: {e}")
            return []

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics.

        Returns:
            Dict with cache hit count, size, etc.
        """
        return {
            "cache_enabled": self.cache_enabled,
            "cache_size": len(self.embedding_cache),
            "memory_mb": len(str(self.embedding_cache)) / (1024 * 1024),
        }

    def clear_cache(self):
        """Clear embedding cache."""
        self.embedding_cache.clear()
        logger.info("Embedding cache cleared")

    def validate_embedding_dimension(self, embedding: List[float]) -> bool:
        """Validate that embedding has correct dimension.

        Args:
            embedding: Embedding vector to validate

        Returns:
            True if dimension matches
        """
        if not embedding:
            return False
        return len(embedding) == self.embedding_dimension
