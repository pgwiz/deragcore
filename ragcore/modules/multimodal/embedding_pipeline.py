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
    ):
        """Initialize embedding pipeline.

        Args:
            embedding_client: OpenAI/Azure embedding client
            embedding_model: Model name (default: text-embedding-3-large)
            embedding_dimension: Output dimension (default: 1536)
            batch_size: Batch processing size
            cache_enabled: Cache embeddings for identical text
        """
        self.embedding_client = embedding_client
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self.batch_size = batch_size
        self.cache_enabled = cache_enabled
        self.embedding_cache: Dict[str, List[float]] = {}

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
        """Get embedding for single text.

        Placeholder for actual embedding API call.

        Args:
            text: Text to embed

        Returns:
            1536-dimensional embedding vector or None
        """
        try:
            # Placeholder - in production would call OpenAI/Azure embedding API
            # with model="text-embedding-3-large"
            # Returns 1536-dim vector

            # For now, return placeholder
            if not text:
                return None

            # Deterministic placeholder based on text length
            # (for testability)
            seed_value = len(text) % 100
            embedding = [0.1 * (seed_value % 10) / 10.0 + 0.001 * i for i in range(self.embedding_dimension)]
            return embedding

        except Exception as e:
            logger.error(f"Error calling embedding API: {e}")
            return None

    async def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for batch of texts.

        Placeholder for actual batch embedding API call.

        Args:
            texts: List of texts to embed

        Returns:
            List of 1536-dimensional embeddings
        """
        embeddings = []
        for text in texts:
            embedding = await self._get_embedding(text)
            if embedding:
                embeddings.append(embedding)

        return embeddings

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
