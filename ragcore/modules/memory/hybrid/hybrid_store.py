"""Hybrid memory store - PostgreSQL + ChromaDB with intelligent fallback."""

import logging
from typing import Optional, List, Dict, Any, Tuple
from uuid import UUID
from datetime import datetime
import asyncio

from ragcore.modules.memory.long_term import LongTermMemoryStore
from ragcore.modules.memory.models import LongTermMemory
from ragcore.core.ai_controller import AIController
from ragcore.config import settings

logger = logging.getLogger(__name__)


class HybridMemoryStore(LongTermMemoryStore):
    """Extended LongTermMemoryStore with ChromaDB hybrid support.

    Maintains PostgreSQL as source of truth with optional ChromaDB cache layer.
    Supports three deployment modes: postgres_only, hybrid, chroma_primary.
    """

    def __init__(
        self,
        chroma_client_manager=None,
        collection_manager=None,
        sync_manager=None,
        smart_router=None,
        deployment_mode: str = "hybrid",
        embedding_provider: str = "azure",
        embedding_model: str = "phi-4",
    ):
        """Initialize hybrid memory store.

        Args:
            chroma_client_manager: ChromaDB client manager (optional)
            collection_manager: ChromaDB collection manager (optional)
            sync_manager: ChromaDB sync manager (optional)
            smart_router: Smart search router (optional)
            deployment_mode: postgres_only, hybrid, or chroma_primary
            embedding_provider: Embedding provider (azure, openai, ollama)
            embedding_model: Embedding model name
        """
        super().__init__()

        self.chroma_enabled = deployment_mode != "postgres_only"
        self.chroma_primary = deployment_mode == "chroma_primary"
        self.deployment_mode = deployment_mode

        self.chroma_client_manager = chroma_client_manager
        self.collection_manager = collection_manager
        self.sync_manager = sync_manager
        self.smart_router = smart_router

        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.embedding_dimension = 1536  # Standard embedding dimension for OpenAI embeddings

        logger.info(
            f"HybridMemoryStore initialized: "
            f"mode={deployment_mode}, "
            f"chroma_enabled={self.chroma_enabled}"
        )

    async def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text using EmbeddingProviderAdapter.

        Args:
            text: Text to embed

        Returns:
            1536-dimensional embedding vector or None
        """
        try:
            if not text or not text.strip():
                return None

            # Use EmbeddingProviderAdapter for real embedding generation
            from ragcore.modules.multimodal.providers.embedding_adapter import EmbeddingProviderAdapter
            from ragcore.core.model_provider_registry import ModelProviderRegistry

            # Get or create registry and adapter
            registry = ModelProviderRegistry()
            adapter = EmbeddingProviderAdapter(
                registry=registry,
                embedding_dimension=self.embedding_dimension,
                model_id=self.embedding_model,
            )

            # Generate embedding
            embedding = await adapter.embed_text(text)

            # Validate dimension
            if embedding and len(embedding) == self.embedding_dimension:
                logger.debug(f"Generated {self.embedding_dimension}-dim embedding for text: {text[:50]}...")
                return embedding

            if embedding:
                logger.warning(
                    f"Embedding dimension mismatch: got {len(embedding) if embedding else 0}, "
                    f"expected {self.embedding_dimension}"
                )

            return None

        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None

    async def save_memory(
        self,
        session_id: UUID,
        memory_type: str,
        content: str,
        summary: Optional[str] = None,
        source: Optional[str] = None,
        tags: Optional[List[str]] = None,
        importance_score: float = 0.5,
        ttl_seconds: Optional[int] = None,
        context_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[UUID] = None,
    ) -> Optional[UUID]:
        """Save memory to PostgreSQL and queue ChromaDB sync.

        Args:
            (same as parent LongTermMemoryStore.save_memory)

        Returns:
            Memory ID if successful
        """
        # Save to PostgreSQL (always)
        memory_id = await super().save_memory(
            session_id=session_id,
            memory_type=memory_type,
            content=content,
            summary=summary,
            source=source,
            tags=tags,
            importance_score=importance_score,
            ttl_seconds=ttl_seconds,
            context_data=context_data,
            user_id=user_id,
        )

        if not memory_id:
            return None

        # Queue ChromaDB sync if enabled (non-blocking)
        if self.chroma_enabled and self.sync_manager:
            try:
                # Get embedding
                embedding = await self._get_embedding(content)

                # Queue sync (fire-and-forget)
                asyncio.create_task(
                    self.sync_manager.sync_single_memory(
                        session_id=session_id,
                        memory_id=memory_id,
                        embedding=embedding or [],
                        memory_type=memory_type,
                        document=content,
                        metadata={
                            "summary": summary or "",
                            "source": source or "",
                            "tags": tags or [],
                            "importance_score": importance_score,
                            "memory_type": memory_type,
                            "is_active": True,
                        },
                        operation="insert",
                    )
                )

                logger.debug(
                    f"Queued ChromaDB sync for memory {memory_id} (non-blocking)"
                )

            except Exception as e:
                logger.warning(f"Failed to queue ChromaDB sync: {e}")
                # Don't fail the save if sync fails

        return memory_id

    async def search_semantic(
        self,
        session_id: UUID,
        query: str,
        limit: int = 5,
        memory_type: Optional[str] = None,
        min_similarity: float = 0.5,
    ) -> List[Tuple[LongTermMemory, float]]:
        """Search memories by semantic similarity.

        Uses smart router to choose between ChromaDB and pgvector.

        Args:
            session_id: Session to search in
            query: Query text
            limit: Max results to return
            memory_type: Optional filter by memory type
            min_similarity: Minimum similarity threshold (0.0-1.0)

        Returns:
            List of (memory, similarity_score) tuples sorted by relevance
        """
        if not self.smart_router:
            logger.warning("Smart router not configured, skipping semantic search")
            return []

        try:
            # Get embedding for query
            query_embedding = await self._get_embedding(query)
            if not query_embedding:
                logger.warning("Could not get embedding for query")
                return []

            # Route search via smart router
            result = await self.smart_router.route_search(
                session_id=session_id,
                query_embedding=query_embedding,
                limit=limit,
                memory_type=memory_type or "all",
            )

            if not result.get("results"):
                logger.debug(
                    f"No search results from {result.get('backend_used', 'unknown')}"
                )
                return []

            # Convert results to (memory, similarity) tuples
            memories = []
            for memory_id, similarity, metadata in result["results"]:
                if similarity >= min_similarity:
                    # Fetch full memory from PostgreSQL
                    try:
                        memory = await self.get_memory(UUID(memory_id))
                        if memory:
                            memories.append((memory, similarity))
                    except Exception as e:
                        logger.warning(
                            f"Could not fetch memory {memory_id}: {e}"
                        )

            logger.info(
                f"Semantic search: {len(memories)} results from {result.get('backend_used')} "
                f"(latency={result.get('latency_ms', 0)}ms)"
            )

            return memories

        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []

    async def get_search_performance_stats(
        self,
        window_minutes: int = 60,
    ) -> dict:
        """Get search performance metrics from smart router.

        Args:
            window_minutes: Time window for stats (unused, router maintains full history)

        Returns:
            Dict with performance comparisons
        """
        if not self.smart_router:
            return {
                "status": "no_router",
                "message": "Smart router not configured",
            }

        return self.smart_router.get_performance_stats()

    async def sync_all_memories_to_chroma(
        self,
        session_id: UUID,
    ) -> dict:
        """Force full sync of all memories for session to ChromaDB.

        Args:
            session_id: Session to resync

        Returns:
            Dict with sync results {synced_count, failed_count, inconsistencies}
        """
        if not self.sync_manager:
            return {
                "synced_count": 0,
                "failed_count": 0,
                "message": "Sync manager not configured",
            }

        try:
            # Get all active memories from PostgreSQL
            # (Using parent class method would require modification)
            logger.info(f"Starting full resync for session {session_id}")

            # In production, would fetch from database
            # For now, return placeholder
            return {
                "synced_count": 0,
                "failed_count": 0,
                "message": "Full resync requires database query implementation",
            }

        except Exception as e:
            logger.error(f"Error in full sync: {e}")
            return {
                "synced_count": 0,
                "failed_count": 0,
                "errors": [str(e)],
            }

    async def get_chroma_sync_status(
        self,
        session_id: UUID,
    ) -> dict:
        """Get ChromaDB sync status for session.

        Args:
            session_id: Session UUID

        Returns:
            Dict with sync status information
        """
        if not self.sync_manager:
            return {
                "status": "disabled",
                "message": "ChromaDB not configured",
            }

        queue_status = self.sync_manager.get_queue_status()

        return {
            "deployment_mode": self.deployment_mode,
            "chroma_enabled": self.chroma_enabled,
            "chroma_available": (
                self.chroma_client_manager.is_healthy()
                if self.chroma_client_manager
                else False
            ),
            "queue_status": queue_status,
            "message": f"Queue: {queue_status.get('queue_size')} items, "
            f"{queue_status.get('overdue_count')} overdue",
        }

    async def get_chroma_health(self) -> dict:
        """Get ChromaDB health status.

        Returns:
            Dict with health information
        """
        if not self.chroma_client_manager:
            return {
                "status": "disabled",
                "message": "ChromaDB manager not configured",
            }

        if not self.chroma_enabled:
            return {
                "status": "disabled",
                "message": "ChromaDB disabled in deployment mode",
            }

        health = await self.chroma_client_manager.health_check()
        return health
