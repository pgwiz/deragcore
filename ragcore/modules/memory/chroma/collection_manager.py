"""ChromaDB collection manager - Collection lifecycle, document management, semantic search."""

import logging
from typing import List, Optional, Tuple, Dict, Any
from uuid import UUID
from datetime import datetime

logger = logging.getLogger(__name__)


class ChromaCollectionManager:
    """Manages ChromaDB collections for memory storage."""

    def __init__(self, client_manager):
        """Initialize collection manager.

        Args:
            client_manager: ChromaClientManager instance
        """
        self.client_manager = client_manager
        self.collections_cache = {}  # {collection_key: collection_obj}

    def _make_collection_name(
        self,
        session_id: UUID,
        memory_type: str = "all",
    ) -> str:
        """Generate collection name from session and type.

        Args:
            session_id: Session UUID
            memory_type: Memory type (finding, decision, all, etc)

        Returns:
            Collection name string
        """
        deployment_mode = self.client_manager.config.deployment_mode
        return f"chroma_{deployment_mode}_{str(session_id)[:8]}_{memory_type}"

    async def get_or_create_collection(
        self,
        session_id: UUID,
        memory_type: str = "all",
    ):
        """Get or create collection for session.

        Args:
            session_id: Session UUID
            memory_type: Memory type filter

        Returns:
            ChromaDB collection object or None if unavailable
        """
        client = await self.client_manager.get_client()
        if client is None:
            return None

        collection_name = self._make_collection_name(session_id, memory_type)

        # Check cache
        if collection_name in self.collections_cache:
            return self.collections_cache[collection_name]

        try:
            # Get or create with metadata schema
            collection = client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "session_id": str(session_id),
                    "memory_type": memory_type,
                    "created_at": datetime.utcnow().isoformat(),
                    "description": f"Collection for session memories (type={memory_type})",
                },
            )

            self.collections_cache[collection_name] = collection
            logger.debug(
                f"Collection created/retrieved: {collection_name} "
                f"(count={collection.count()})"
            )

            return collection

        except Exception as e:
            logger.error(f"Failed to get/create collection {collection_name}: {e}")
            return None

    async def add_memories(
        self,
        session_id: UUID,
        memories: List[Dict[str, Any]],
        memory_type: str = "all",
    ) -> dict:
        """Batch add memories to collection.

        Args:
            session_id: Session UUID
            memories: List of memory dicts with id, embedding, metadata
                     Format: {"id": str, "embedding": [float...], "metadata": {...}, "document": str}
            memory_type: Memory type for collection selection

        Returns:
            Dict with {success_count, failed_count, failed_ids, errors}
        """
        collection = await self.get_or_create_collection(session_id, memory_type)
        if collection is None:
            return {
                "success_count": 0,
                "failed_count": len(memories),
                "failed_ids": [m["id"] for m in memories],
                "errors": ["ChromaDB unavailable"],
            }

        success = 0
        failed = 0
        failed_ids = []
        errors = []

        try:
            # Prepare data for ChromaDB add
            ids = [m["id"] for m in memories]
            embeddings = [m.get("embedding", []) for m in memories]
            documents = [m.get("document", "") for m in memories]
            metadatas = [m.get("metadata", {}) for m in memories]

            # Add to collection
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

            success = len(memories)
            logger.info(
                f"Added {success} memories to collection {self._make_collection_name(session_id, memory_type)}"
            )

        except Exception as e:
            logger.error(f"Error adding memories to ChromaDB: {e}")
            failed = len(memories)
            failed_ids = [m["id"] for m in memories]
            errors = [str(e)]

        return {
            "success_count": success,
            "failed_count": failed,
            "failed_ids": failed_ids,
            "errors": errors,
        }

    async def delete_memories(
        self,
        session_id: UUID,
        memory_ids: List[str],
        memory_type: str = "all",
    ) -> dict:
        """Delete memories from collection.

        Args:
            session_id: Session UUID
            memory_ids: List of memory IDs to delete
            memory_type: Memory type for collection selection

        Returns:
            Dict with {deleted_count, failed_count, errors}
        """
        collection = await self.get_or_create_collection(session_id, memory_type)
        if collection is None:
            return {
                "deleted_count": 0,
                "failed_count": len(memory_ids),
                "errors": ["ChromaDB unavailable"],
            }

        try:
            collection.delete(ids=memory_ids)
            logger.info(f"Deleted {len(memory_ids)} memories from ChromaDB")

            return {
                "deleted_count": len(memory_ids),
                "failed_count": 0,
                "errors": [],
            }

        except Exception as e:
            logger.error(f"Error deleting memories: {e}")
            return {
                "deleted_count": 0,
                "failed_count": len(memory_ids),
                "errors": [str(e)],
            }

    async def semantic_search(
        self,
        session_id: UUID,
        query_embedding: List[float],
        limit: int = 5,
        memory_type: str = "all",
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search collection by semantic similarity.

        Args:
            session_id: Session UUID
            query_embedding: Query embedding vector (1536 dims for Claude)
            limit: Max results to return
            memory_type: Memory type for collection selection
            where: Optional metadata filter dict

        Returns:
            List of (memory_id, distance, metadata) tuples sorted by relevance
        """
        collection = await self.get_or_create_collection(session_id, memory_type)
        if collection is None:
            logger.warning("ChromaDB unavailable for semantic search")
            return []

        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where,
            )

            # Format results as (id, distance, metadata) tuples
            formatted = []
            if results and results.get("ids") and results["ids"][0]:
                for i, memory_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i] if results.get("distances") else 0.0
                    metadata = (
                        results["metadatas"][0][i]
                        if results.get("metadatas")
                        else {}
                    )

                    # ChromaDB returns distance (lower is more similar)
                    # Convert to similarity score (0-1, higher is better)
                    similarity = 1.0 / (1.0 + distance)

                    formatted.append((memory_id, similarity, metadata))

            logger.debug(
                f"Semantic search found {len(formatted)} results "
                f"(query type: {memory_type})"
            )

            return formatted

        except Exception as e:
            logger.error(f"Error during semantic search: {e}")
            return []

    async def cleanup_collection(
        self,
        session_id: UUID,
        memory_type: str = "all",
    ) -> dict:
        """Clean up collection by removing inactive memories and consolidating.

        Args:
            session_id: Session UUID
            memory_type: Memory type for collection selection

        Returns:
            Dict with {deleted_count, consolidated, message}
        """
        collection = await self.get_or_create_collection(session_id, memory_type)
        if collection is None:
            return {
                "deleted_count": 0,
                "consolidated": False,
                "message": "ChromaDB unavailable",
            }

        try:
            # Get collection stats
            initial_count = collection.count()

            # Get all documents with is_active=False metadata
            # Note: This is a simplified approach; in production would use where filter
            results = collection.get()

            inactive_ids = [
                doc_id
                for doc_id, metadata in zip(
                    results.get("ids", []), results.get("metadatas", [])
                )
                if not metadata.get("is_active", True)
            ]

            deleted = 0
            if inactive_ids:
                await self.delete_memories(session_id, inactive_ids, memory_type)
                deleted = len(inactive_ids)

            final_count = collection.count()

            logger.info(
                f"Collection cleanup: initial={initial_count}, "
                f"deleted={deleted}, final={final_count}"
            )

            return {
                "deleted_count": deleted,
                "consolidated": True,
                "message": f"Removed {deleted} inactive memories",
            }

        except Exception as e:
            logger.error(f"Error cleaning up collection: {e}")
            return {
                "deleted_count": 0,
                "consolidated": False,
                "message": f"Cleanup error: {str(e)}",
            }

    async def delete_collection(
        self,
        session_id: UUID,
        memory_type: str = "all",
    ) -> bool:
        """Delete entire collection (session cleanup).

        Args:
            session_id: Session UUID
            memory_type: Memory type for collection selection

        Returns:
            True if successful
        """
        client = await self.client_manager.get_client()
        if client is None:
            return False

        collection_name = self._make_collection_name(session_id, memory_type)

        try:
            client.delete_collection(name=collection_name)

            # Remove from cache
            if collection_name in self.collections_cache:
                del self.collections_cache[collection_name]

            logger.info(f"Deleted collection: {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {e}")
            return False
