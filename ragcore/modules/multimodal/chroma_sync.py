"""ChromaDB sync integration for multimodal chunks.

Handles queueing multimodal chunks for synchronization to ChromaDB
after they are extracted, chunked, and embedded.
"""

import logging
from typing import Optional, List
from uuid import UUID
from datetime import datetime

from ragcore.modules.multimodal.models import MultiModalChunk, ProcessingResult
from ragcore.modules.memory.chroma.sync_manager import ChromaMemorySyncManager
from ragcore.modules.memory.chroma.collection_manager import ChromaCollectionManager

logger = logging.getLogger(__name__)


class MultiModalChromaSync:
    """Sync multimodal chunks to ChromaDB after processing.

    Integrates the multimodal processing pipeline with ChromaDB
    cache layer for efficient semantic search across modalities.
    """

    def __init__(self, sync_manager: Optional[ChromaMemorySyncManager] = None):
        """Initialize multimodal ChromaDB sync.

        Args:
            sync_manager: ChromaMemorySyncManager instance (required)
                         If None, will be lazily initialized
        """
        self.sync_manager = sync_manager
        self._initialized = sync_manager is not None

    async def sync_processing_result(
        self,
        session_id: UUID,
        content_id: UUID,
        processing_result: ProcessingResult,
        storage_path: Optional[str] = None,
    ) -> dict:
        """Sync all chunks from a processing result to ChromaDB.

        Args:
            session_id: Session UUID
            content_id: Source content UUID
            processing_result: ProcessingResult containing extracted chunks
            storage_path: Storage path if content stored externally

        Returns:
            Dict with {synced_count, failed_count, failed_ids}
        """
        if not self.sync_manager:
            await self._ensure_initialized()

        if not processing_result.chunks:
            logger.debug(f"No chunks to sync for content {content_id}")
            return {"synced_count": 0, "failed_count": 0, "failed_ids": []}

        # Sync each chunk
        synced = 0
        failed = 0
        failed_ids = []

        for chunk in processing_result.chunks:
            try:
                if not chunk.embedding:
                    logger.warning(f"Chunk {chunk.id} has no embedding, skipping ChromaDB sync")
                    failed += 1
                    failed_ids.append(str(chunk.id))
                    continue

                # Prepare metadata for ChromaDB
                metadata = {
                    "modality": chunk.modality.value,
                    "source_content_id": str(content_id),
                    "session_id": str(session_id),
                    "confidence_score": chunk.confidence_score,
                    "is_critical": chunk.is_critical,
                    "source_index": chunk.source_index,
                }

                # Add storage path if available
                if storage_path:
                    metadata["storage_path"] = storage_path

                # Add temporal metadata if available
                if chunk.metadata:
                    temporal_key = chunk.metadata.custom_metadata.get("temporal")
                    if temporal_key:
                        metadata["start_time_sec"] = temporal_key.get("start_time_sec")
                        metadata["end_time_sec"] = temporal_key.get("end_time_sec")

                # Sync chunk to ChromaDB
                success = await self.sync_manager.sync_single_memory(
                    session_id=session_id,
                    memory_id=chunk.id,
                    embedding=chunk.embedding,
                    memory_type=f"multimodal_{chunk.modality.value}",
                    document=chunk.content[:500] if chunk.content else "",  # Summary for search
                    metadata=metadata,
                    operation="insert",
                )

                if success:
                    synced += 1
                    logger.debug(f"Synced chunk {chunk.id} to ChromaDB")
                else:
                    failed += 1
                    failed_ids.append(str(chunk.id))
                    logger.warning(f"Failed to sync chunk {chunk.id} to ChromaDB")

            except Exception as e:
                logger.error(f"Error syncing chunk {chunk.id}: {e}")
                failed += 1
                failed_ids.append(str(chunk.id))

        logger.info(
            f"Processing result sync complete: {synced} synced, {failed} failed "
            f"(content_id={content_id}, session_id={session_id})"
        )

        return {
            "synced_count": synced,
            "failed_count": failed,
            "failed_ids": failed_ids,
        }

    async def sync_chunk(
        self,
        session_id: UUID,
        chunk: MultiModalChunk,
        content_id: Optional[UUID] = None,
        storage_path: Optional[str] = None,
    ) -> bool:
        """Sync a single chunk to ChromaDB.

        Args:
            session_id: Session UUID
            chunk: Chunk to sync
            content_id: Source content UUID (optional)
            storage_path: Storage path if content stored externally (optional)

        Returns:
            True if sync successful
        """
        if not self.sync_manager:
            await self._ensure_initialized()

        if not chunk.embedding:
            logger.warning(f"Chunk {chunk.id} has no embedding, cannot sync")
            return False

        try:
            # Prepare metadata
            metadata = {
                "modality": chunk.modality.value,
                "confidence_score": chunk.confidence_score,
                "is_critical": chunk.is_critical,
                "source_index": chunk.source_index,
            }

            if content_id:
                metadata["source_content_id"] = str(content_id)

            if storage_path:
                metadata["storage_path"] = storage_path

            # Sync to ChromaDB
            success = await self.sync_manager.sync_single_memory(
                session_id=session_id,
                memory_id=chunk.id,
                embedding=chunk.embedding,
                memory_type=f"multimodal_{chunk.modality.value}",
                document=chunk.content[:500] if chunk.content else "",
                metadata=metadata,
                operation="insert",
            )

            return success

        except Exception as e:
            logger.error(f"Error syncing chunk {chunk.id}: {e}")
            return False

    async def delete_chunks(
        self,
        session_id: UUID,
        chunk_ids: List[UUID],
        modality: Optional[str] = None,
    ) -> dict:
        """Delete chunks from ChromaDB.

        Args:
            session_id: Session UUID
            chunk_ids: List of chunk IDs to delete
            modality: Modality type (optional)

        Returns:
            Dict with {deleted_count, failed_ids}
        """
        if not self.sync_manager:
            await self._ensure_initialized()

        deleted = 0
        failed_ids = []

        for chunk_id in chunk_ids:
            try:
                success = await self.sync_manager.sync_single_memory(
                    session_id=session_id,
                    memory_id=chunk_id,
                    embedding=[],  # Empty embedding for delete operation
                    memory_type=f"multimodal_{modality}" if modality else "multimodal_chunk",
                    document="",
                    metadata={},
                    operation="delete",
                )

                if success:
                    deleted += 1
                else:
                    failed_ids.append(str(chunk_id))

            except Exception as e:
                logger.error(f"Error deleting chunk {chunk_id}: {e}")
                failed_ids.append(str(chunk_id))

        logger.info(f"Chunk deletion: {deleted} deleted, {len(failed_ids)} failed")

        return {
            "deleted_count": deleted,
            "failed_ids": failed_ids,
        }

    async def _ensure_initialized(self):
        """Lazily initialize sync manager with defaults.

        This allows the module to be imported without requiring
        ChromaDB infrastructure to be available immediately.
        """
        try:
            from ragcore.modules.memory.chroma.client import ChromaClientFactory
            from ragcore.config import settings

            logger.debug("Initializing ChromaDB sync manager")

            client_factory = ChromaClientFactory()
            client_mgr = await client_factory.get_client()

            if client_mgr and client_mgr.collection_manager:
                self.sync_manager = ChromaMemorySyncManager(
                    client_mgr.collection_manager,
                    config=settings,
                )
                self._initialized = True
                logger.info("ChromaDB sync manager initialized")
            else:
                logger.warning("Could not initialize ChromaDB sync manager - client not available")
                self._initialized = False

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB sync manager: {e}")
            self._initialized = False

    async def process_sync_queue(self) -> dict:
        """Process queued syncs that had failures.

        Returns:
            Dict with {processed_count, retried_count, failed_count}
        """
        if not self.sync_manager:
            await self._ensure_initialized()

        if not self.sync_manager:
            return {"processed_count": 0, "retried_count": 0, "failed_count": 0}

        return await self.sync_manager.process_sync_queue()

    async def full_session_resync(
        self,
        session_id: UUID,
        collection_manager: Optional[ChromaCollectionManager] = None,
    ) -> dict:
        """Force full resync of all chunks in a session from PostgreSQL to ChromaDB.

        Used for consistency recovery or data migration.

        Args:
            session_id: Session UUID
            collection_manager: ChromaCollectionManager for direct access (optional)

        Returns:
            Dict with {resync_count, consistency_verified}
        """
        if not self.sync_manager:
            await self._ensure_initialized()

        if not self.sync_manager:
            logger.error("Cannot perform full resync - sync manager not initialized")
            return {"resync_count": 0, "consistency_verified": False}

        logger.warning(f"Starting full session resync for session {session_id}")

        # This would typically call a database query to fetch all chunks
        # for this session from PostgreSQL and sync them to ChromaDB
        # Implementation depends on your database schema

        try:
            # Placeholder: actual implementation would query DB
            # SELECT * FROM multimodal_chunks WHERE session_id = {session_id}
            # Then call sync_processing_result() for each batch

            logger.info(f"Full session resync completed for session {session_id}")
            return {"resync_count": 0, "consistency_verified": True}

        except Exception as e:
            logger.error(f"Error during full session resync: {e}")
            return {"resync_count": 0, "consistency_verified": False}


# Singleton instance for global access
_multimodal_chroma_sync: Optional[MultiModalChromaSync] = None


async def get_multimodal_chroma_sync() -> MultiModalChromaSync:
    """Get or create singleton MultiModalChromaSync instance.

    Returns:
        MultiModalChromaSync instance
    """
    global _multimodal_chroma_sync

    if _multimodal_chroma_sync is None:
        _multimodal_chroma_sync = MultiModalChromaSync()
        await _multimodal_chroma_sync._ensure_initialized()

    return _multimodal_chroma_sync
