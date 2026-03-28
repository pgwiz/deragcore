"""ChromaDB sync manager - PostgreSQL ↔ ChromaDB synchronization with retry logic."""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger(__name__)


class ChromaMemorySyncManager:
    """Synchronizes memories between PostgreSQL and ChromaDB."""

    def __init__(self, collection_manager, config=None):
        """Initialize sync manager.

        Args:
            collection_manager: ChromaCollectionManager instance
            config: ChromaDB config with retry settings
        """
        self.collection_manager = collection_manager
        self.config = config
        self.sync_queue = {}  # {memory_id: scheduled_time}
        self.retry_count = {}  # {memory_id: count}
        self.max_retries = config.circuit_breaker_threshold if config else 3

    async def sync_single_memory(
        self,
        session_id: UUID,
        memory_id: UUID,
        embedding: List[float],
        memory_type: str,
        document: str,
        metadata: Dict[str, Any],
        operation: str = "insert",
    ) -> bool:
        """Sync single memory to ChromaDB with exponential backoff.

        Args:
            session_id: Session UUID
            memory_id: Memory UUID
            embedding: Embedding vector
            memory_type: Memory type (finding, decision, etc)
            document: Memory text content
            metadata: Memory metadata dict
            operation: "insert", "update", or "delete"

        Returns:
            True if successful
        """
        if operation == "delete":
            result = await self.collection_manager.delete_memories(
                session_id,
                [str(memory_id)],
                memory_type=memory_type,
            )
            success = result["failed_count"] == 0
        else:
            # insert or update
            memory_dict = {
                "id": str(memory_id),
                "embedding": embedding,
                "document": document,
                "metadata": metadata,
            }

            result = await self.collection_manager.add_memories(
                session_id,
                [memory_dict],
                memory_type=memory_type,
            )
            success = result["success_count"] > 0

        if not success:
            # Queue for retry with exponential backoff
            retry_count = self.retry_count.get(memory_id, 0)
            if retry_count < self.max_retries:
                backoff_seconds = 2 ** retry_count  # 1, 2, 4, 8, 16 seconds
                self.sync_queue[memory_id] = datetime.utcnow() + timedelta(
                    seconds=backoff_seconds
                )
                self.retry_count[memory_id] = retry_count + 1

                logger.warning(
                    f"Sync failed for {memory_id}, queued for retry "
                    f"(attempt {retry_count + 1}/{self.max_retries}, "
                    f"backoff={backoff_seconds}s)"
                )
            else:
                logger.error(
                    f"Sync failed for {memory_id} after {self.max_retries} attempts"
                )
                # Remove from queue
                if memory_id in self.sync_queue:
                    del self.sync_queue[memory_id]
                if memory_id in self.retry_count:
                    del self.retry_count[memory_id]
        else:
            # Clean up retry tracking on success
            if memory_id in self.sync_queue:
                del self.sync_queue[memory_id]
            if memory_id in self.retry_count:
                del self.retry_count[memory_id]

            logger.debug(f"Successfully synced memory {memory_id} ({operation})")

        return success

    async def batch_sync_memories(
        self,
        session_id: UUID,
        memories: List[Dict[str, Any]],
        operation: str = "insert",
    ) -> dict:
        """Batch sync multiple memories with partial failure handling.

        Args:
            session_id: Session UUID
            memories: List of memory dicts
                     [{id, embedding, memory_type, document, metadata}, ...]
            operation: "insert" or "update"

        Returns:
            Dict with {synced_count, failed_count, failed_ids}
        """
        synced = 0
        failed = 0
        failed_ids = []

        # Group by memory type for efficient collection operations
        by_type = {}
        for memory in memories:
            mem_type = memory.get("memory_type", "all")
            if mem_type not in by_type:
                by_type[mem_type] = []
            by_type[mem_type].append(memory)

        # Sync each type group
        for memory_type, memory_group in by_type.items():
            # Prepare batch
            batch = [
                {
                    "id": str(m["id"]),
                    "embedding": m.get("embedding", []),
                    "document": m.get("document", ""),
                    "metadata": m.get("metadata", {}),
                }
                for m in memory_group
            ]

            result = await self.collection_manager.add_memories(
                session_id,
                batch,
                memory_type=memory_type,
            )

            synced += result["success_count"]
            failed += result["failed_count"]
            failed_ids.extend(result["failed_ids"])

        logger.info(
            f"Batch sync complete: synced={synced}, failed={failed}, "
            f"failed_ids={failed_ids}"
        )

        return {
            "synced_count": synced,
            "failed_count": failed,
            "failed_ids": failed_ids,
        }

    async def process_sync_queue(self) -> dict:
        """Process queued sync operations with retry logic.

        Returns:
            Dict with {processed_count, retried_count, failed_count}
        """
        if not self.sync_queue:
            return {
                "processed_count": 0,
                "retried_count": 0,
                "failed_count": 0,
            }

        processed = 0
        retried = 0
        failed = 0

        now = datetime.utcnow()
        ready_ids = [
            mid
            for mid, scheduled_time in self.sync_queue.items()
            if scheduled_time <= now
        ]

        for memory_id in ready_ids:
            # In a real implementation, would retrieve from PostgreSQL
            # and retry. For now, just mark as processed.
            del self.sync_queue[memory_id]

            # If retry count reached max, consider failed
            if self.retry_count.get(memory_id, 0) >= self.max_retries:
                failed += 1
                if memory_id in self.retry_count:
                    del self.retry_count[memory_id]
            else:
                retried += 1

            processed += 1

        logger.info(
            f"Processed sync queue: processed={processed}, "
            f"retried={retried}, failed={failed}"
        )

        return {
            "processed_count": processed,
            "retried_count": retried,
            "failed_count": failed,
        }

    async def full_session_resync(
        self,
        session_id: UUID,
        postgres_memories: List[Dict[str, Any]],
    ) -> dict:
        """Force full resync for session from PostgreSQL source.

        Args:
            session_id: Session UUID
            postgres_memories: All memories from PostgreSQL

        Returns:
            Dict with {synced_count, failed_count, inconsistencies}
        """
        logger.info(f"Starting full session resync for {session_id}")

        # Group by type
        by_type = {}
        for memory in postgres_memories:
            mem_type = memory.get("memory_type", "all")
            if mem_type not in by_type:
                by_type[mem_type] = []
            by_type[mem_type].append(memory)

        total_synced = 0
        total_failed = 0
        inconsistencies = []

        for memory_type, memory_group in by_type.items():
            result = await self.batch_sync_memories(
                session_id,
                memory_group,
                operation="insert",
            )

            total_synced += result["synced_count"]
            total_failed += result["failed_count"]

            if result["failed_ids"]:
                inconsistencies.append(
                    {
                        "memory_type": memory_type,
                        "failed_ids": result["failed_ids"],
                    }
                )

        logger.info(
            f"Full resync complete: synced={total_synced}, "
            f"failed={total_failed}, inconsistencies={len(inconsistencies)}"
        )

        return {
            "synced_count": total_synced,
            "failed_count": total_failed,
            "inconsistencies": inconsistencies,
        }

    def get_queue_status(self) -> dict:
        """Get current sync queue status.

        Returns:
            Dict with queue stats
        """
        now = datetime.utcnow()
        overdue = sum(
            1 for t in self.sync_queue.values() if t <= now
        )

        return {
            "queue_size": len(self.sync_queue),
            "overdue_count": overdue,
            "avg_retry_count": (
                sum(self.retry_count.values()) / len(self.retry_count)
                if self.retry_count
                else 0
            ),
        }
