"""Long-term memory store with semantic search."""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from uuid import UUID

from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from ragcore.db.database import async_session_factory
from ragcore.modules.memory.models import (
    LongTermMemory,
    EpisodicSnapshot,
    MemoryAccessLog,
)

logger = logging.getLogger(__name__)


class LongTermMemoryStore:
    """Manage long-term memory persistence and retrieval."""

    def __init__(self):
        """Initialize memory store."""
        self.default_ttl_seconds = 365 * 24 * 60 * 60  # 1 year
        logger.info("LongTermMemoryStore initialized")

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
        """Save a memory entry.

        Args:
            session_id: Session this memory belongs to
            memory_type: Type of memory (finding, decision, insight, error)
            content: Full memory content
            summary: Optional short summary
            source: Origin of memory
            tags: Categorization tags
            importance_score: 0.0-1.0 importance rating
            ttl_seconds: Time to live (default 1 year)
            context_data: Additional context
            user_id: User ID if available

        Returns:
            Memory ID if successful
        """
        try:
            ttl = ttl_seconds or self.default_ttl_seconds
            expires_at = datetime.utcnow() + timedelta(seconds=ttl)

            async with async_session_factory() as session:
                memory = LongTermMemory(
                    session_id=session_id,
                    user_id=user_id,
                    memory_type=memory_type,
                    content=content,
                    summary=summary or content[:200],
                    source=source,
                    tags=tags or [],
                    importance_score=importance_score,
                    ttl_seconds=ttl,
                    context_data=context_data or {},
                    expires_at=expires_at,
                    is_active=True,
                )
                session.add(memory)
                await session.commit()
                await session.refresh(memory)

                logger.debug(f"Saved memory {memory.id} of type {memory_type}")
                return memory.id

        except Exception as e:
            logger.error(f"Error saving memory: {str(e)}", exc_info=True)
            return None

    async def get_memory(self, memory_id: UUID) -> Optional[LongTermMemory]:
        """Get a specific memory entry.

        Args:
            memory_id: Memory ID to retrieve

        Returns:
            LongTermMemory object or None
        """
        try:
            async with async_session_factory() as session:
                stmt = select(LongTermMemory).where(
                    and_(
                        LongTermMemory.id == memory_id,
                        LongTermMemory.is_active == True,
                    )
                )
                result = await session.execute(stmt)
                memory = result.scalar_one_or_none()

                if memory:
                    # Update access tracking
                    memory.access_count += 1
                    memory.last_accessed_at = datetime.utcnow()
                    session.add(memory)
                    await session.commit()

                return memory

        except Exception as e:
            logger.error(f"Error retrieving memory {memory_id}: {str(e)}", exc_info=True)
            return None

    async def search_session_memory(
        self,
        session_id: UUID,
        query: Optional[str] = None,
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[LongTermMemory]:
        """Search memories in a session.

        Args:
            session_id: Session to search in
            query: Text query (will do substring match)
            memory_type: Filter by memory type
            tags: Filter by tags (any match)
            limit: Max results

        Returns:
            List of matching memories
        """
        try:
            async with async_session_factory() as session:
                stmt = select(LongTermMemory).where(
                    and_(
                        LongTermMemory.session_id == session_id,
                        LongTermMemory.is_active == True,
                    )
                )

                # Add filters
                if memory_type:
                    stmt = stmt.where(LongTermMemory.memory_type == memory_type)

                if query:
                    # Simple substring search
                    stmt = stmt.where(
                        LongTermMemory.content.ilike(f"%{query}%")
                    )

                if tags:
                    # Match any tag
                    tag_conditions = [
                        LongTermMemory.tags.contains([tag])
                        for tag in tags
                    ]
                    stmt = stmt.where(or_(*tag_conditions))

                # Order by importance and recent access
                stmt = stmt.order_by(
                    LongTermMemory.importance_score.desc(),
                    LongTermMemory.last_accessed_at.desc(),
                ).limit(limit)

                result = await session.execute(stmt)
                memories = result.scalars().all()

                logger.debug(f"Found {len(memories)} memories in session {session_id}")
                return memories

        except Exception as e:
            logger.error(f"Error searching session memory: {str(e)}", exc_info=True)
            return []

    async def search_semantic(
        self,
        session_id: UUID,
        embedding: List[float],
        limit: int = 5,
        min_similarity: float = 0.5,
    ) -> List[tuple[LongTermMemory, float]]:
        """Semantic similarity search using pgvector.

        Args:
            session_id: Session to search in
            embedding: 1536-dim embedding vector
            limit: Max results
            min_similarity: Minimum similarity threshold (0.0-1.0)

        Returns:
            List of (memory, similarity_score) tuples
        """
        try:
            from sqlalchemy import func, literal_column

            async with async_session_factory() as session:
                # Calculate similarity using cosine distance
                # pgvector uses <-> for cosine distance (smaller = more similar)
                distance_expr = LongTermMemory.embedding.op("<->")(embedding)
                similarity_expr = (1 - distance_expr).label("similarity")

                stmt = select(
                    LongTermMemory,
                    similarity_expr,
                ).where(
                    and_(
                        LongTermMemory.session_id == session_id,
                        LongTermMemory.is_active == True,
                        LongTermMemory.embedding.isnot(None),
                    )
                )

                stmt = stmt.order_by(distance_expr).limit(limit)

                result = await session.execute(stmt)
                rows = result.fetchall()

                # Filter by min similarity and format results
                results = [
                    (row[0], float(row[1]))
                    for row in rows
                    if float(row[1]) >= min_similarity
                ]

                logger.debug(
                    f"Semantic search found {len(results)} memories with similarity >= {min_similarity}"
                )
                return results

        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}", exc_info=True)
            return []

    async def get_session_memories(
        self,
        session_id: UUID,
        memory_type: Optional[str] = None,
        limit: int = 20,
    ) -> List[LongTermMemory]:
        """Get all memories for a session.

        Args:
            session_id: Session ID
            memory_type: Optional filter by type
            limit: Max results

        Returns:
            List of memories
        """
        return await self.search_session_memory(
            session_id=session_id,
            memory_type=memory_type,
            limit=limit,
        )

    async def delete_memory(self, memory_id: UUID) -> bool:
        """Delete or mark a memory as inactive.

        Args:
            memory_id: Memory to delete

        Returns:
            Success status
        """
        try:
            async with async_session_factory() as session:
                stmt = select(LongTermMemory).where(LongTermMemory.id == memory_id)
                result = await session.execute(stmt)
                memory = result.scalar_one_or_none()

                if memory:
                    memory.is_active = False
                    session.add(memory)
                    await session.commit()
                    logger.info(f"Marked memory {memory_id} as inactive")
                    return True

                return False

        except Exception as e:
            logger.error(f"Error deleting memory {memory_id}: {str(e)}", exc_info=True)
            return False

    async def log_access(
        self,
        memory_id: UUID,
        session_id: UUID,
        access_type: str = "retrieval",
        query: Optional[str] = None,
        similarity_score: Optional[float] = None,
        user_id: Optional[UUID] = None,
    ) -> bool:
        """Log memory access for analytics.

        Args:
            memory_id: Memory that was accessed
            session_id: Session context
            access_type: Type of access (retrieval, update, delete)
            query: Query that matched this memory
            similarity_score: Similarity if from semantic search
            user_id: User ID if available

        Returns:
            Success status
        """
        try:
            async with async_session_factory() as session:
                log = MemoryAccessLog(
                    memory_id=memory_id,
                    session_id=session_id,
                    user_id=user_id,
                    access_type=access_type,
                    query=query,
                    similarity_score=similarity_score,
                )
                session.add(log)
                await session.commit()
                return True

        except Exception as e:
            logger.error(f"Error logging memory access: {str(e)}", exc_info=True)
            return False

    async def cleanup_expired_memories(self) -> Dict[str, int]:
        """Clean up expired memories.

        Returns:
            Stats {deleted: count, errors: count}
        """
        try:
            async with async_session_factory() as session:
                # Mark expired memories as inactive
                stmt = select(LongTermMemory).where(
                    and_(
                        LongTermMemory.expires_at <= datetime.utcnow(),
                        LongTermMemory.is_active == True,
                    )
                )

                result = await session.execute(stmt)
                memories = result.scalars().all()

                deleted_count = 0
                for memory in memories:
                    memory.is_active = False
                    session.add(memory)
                    deleted_count += 1

                await session.commit()

                logger.info(f"Cleanup: Marked {deleted_count} expired memories as inactive")
                return {"deleted": deleted_count, "errors": 0}

        except Exception as e:
            logger.error(f"Error in memory cleanup: {str(e)}", exc_info=True)
            return {"deleted": 0, "errors": 1}


# Global memory store instance
memory_store = LongTermMemoryStore()
