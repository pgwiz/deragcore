"""Episodic memory management for research/chat episodes."""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from uuid import UUID

from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from ragcore.db.database import async_session_factory
from ragcore.modules.memory.models import EpisodicSnapshot

logger = logging.getLogger(__name__)


class EpisodicMemory:
    """Manage episodic memory for research and chat episodes."""

    def __init__(self):
        """Initialize episodic memory."""
        self.default_ttl_seconds = 365 * 24 * 60 * 60  # 1 year
        logger.info("EpisodicMemory initialized")

    async def save_episode(
        self,
        session_id: UUID,
        episode_type: str,
        title: str,
        description: str,
        input_query: str,
        output_summary: Optional[str] = None,
        key_findings: Optional[List[Dict[str, Any]]] = None,
        sources_used: Optional[List[str]] = None,
        tools_invoked: Optional[List[Dict[str, Any]]] = None,
        actions_taken: Optional[List[Dict[str, Any]]] = None,
        related_memories: Optional[List[UUID]] = None,
        duration_ms: Optional[int] = None,
        tokens_used: int = 0,
        success: bool = True,
        tags: Optional[List[str]] = None,
        user_id: Optional[UUID] = None,
    ) -> Optional[UUID]:
        """Save an episodic memory snapshot.

        Args:
            session_id: Session this episode belongs to
            episode_type: Type (research, chat, analysis)
            title: Episode title
            description: Episode description
            input_query: User query/input
            output_summary: Summary of findings/output
            key_findings: List of findings with confidence
            sources_used: List of sources (URLs, files)
            tools_invoked: Tools used during episode
            actions_taken: Actions taken
            related_memories: Related memory IDs
            duration_ms: How long episode took
            tokens_used: Tokens consumed
            success: Whether episode succeeded
            tags: Episode tags
            user_id: User ID if available

        Returns:
            Episode ID if successful
        """
        try:
            async with async_session_factory() as session:
                # Get episode count to set episode_number
                stmt = select(EpisodicSnapshot).where(
                    EpisodicSnapshot.session_id == session_id
                )
                result = await session.execute(stmt)
                episode_count = len(result.scalars().all())
                episode_number = episode_count + 1

                expires_at = datetime.utcnow() + timedelta(seconds=self.default_ttl_seconds)

                episode = EpisodicSnapshot(
                    session_id=session_id,
                    user_id=user_id,
                    episode_number=episode_number,
                    episode_type=episode_type,
                    title=title,
                    description=description,
                    input_query=input_query,
                    output_summary=output_summary,
                    key_findings=key_findings or [],
                    sources_used=sources_used or [],
                    tools_invoked=tools_invoked or [],
                    actions_taken=actions_taken or [],
                    related_memories=related_memories or [],
                    duration_ms=duration_ms,
                    tokens_used=tokens_used,
                    success=success,
                    tags=tags or [],
                    expires_at=expires_at,
                )
                session.add(episode)
                await session.commit()
                await session.refresh(episode)

                logger.info(
                    f"Saved episode {episode.id}: {episode_type} #{episode_number}"
                )
                return episode.id

        except Exception as e:
            logger.error(f"Error saving episode: {str(e)}", exc_info=True)
            return None

    async def get_episode(self, episode_id: UUID) -> Optional[EpisodicSnapshot]:
        """Get an episodic snapshot.

        Args:
            episode_id: Episode to retrieve

        Returns:
            EpisodicSnapshot or None
        """
        try:
            async with async_session_factory() as session:
                stmt = select(EpisodicSnapshot).where(
                    EpisodicSnapshot.id == episode_id
                )
                result = await session.execute(stmt)
                return result.scalar_one_or_none()

        except Exception as e:
            logger.error(f"Error retrieving episode {episode_id}: {str(e)}", exc_info=True)
            return None

    async def get_session_episodes(
        self,
        session_id: UUID,
        episode_type: Optional[str] = None,
        limit: int = 20,
    ) -> List[EpisodicSnapshot]:
        """Get all episodes in a session.

        Args:
            session_id: Session ID
            episode_type: Optional filter by type
            limit: Max results

        Returns:
            List of episodes ordered by recency
        """
        try:
            async with async_session_factory() as session:
                stmt = select(EpisodicSnapshot).where(
                    EpisodicSnapshot.session_id == session_id
                )

                if episode_type:
                    stmt = stmt.where(EpisodicSnapshot.episode_type == episode_type)

                stmt = stmt.order_by(
                    EpisodicSnapshot.episode_number.desc()
                ).limit(limit)

                result = await session.execute(stmt)
                episodes = result.scalars().all()

                logger.debug(f"Retrieved {len(episodes)} episodes from session {session_id}")
                return episodes

        except Exception as e:
            logger.error(f"Error retrieving session episodes: {str(e)}", exc_info=True)
            return []

    async def get_recent_episodes(
        self,
        session_id: UUID,
        days_back: int = 7,
        limit: int = 10,
    ) -> List[EpisodicSnapshot]:
        """Get recent episodes from a session.

        Args:
            session_id: Session ID
            days_back: How many days back to look
            limit: Max results

        Returns:
            List of recent episodes
        """
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)

            async with async_session_factory() as session:
                stmt = select(EpisodicSnapshot).where(
                    and_(
                        EpisodicSnapshot.session_id == session_id,
                        EpisodicSnapshot.created_at >= cutoff_date,
                    )
                ).order_by(
                    EpisodicSnapshot.created_at.desc()
                ).limit(limit)

                result = await session.execute(stmt)
                episodes = result.scalars().all()

                return episodes

        except Exception as e:
            logger.error(f"Error retrieving recent episodes: {str(e)}", exc_info=True)
            return []

    async def search_episodes(
        self,
        session_id: UUID,
        query: str,
        limit: int = 10,
    ) -> List[EpisodicSnapshot]:
        """Search episodes by title, description, or query.

        Args:
            session_id: Session ID
            query: Search query
            limit: Max results

        Returns:
            List of matching episodes
        """
        try:
            async with async_session_factory() as session:
                stmt = select(EpisodicSnapshot).where(
                    and_(
                        EpisodicSnapshot.session_id == session_id,
                    )
                )

                # Search in multiple fields
                stmt = stmt.filter(
                    or_(
                        EpisodicSnapshot.title.ilike(f"%{query}%"),
                        EpisodicSnapshot.description.ilike(f"%{query}%"),
                        EpisodicSnapshot.input_query.ilike(f"%{query}%"),
                        EpisodicSnapshot.output_summary.ilike(f"%{query}%"),
                    )
                )

                stmt = stmt.order_by(
                    EpisodicSnapshot.created_at.desc()
                ).limit(limit)

                result = await session.execute(stmt)
                episodes = result.scalars().all()

                logger.debug(f"Found {len(episodes)} episodes matching '{query}'")
                return episodes

        except Exception as e:
            logger.error(f"Error searching episodes: {str(e)}", exc_info=True)
            return []

    async def get_episode_summary(
        self,
        session_id: UUID,
        num_recent: int = 5,
    ) -> Dict[str, Any]:
        """Get a summary of recent episodes.

        Args:
            session_id: Session ID
            num_recent: Number of recent episodes to summarize

        Returns:
            Summary dict with episode stats and findings
        """
        try:
            episodes = await self.get_recent_episodes(
                session_id=session_id,
                days_back=30,
                limit=num_recent,
            )

            all_findings = []
            all_sources = set()
            total_tokens = 0

            for episode in episodes:
                all_findings.extend(episode.key_findings)
                total_tokens += episode.tokens_used
                all_sources.update(episode.sources_used)

            return {
                "episode_count": len(episodes),
                "unique_sources": len(all_sources),
                "total_tokens": total_tokens,
                "total_findings": len(all_findings),
                "recent_episode_types": list(set(e.episode_type for e in episodes)),
                "successful_episodes": sum(1 for e in episodes if e.success),
            }

        except Exception as e:
            logger.error(f"Error getting episode summary: {str(e)}", exc_info=True)
            return {}


# Global episodic memory instance
episodic_memory = EpisodicMemory()
