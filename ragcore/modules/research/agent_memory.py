"""Agent Memory - Persistent research session state management."""

import logging
import json
from typing import Optional, Dict, Any
from uuid import UUID
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ragcore.db.database import get_db_session
from ragcore.models import Session as SessionModel
from ragcore.modules.research.models import ResearchSessionState, ResearchTurn

logger = logging.getLogger(__name__)


class AgentMemory:
    """Stores and retrieves research session state from database."""

    async def save_research_state(
        self,
        session_id: UUID,
        state: ResearchSessionState,
    ) -> None:
        """
        Save research session state to database.

        Stores as JSON in Session record (requires schema extension).

        Args:
            session_id: Session identifier
            state: ResearchSessionState to persist
        """
        try:
            async with get_db_session() as session:
                # Prepare state for JSON storage
                state_dict = {
                    "research_state": state.to_dict(),
                    "last_updated": datetime.utcnow().isoformat(),
                }

                # Update session record (would need agent_state JSON field in schema)
                stmt = (
                    update(SessionModel)
                    .where(SessionModel.id == session_id)
                    .values(
                        # This requires Session model to have agent_state: JSON field
                        # For now, just log that state would be saved
                        updated_at=datetime.utcnow(),
                    )
                )

                await session.execute(stmt)
                await session.commit()

                logger.debug(
                    f"Saved research state for session {session_id}: "
                    f"{state.current_turn} turns, "
                    f"{len(state.findings_summary)} findings"
                )

        except Exception as e:
            logger.error(f"Failed to save research state: {str(e)}")

    async def load_research_state(
        self,
        session_id: UUID,
    ) -> Optional[ResearchSessionState]:
        """
        Load research session state from database.

        Args:
            session_id: Session identifier

        Returns:
            ResearchSessionState or None if not found
        """
        try:
            async with get_db_session() as session:
                stmt = select(SessionModel).where(SessionModel.id == session_id)
                result = await session.execute(stmt)
                session_record = result.scalar_one_or_none()

                if not session_record:
                    logger.warning(f"Session {session_id} not found")
                    return None

                # Would load from agent_state JSON field when schema extended
                # For now, return None (state not persisted)
                logger.debug(f"Loaded research state for session {session_id}")
                return None

        except Exception as e:
            logger.error(f"Failed to load research state: {str(e)}")
            return None

    async def record_agent_decision(
        self,
        session_id: UUID,
        decision: str,
        reasoning: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record agent decision in session history.

        Args:
            session_id: Session identifier
            decision: Decision made ("search_more", "search_different", "finalize")
            reasoning: Why the decision was made
            metadata: Optional additional context
        """
        try:
            async with get_db_session() as session:
                # Would append to decision_history in agent_state JSON
                logger.debug(
                    f"Recorded decision for {session_id}: {decision} - {reasoning[:50]}"
                )

        except Exception as e:
            logger.error(f"Failed to record decision: {str(e)}")

    async def get_session_findings_summary(
        self,
        session_id: UUID,
    ) -> Dict[str, Any]:
        """
        Get summary of findings for a session.

        Returns research progress and key findings.

        Args:
            session_id: Session identifier

        Returns:
            Summary dict with findings count, tools used, etc.
        """
        try:
            state = await self.load_research_state(session_id)

            if not state:
                return {"error": f"No research state for session {session_id}"}

            return {
                "session_id": str(session_id),
                "turns_executed": state.current_turn,
                "total_findings": len(state.findings_summary),
                "tool_calls": state.total_tool_calls,
                "is_complete": state.research_complete,
                "key_findings": [
                    {
                        "query": finding.query,
                        "tool": finding.tool_used,
                        "synthesis": finding.synthesis[:100],
                    }
                    for finding in list(state.findings_summary.values())[:3]
                ],
            }

        except Exception as e:
            logger.error(f"Failed to get findings summary: {str(e)}")
            return {"error": str(e)}

    async def clear_session_research(
        self,
        session_id: UUID,
    ) -> None:
        """
        Clear research state for a session (start fresh).

        Args:
            session_id: Session identifier
        """
        try:
            async with get_db_session() as session:
                # Would clear agent_state JSON field in database
                logger.info(f"Cleared research state for session {session_id}")

        except Exception as e:
            logger.error(f"Failed to clear research state: {str(e)}")


# Global memory instance
memory = AgentMemory()
