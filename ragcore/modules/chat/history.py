"""Chat history management - Store and retrieve conversation turns."""

import logging
from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ragcore.db.database import get_db_session
from ragcore.models import Session as SessionModel

logger = logging.getLogger(__name__)


class ChatTurn:
    """Single conversation turn (user or assistant message)."""

    def __init__(
        self,
        role: str,
        content: str,
        created_at: datetime,
        sources: Optional[List[UUID]] = None,
    ):
        self.role = role  # 'user' or 'assistant'
        self.content = content
        self.created_at = created_at
        self.sources = sources or []  # File IDs used for retrieval

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "sources": [str(s) for s in self.sources],
        }

    def to_message(self) -> Dict[str, str]:
        """Convert to message format for AI provider."""
        return {"role": self.role, "content": self.content}


class ChatHistoryManager:
    """Manage chat session history."""

    async def get_recent(
        self,
        session_id: UUID,
        limit: int = 10,
    ) -> List[ChatTurn]:
        """
        Get recent chat turns for a session.

        For Phase 2: We store history in a simple JSON array in Session.metadata.
        In production, this would use separate ChatMessage table for scalability.

        Args:
            session_id: Session ID
            limit: Max turns to return

        Returns:
            List of ChatTurn objects (oldest first)
        """
        async with get_db_session() as session:
            session_record = await session.get(SessionModel, session_id)

            if not session_record:
                logger.warning(f"Session {session_id} not found")
                return []

            # For now, history is a placeholder
            # In production: SELECT * FROM chat_message WHERE session_id = session_id ORDER BY created_at
            # For Phase 2: Return empty list (will implement in Chat router)
            return []

    async def get_all(
        self,
        session_id: UUID,
    ) -> List[ChatTurn]:
        """
        Get complete conversation history for a session.

        Args:
            session_id: Session ID

        Returns:
            All ChatTurn objects in order
        """
        return await self.get_recent(session_id, limit=10000)

    async def clear(
        self,
        session_id: UUID,
    ) -> None:
        """
        Clear all history for a session.

        Args:
            session_id: Session ID to clear
        """
        # In production: DELETE FROM chat_message WHERE session_id = session_id
        # For Phase 2: Log only
        logger.info(f"Cleared history for session {session_id}")

    def format_as_messages(
        self,
        history: List[ChatTurn],
    ) -> List[Dict[str, str]]:
        """
        Convert history to message format for AI provider.

        Args:
            history: List of ChatTurn objects

        Returns:
            List of {role, content} dicts
        """
        return [turn.to_message() for turn in history]
