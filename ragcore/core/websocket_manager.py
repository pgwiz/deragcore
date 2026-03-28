"""WebSocket connection manager - Handle concurrent streaming connections."""

import logging
from typing import Dict, Set
from uuid import UUID

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manage WebSocket connections for streaming chat and file progress."""

    def __init__(self):
        """Initialize manager with empty connection registry."""
        # sessions_id -> set of WebSocket connections
        self._connections: Dict[str, Set[WebSocket]] = {}

    async def connect(
        self,
        session_id: str,
        websocket: WebSocket,
    ) -> None:
        """
        Register a WebSocket connection for a session.

        Args:
            session_id: Session identifier (UUID as string)
            websocket: WebSocket connection
        """
        await websocket.accept()

        if session_id not in self._connections:
            self._connections[session_id] = set()

        self._connections[session_id].add(websocket)
        logger.debug(
            f"WebSocket connected: session={session_id}, "
            f"total={len(self._connections[session_id])}"
        )

    async def disconnect(
        self,
        session_id: str,
        websocket: WebSocket,
    ) -> None:
        """
        Unregister a WebSocket connection.

        Args:
            session_id: Session identifier
            websocket: WebSocket connection to remove
        """
        if session_id in self._connections:
            self._connections[session_id].discard(websocket)

            if not self._connections[session_id]:
                del self._connections[session_id]

        logger.debug(f"WebSocket disconnected: session={session_id}")

    async def broadcast(
        self,
        session_id: str,
        message: dict,
    ) -> None:
        """
        Send message to all connections in a session.

        Args:
            session_id: Target session
            message: JSON message to broadcast

        Example:
            await manager.broadcast(str(session_id), {
                "type": "token",
                "delta": "Hello"
            })
        """
        if session_id not in self._connections:
            logger.debug(f"No connections for session {session_id}")
            return

        # Send to all connections, skip failures
        disconnected = []
        for websocket in list(self._connections[session_id]):
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.warning(f"Failed to send to connection: {str(e)}")
                disconnected.append(websocket)

        # Remove disconnected websockets
        for ws in disconnected:
            await self.disconnect(session_id, ws)

    async def send_to_session(
        self,
        session_id: str,
        message_type: str,
        data: dict,
    ) -> None:
        """
        Send typed message to session.

        Args:
            session_id: Target session
            message_type: Type of message ('token', 'sources', 'error', 'done', etc)
            data: Message payload
        """
        message = {"type": message_type, **data}
        await self.broadcast(session_id, message)

    def get_session_connection_count(self, session_id: str) -> int:
        """Get number of active connections for session."""
        return len(self._connections.get(session_id, set()))

    def get_active_sessions(self) -> Dict[str, int]:
        """Get count of connections per session."""
        return {
            session_id: len(conns)
            for session_id, conns in self._connections.items()
            if conns
        }


# Global manager instance
websocket_manager = WebSocketManager()
