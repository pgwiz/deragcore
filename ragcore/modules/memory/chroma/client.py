"""ChromaDB client wrapper - Connection pooling, initialization, health checks."""

import logging
from typing import Optional
import asyncio
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ChromaConfig:
    """ChromaDB configuration."""

    def __init__(
        self,
        enabled: bool = True,
        deployment_mode: str = "hybrid",  # hybrid, chroma_primary, postgres_only
        host: str = "localhost",
        port: int = 8000,
        persistence_mode: str = "persistent",  # persistent, ephemeral
        persistence_path: str = "/data/chroma",
        connection_timeout_seconds: int = 5,
        connection_pool_size: int = 10,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_reset_minutes: int = 10,
    ):
        self.enabled = enabled
        self.deployment_mode = deployment_mode
        self.host = host
        self.port = port
        self.persistence_mode = persistence_mode
        self.persistence_path = persistence_path
        self.connection_timeout_seconds = connection_timeout_seconds
        self.connection_pool_size = connection_pool_size
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_reset_minutes = circuit_breaker_reset_minutes

    @property
    def url(self) -> str:
        """ChromaDB HTTP client URL."""
        return f"http://{self.host}:{self.port}"


class ChromaClientManager:
    """Manages ChromaDB client connection with pooling and health checks."""

    def __init__(self, config: ChromaConfig):
        """Initialize ChromaDB client manager.

        Args:
            config: ChromaDB configuration
        """
        self.config = config
        self.client = None
        self.last_health_check = None
        self.consecutive_failures = 0
        self.circuit_breaker_until = None
        self.connection_timeout = config.connection_timeout_seconds

        logger.info(
            f"ChromaClientManager initialized: "
            f"mode={config.deployment_mode}, "
            f"url={config.url}, "
            f"persistence={config.persistence_mode}"
        )

    async def initialize(self) -> bool:
        """Initialize ChromaDB client connection.

        Returns:
            True if successfully initialized, False otherwise
        """
        if not self.config.enabled:
            logger.info("ChromaDB disabled, skipping initialization")
            return False

        try:
            import chromadb

            if self.config.persistence_mode == "persistent":
                self.client = chromadb.HttpClient(
                    host=self.config.host,
                    port=self.config.port,
                )
            else:
                self.client = chromadb.HttpClient(
                    host=self.config.host,
                    port=self.config.port,
                )

            # Attempt health check
            health = await self.health_check()
            if health["status"] == "healthy":
                logger.info("ChromaDB client initialized successfully")
                return True
            else:
                logger.warning(f"ChromaDB health check failed: {health}")
                return False

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            self.consecutive_failures += 1
            return False

    async def health_check(self) -> dict:
        """Check ChromaDB health and connectivity.

        Returns:
            Dict with status, latency_ms, and message
        """
        if not self.config.enabled or self.client is None:
            return {
                "status": "unavailable",
                "latency_ms": None,
                "message": "ChromaDB not enabled or not initialized",
            }

        # Check circuit breaker
        if self.circuit_breaker_until and datetime.utcnow() < self.circuit_breaker_until:
            return {
                "status": "circuit_breaker_open",
                "latency_ms": None,
                "message": f"Circuit breaker open until {self.circuit_breaker_until}",
            }

        try:
            start = datetime.utcnow()

            # Simple health check: count collections
            try:
                collections = self.client.list_collections()
                latency_ms = (datetime.utcnow() - start).total_seconds() * 1000

                self.last_health_check = datetime.utcnow()
                self.consecutive_failures = 0

                return {
                    "status": "healthy",
                    "latency_ms": round(latency_ms, 2),
                    "message": f"ChromaDB healthy ({len(collections)} collections)",
                }
            except Exception as e:
                latency_ms = (datetime.utcnow() - start).total_seconds() * 1000
                self.consecutive_failures += 1

                # Activate circuit breaker if threshold reached
                if self.consecutive_failures >= self.config.circuit_breaker_threshold:
                    self.circuit_breaker_until = datetime.utcnow() + timedelta(
                        minutes=self.config.circuit_breaker_reset_minutes
                    )
                    logger.warning(
                        f"Circuit breaker activated: {self.consecutive_failures} consecutive failures"
                    )

                return {
                    "status": "unhealthy",
                    "latency_ms": round(latency_ms, 2),
                    "message": f"ChromaDB health check failed: {str(e)}",
                }

        except Exception as e:
            logger.error(f"Health check error: {e}")
            return {
                "status": "error",
                "latency_ms": None,
                "message": f"Health check error: {str(e)}",
            }

    async def get_client(self):
        """Get ChromaDB client, initializing if needed.

        Returns:
            ChromaDB client or None if unavailable
        """
        if not self.config.enabled:
            return None

        # Check circuit breaker
        if self.circuit_breaker_until and datetime.utcnow() < self.circuit_breaker_until:
            logger.debug("Circuit breaker is open, returning None")
            return None

        if self.client is None:
            await self.initialize()

        return self.client

    async def reset_circuit_breaker(self) -> None:
        """Manually reset circuit breaker."""
        self.circuit_breaker_until = None
        self.consecutive_failures = 0
        logger.info("Circuit breaker reset")

    async def close(self) -> None:
        """Close ChromaDB client connection."""
        if self.client is not None:
            try:
                # ChromaDB HTTP client doesn't need explicit close
                # but we can clean up in derived classes
                logger.info("ChromaDB client closed")
                self.client = None
            except Exception as e:
                logger.error(f"Error closing ChromaDB client: {e}")

    def is_healthy(self) -> bool:
        """Quick check if ChromaDB is likely healthy.

        Returns:
            True if not circuit breaker open and client initialized
        """
        if not self.config.enabled:
            return False

        if self.circuit_breaker_until and datetime.utcnow() < self.circuit_breaker_until:
            return False

        return self.client is not None
