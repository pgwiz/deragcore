"""ChromaDB performance router - Smart backend selection based on latency metrics."""

import logging
from typing import List, Tuple, Optional, Dict, Any
from uuid import UUID
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Track performance metrics for a backend."""

    def __init__(self, max_samples: int = 100):
        """Initialize metrics tracker.

        Args:
            max_samples: Maximum latency samples to keep
        """
        self.max_samples = max_samples
        self.latencies = deque(maxlen=max_samples)
        self.failure_count = 0
        self.success_count = 0
        self.last_updated = None

    def record_success(self, latency_ms: float) -> None:
        """Record successful operation.

        Args:
            latency_ms: Operation latency in milliseconds
        """
        self.latencies.append(latency_ms)
        self.success_count += 1
        self.last_updated = datetime.utcnow()

    def record_failure(self) -> None:
        """Record failed operation."""
        self.failure_count += 1
        self.last_updated = datetime.utcnow()

    def get_p50_latency(self) -> float:
        """Get 50th percentile latency.

        Returns:
            P50 latency in ms, or None if no data
        """
        if not self.latencies:
            return None

        sorted_latencies = sorted(self.latencies)
        idx = len(sorted_latencies) // 2
        return sorted_latencies[idx]

    def get_p95_latency(self) -> float:
        """Get 95th percentile latency."""
        if not self.latencies:
            return None

        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[idx]

    def get_avg_latency(self) -> float:
        """Get average latency."""
        if not self.latencies:
            return None

        return sum(self.latencies) / len(self.latencies)

    def get_error_rate(self) -> float:
        """Get error rate (0.0-1.0)."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.failure_count / total


class SmartSearchRouter:
    """Routes searches to optimal backend based on performance history."""

    def __init__(
        self,
        chroma_manager=None,
        pgvector_fallback=None,
        performance_threshold_ratio: float = 0.8,
    ):
        """Initialize smart router.

        Args:
            chroma_manager: ChromaDB client manager
            pgvector_fallback: Fallback pgvector search function
            performance_threshold_ratio: Ratio for backend preference (chroma should be < pgvector * ratio)
        """
        self.chroma_manager = chroma_manager
        self.pgvector_fallback = pgvector_fallback
        self.performance_threshold_ratio = performance_threshold_ratio

        self.chroma_metrics = PerformanceMetrics()
        self.pgvector_metrics = PerformanceMetrics()
        self.session_preferences = {}  # {session_id: "chroma"|"pgvector"}

    def _should_prefer_chroma(self) -> bool:
        """Decide if ChromaDB should be preferred based on metrics.

        Returns:
            True if ChromaDB should be preferred
        """
        chroma_p50 = self.chroma_metrics.get_p50_latency()
        pgvector_p50 = self.pgvector_metrics.get_p50_latency()

        # Not enough data
        if chroma_p50 is None or pgvector_p50 is None:
            # Prefer ChromaDB if available (generally faster)
            return self.chroma_manager and self.chroma_manager.is_healthy()

        # ChromaDB is faster if p50 is significantly better
        # (80% of pgvector speed or less)
        return chroma_p50 < (pgvector_p50 * self.performance_threshold_ratio)

    async def route_search(
        self,
        session_id: UUID,
        query_embedding: List[float],
        limit: int = 5,
        memory_type: str = "all",
        where_filter: Optional[Dict[str, Any]] = None,
        force_backend: Optional[str] = None,
    ) -> dict:
        """Route search to optimal backend with fallback.

        Args:
            session_id: Session UUID
            query_embedding: Query embedding vector
            limit: Max results to return
            memory_type: Memory type filter
            where_filter: Optional metadata filter
            force_backend: Force specific backend ("chroma" or "pgvector")

        Returns:
            Dict with {results, backend_used, latency_ms, fallback_occurred}
        """
        if force_backend == "chroma" and self.chroma_manager:
            return await self._search_chroma(
                session_id, query_embedding, limit, memory_type, where_filter
            )

        if force_backend == "pgvector" and self.pgvector_fallback:
            return await self._search_pgvector(
                session_id, query_embedding, limit, memory_type, where_filter
            )

        # Smart routing: try preferred backend first
        prefer_chroma = self._should_prefer_chroma()

        if prefer_chroma and self.chroma_manager:
            result = await self._search_chroma(
                session_id, query_embedding, limit, memory_type, where_filter
            )
            if result["results"] or not self.pgvector_fallback:
                return result

            # Fallback to pgvector
            logger.info(
                f"ChromaDB search for session {session_id} failed, "
                f"falling back to pgvector"
            )
            result2 = await self._search_pgvector(
                session_id, query_embedding, limit, memory_type, where_filter
            )
            result2["fallback_occurred"] = True
            return result2

        else:
            # Prefer pgvector
            if self.pgvector_fallback:
                result = await self._search_pgvector(
                    session_id, query_embedding, limit, memory_type, where_filter
                )
                return result

            # Fall back to ChromaDB
            if self.chroma_manager:
                return await self._search_chroma(
                    session_id, query_embedding, limit, memory_type, where_filter
                )

        # No backend available
        logger.error("No search backend available")
        return {
            "results": [],
            "backend_used": "none",
            "latency_ms": 0,
            "fallback_occurred": False,
            "error": "No search backend available",
        }

    async def _search_chroma(
        self,
        session_id: UUID,
        query_embedding: List[float],
        limit: int,
        memory_type: str,
        where_filter: Optional[Dict[str, Any]],
    ) -> dict:
        """Search ChromaDB with latency tracking.

        Args:
            session_id, query_embedding, limit, memory_type, where_filter: Search params

        Returns:
            Dict with {results, backend_used, latency_ms, fallback_occurred}
        """
        from datetime import datetime

        start = datetime.utcnow()

        try:
            from ragcore.modules.memory.chroma.collection_manager import (
                ChromaCollectionManager,
            )

            if not self.chroma_manager:
                raise ValueError("ChromaDB not available")

            # Use collection manager to search
            # Note: This assumes collection_manager is available
            results = await self.chroma_manager.get_client()  # Placeholder

            # In real implementation:
            # collection_manager = ChromaCollectionManager(self.chroma_manager)
            # results = await collection_manager.semantic_search(...)

            latency_ms = (datetime.utcnow() - start).total_seconds() * 1000
            self.chroma_metrics.record_success(latency_ms)

            return {
                "results": results or [],
                "backend_used": "chroma",
                "latency_ms": latency_ms,
                "fallback_occurred": False,
            }

        except Exception as e:
            latency_ms = (datetime.utcnow() - start).total_seconds() * 1000
            self.chroma_metrics.record_failure()

            logger.warning(f"ChromaDB search error: {e}")

            return {
                "results": [],
                "backend_used": "chroma",
                "latency_ms": latency_ms,
                "fallback_occurred": False,
                "error": str(e),
            }

    async def _search_pgvector(
        self,
        session_id: UUID,
        query_embedding: List[float],
        limit: int,
        memory_type: str,
        where_filter: Optional[Dict[str, Any]],
    ) -> dict:
        """Search pgvector with latency tracking.

        Returns:
            Dict with {results, backend_used, latency_ms, fallback_occurred}
        """
        from datetime import datetime

        start = datetime.utcnow()

        try:
            if not self.pgvector_fallback:
                raise ValueError("pgvector not available")

            results = await self.pgvector_fallback(
                session_id, query_embedding, limit
            )

            latency_ms = (datetime.utcnow() - start).total_seconds() * 1000
            self.pgvector_metrics.record_success(latency_ms)

            return {
                "results": results or [],
                "backend_used": "pgvector",
                "latency_ms": latency_ms,
                "fallback_occurred": False,
            }

        except Exception as e:
            latency_ms = (datetime.utcnow() - start).total_seconds() * 1000
            self.pgvector_metrics.record_failure()

            logger.warning(f"pgvector search error: {e}")

            return {
                "results": [],
                "backend_used": "pgvector",
                "latency_ms": latency_ms,
                "fallback_occurred": False,
                "error": str(e),
            }

    def get_performance_stats(self) -> dict:
        """Get performance comparison metrics.

        Returns:
            Dict with latency and error rate stats
        """
        return {
            "chroma": {
                "p50_latency_ms": self.chroma_metrics.get_p50_latency(),
                "p95_latency_ms": self.chroma_metrics.get_p95_latency(),
                "avg_latency_ms": self.chroma_metrics.get_avg_latency(),
                "error_rate": self.chroma_metrics.get_error_rate(),
                "success_count": self.chroma_metrics.success_count,
                "failure_count": self.chroma_metrics.failure_count,
            },
            "pgvector": {
                "p50_latency_ms": self.pgvector_metrics.get_p50_latency(),
                "p95_latency_ms": self.pgvector_metrics.get_p95_latency(),
                "avg_latency_ms": self.pgvector_metrics.get_avg_latency(),
                "error_rate": self.pgvector_metrics.get_error_rate(),
                "success_count": self.pgvector_metrics.success_count,
                "failure_count": self.pgvector_metrics.failure_count,
            },
            "preferred_backend": (
                "chroma" if self._should_prefer_chroma() else "pgvector"
            ),
        }
