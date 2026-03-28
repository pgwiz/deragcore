"""Tests for Sprint 4: ChromaDB Integration."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from uuid import uuid4

from ragcore.modules.memory.chroma.client import ChromaClientManager, ChromaConfig
from ragcore.modules.memory.chroma.collection_manager import ChromaCollectionManager
from ragcore.modules.memory.chroma.sync_manager import ChromaMemorySyncManager
from ragcore.modules.memory.chroma.performance_router import (
    SmartSearchRouter,
    PerformanceMetrics,
)


class TestChromaConfig:
    """Test ChromaDB configuration."""

    def test_init_defaults(self):
        """Test default configuration values."""
        config = ChromaConfig()
        assert config.enabled is True
        assert config.deployment_mode == "hybrid"
        assert config.host == "localhost"
        assert config.port == 8000
        assert config.persistence_mode == "persistent"

    def test_init_custom(self):
        """Test custom configuration."""
        config = ChromaConfig(
            enabled=True,
            deployment_mode="chroma_primary",
            host="chroma.example.com",
            port=9000,
        )
        assert config.deployment_mode == "chroma_primary"
        assert config.host == "chroma.example.com"
        assert config.port == 9000

    def test_url_property(self):
        """Test URL generation."""
        config = ChromaConfig(host="chroma", port=8000)
        assert config.url == "http://chroma:8000"


class TestChromaClientManager:
    """Test ChromaDB client manager."""

    def test_init(self):
        """Test manager initialization."""
        config = ChromaConfig()
        manager = ChromaClientManager(config)
        assert manager.config == config
        assert manager.client is None
        assert manager.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_health_check_unavailable(self):
        """Test health check when disabled."""
        config = ChromaConfig(enabled=False)
        manager = ChromaClientManager(config)

        health = await manager.health_check()
        assert health["status"] == "unavailable"

    @pytest.mark.asyncio
    async def test_health_check_circuit_breaker_open(self):
        """Test health check with circuit breaker open."""
        config = ChromaConfig()
        manager = ChromaClientManager(config)
        manager.client = Mock()  # Set client so it passes the None check
        manager.circuit_breaker_until = datetime.utcnow() + timedelta(minutes=1)

        health = await manager.health_check()
        assert health["status"] == "circuit_breaker_open"

    @pytest.mark.asyncio
    async def test_is_healthy_when_disabled(self):
        """Test is_healthy check when disabled."""
        config = ChromaConfig(enabled=False)
        manager = ChromaClientManager(config)
        assert manager.is_healthy() is False

    @pytest.mark.asyncio
    async def test_reset_circuit_breaker(self):
        """Test circuit breaker reset."""
        config = ChromaConfig()
        manager = ChromaClientManager(config)
        manager.consecutive_failures = 5
        manager.circuit_breaker_until = datetime.utcnow() + timedelta(minutes=1)

        await manager.reset_circuit_breaker()
        assert manager.circuit_breaker_until is None
        assert manager.consecutive_failures == 0


class TestChromaCollectionManager:
    """Test ChromaDB collection manager."""

    def test_init(self):
        """Test collection manager initialization."""
        client_manager = Mock()
        manager = ChromaCollectionManager(client_manager)
        assert manager.client_manager == client_manager
        assert len(manager.collections_cache) == 0

    def test_make_collection_name(self):
        """Test collection name generation."""
        client_manager = Mock()
        client_manager.config.deployment_mode = "hybrid"

        manager = ChromaCollectionManager(client_manager)
        session_id = uuid4()

        name = manager._make_collection_name(session_id, "finding")
        assert name.startswith("chroma_hybrid_")
        assert name.endswith("_finding")

    @pytest.mark.asyncio
    async def test_get_or_create_collection_unavailable(self):
        """Test get_or_create when ChromaDB unavailable."""
        client_manager = AsyncMock()
        client_manager.get_client.return_value = None

        manager = ChromaCollectionManager(client_manager)
        session_id = uuid4()

        result = await manager.get_or_create_collection(session_id)
        assert result is None

    @pytest.mark.asyncio
    async def test_add_memories_unavailable(self):
        """Test add_memories when ChromaDB unavailable."""
        client_manager = AsyncMock()
        client_manager.get_client.return_value = None

        manager = ChromaCollectionManager(client_manager)
        session_id = uuid4()

        result = await manager.add_memories(
            session_id,
            [{"id": "1", "embedding": [0.1, 0.2], "document": "test"}],
        )

        assert result["success_count"] == 0
        assert result["failed_count"] == 1

    @pytest.mark.asyncio
    async def test_delete_memories_unavailable(self):
        """Test delete_memories when ChromaDB unavailable."""
        client_manager = AsyncMock()
        client_manager.get_client.return_value = None

        manager = ChromaCollectionManager(client_manager)
        session_id = uuid4()

        result = await manager.delete_memories(session_id, ["1", "2"])

        assert result["deleted_count"] == 0
        assert result["failed_count"] == 2

    @pytest.mark.asyncio
    async def test_semantic_search_unavailable(self):
        """Test semantic search when ChromaDB unavailable."""
        client_manager = AsyncMock()
        client_manager.get_client.return_value = None

        manager = ChromaCollectionManager(client_manager)
        session_id = uuid4()

        result = await manager.semantic_search(
            session_id,
            [0.1] * 1536,  # 1536-dim embedding
        )

        assert result == []


class TestChromaMemorySyncManager:
    """Test ChromaDB sync manager."""

    def test_init(self):
        """Test sync manager initialization."""
        collection_manager = Mock()
        config = ChromaConfig()

        manager = ChromaMemorySyncManager(collection_manager, config)
        assert manager.collection_manager == collection_manager
        assert len(manager.sync_queue) == 0

    @pytest.mark.asyncio
    async def test_sync_single_memory_delete(self):
        """Test deleting single memory."""
        collection_manager = AsyncMock()
        collection_manager.delete_memories.return_value = {
            "deleted_count": 1,
            "failed_count": 0,
            "errors": [],
        }

        config = ChromaConfig()
        manager = ChromaMemorySyncManager(collection_manager, config)

        session_id = uuid4()
        memory_id = uuid4()

        result = await manager.sync_single_memory(
            session_id,
            memory_id,
            [0.1] * 1536,
            "finding",
            "test",
            {},
            operation="delete",
        )

        assert result is True

    def test_get_queue_status(self):
        """Test queue status reporting."""
        collection_manager = Mock()
        config = ChromaConfig()
        manager = ChromaMemorySyncManager(collection_manager, config)

        status = manager.get_queue_status()
        assert status["queue_size"] == 0


class TestPerformanceMetrics:
    """Test performance metrics tracking."""

    def test_init(self):
        """Test metrics initialization."""
        metrics = PerformanceMetrics()
        assert metrics.success_count == 0
        assert metrics.failure_count == 0

    def test_record_success(self):
        """Test recording successful operation."""
        metrics = PerformanceMetrics()
        metrics.record_success(50.5)

        assert metrics.success_count == 1
        assert len(metrics.latencies) == 1

    def test_record_failure(self):
        """Test recording failed operation."""
        metrics = PerformanceMetrics()
        metrics.record_failure()

        assert metrics.failure_count == 1

    def test_get_p50_latency(self):
        """Test 50th percentile latency calculation."""
        metrics = PerformanceMetrics()
        for i in range(1, 11):
            metrics.record_success(i * 10.0)

        p50 = metrics.get_p50_latency()
        assert p50 is not None
        assert 40 <= p50 <= 60  # Should be around 50

    def test_get_error_rate(self):
        """Test error rate calculation."""
        metrics = PerformanceMetrics()
        metrics.record_success(10.0)
        metrics.record_success(20.0)
        metrics.record_failure()

        error_rate = metrics.get_error_rate()
        assert error_rate == 1 / 3

    def test_get_avg_latency(self):
        """Test average latency calculation."""
        metrics = PerformanceMetrics()
        metrics.record_success(10.0)
        metrics.record_success(20.0)
        metrics.record_success(30.0)

        avg = metrics.get_avg_latency()
        assert avg == 20.0


class TestSmartSearchRouter:
    """Test smart search router."""

    def test_init(self):
        """Test router initialization."""
        chroma_manager = Mock()
        pgvector_fallback = Mock()

        router = SmartSearchRouter(
            chroma_manager=chroma_manager,
            pgvector_fallback=pgvector_fallback,
        )

        assert router.chroma_manager == chroma_manager
        assert router.pgvector_fallback == pgvector_fallback

    def test_get_performance_stats(self):
        """Test performance stats generation."""
        router = SmartSearchRouter()

        stats = router.get_performance_stats()
        assert "chroma" in stats
        assert "pgvector" in stats
        assert "preferred_backend" in stats

    def test_metrics_initialization(self):
        """Test that metrics are properly initialized."""
        router = SmartSearchRouter()

        assert isinstance(router.chroma_metrics, PerformanceMetrics)
        assert isinstance(router.pgvector_metrics, PerformanceMetrics)
