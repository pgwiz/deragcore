"""Integration tests for Sprint 4: ChromaDB Hybrid Store."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from uuid import uuid4

from ragcore.modules.memory.hybrid.hybrid_store import HybridMemoryStore
from ragcore.modules.memory.chroma.client import ChromaClientManager, ChromaConfig
from ragcore.modules.memory.chroma.collection_manager import ChromaCollectionManager
from ragcore.modules.memory.chroma.sync_manager import ChromaMemorySyncManager
from ragcore.modules.memory.chroma.performance_router import SmartSearchRouter


class TestHybridMemoryStore:
    """Test hybrid memory store with ChromaDB integration."""

    def test_init_postgres_only_mode(self):
        """Test initialization in postgres_only mode."""
        store = HybridMemoryStore(deployment_mode="postgres_only")
        assert store.chroma_enabled is False
        assert store.chroma_primary is False

    def test_init_hybrid_mode(self):
        """Test initialization in hybrid mode."""
        store = HybridMemoryStore(deployment_mode="hybrid")
        assert store.chroma_enabled is True
        assert store.chroma_primary is False

    def test_init_chroma_primary_mode(self):
        """Test initialization in chroma_primary mode."""
        store = HybridMemoryStore(deployment_mode="chroma_primary")
        assert store.chroma_enabled is True
        assert store.chroma_primary is True

    def test_init_with_managers(self):
        """Test initialization with ChromaDB managers."""
        client_manager = Mock()
        collection_manager = Mock()
        sync_manager = Mock()
        router = Mock()

        store = HybridMemoryStore(
            chroma_client_manager=client_manager,
            collection_manager=collection_manager,
            sync_manager=sync_manager,
            smart_router=router,
        )

        assert store.chroma_client_manager == client_manager
        assert store.collection_manager == collection_manager
        assert store.sync_manager == sync_manager
        assert store.smart_router == router

    @pytest.mark.asyncio
    async def test_get_embedding_placeholder(self):
        """Test embedding retrieval (placeholder)."""
        store = HybridMemoryStore()
        embedding = await store._get_embedding("test text")
        assert embedding is None  # Placeholder

    @pytest.mark.asyncio
    async def test_get_chroma_health_disabled(self):
        """Test health check when manager not configured."""
        store = HybridMemoryStore()
        health = await store.get_chroma_health()
        assert health["status"] == "disabled"

    @pytest.mark.asyncio
    async def test_get_sync_status_disabled(self):
        """Test sync status when manager not configured."""
        store = HybridMemoryStore()
        status = await store.get_chroma_sync_status(uuid4())
        assert status["status"] == "disabled"

    @pytest.mark.asyncio
    async def test_get_search_performance_stats_no_router(self):
        """Test performance stats when router not configured."""
        store = HybridMemoryStore()
        stats = await store.get_search_performance_stats()
        assert "status" in stats
        assert stats["status"] == "no_router"

    @pytest.mark.asyncio
    async def test_search_semantic_no_router(self):
        """Test semantic search without router."""
        store = HybridMemoryStore()
        results = await store.search_semantic(
            session_id=uuid4(),
            query="test query",
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_full_sync_no_manager(self):
        """Test full sync without manager."""
        store = HybridMemoryStore()
        result = await store.sync_all_memories_to_chroma(uuid4())
        assert result["synced_count"] == 0


class TestHybridIntegration:
    """Integration tests for hybrid store components."""

    def test_config_to_client_manager(self):
        """Test creating client manager from config."""
        config = ChromaConfig(
            enabled=True,
            deployment_mode="hybrid",
            host="localhost",
            port=8000,
        )

        manager = ChromaClientManager(config)
        assert manager.config.deployment_mode == "hybrid"
        assert manager.is_healthy() is False  # Not initialized

    @pytest.mark.asyncio
    async def test_hybrid_store_with_sync_manager(self):
        """Test hybrid store integration with sync manager."""
        client_manager = Mock()
        collection_manager = AsyncMock()
        collection_manager.add_memories.return_value = {
            "success_count": 1,
            "failed_count": 0,
        }

        config = ChromaConfig()
        sync_manager = ChromaMemorySyncManager(collection_manager, config)

        store = HybridMemoryStore(
            chroma_client_manager=client_manager,
            collection_manager=collection_manager,
            sync_manager=sync_manager,
            deployment_mode="hybrid",
        )

        assert store.chroma_enabled is True
        sync_queue = store.sync_manager.get_queue_status()
        assert sync_queue["queue_size"] == 0

    def test_deployment_modes_incompatibility(self):
        """Test that modes are mutually exclusive."""
        # postgres_only mode with chroma_enabled should be False
        store1 = HybridMemoryStore(deployment_mode="postgres_only")
        assert store1.chroma_enabled is False

        # hybrid mode should enable ChromaDB
        store2 = HybridMemoryStore(deployment_mode="hybrid")
        assert store2.chroma_enabled is True

        # chroma_primary should enable and set primary flag
        store3 = HybridMemoryStore(deployment_mode="chroma_primary")
        assert store3.chroma_enabled is True
        assert store3.chroma_primary is True

    @pytest.mark.asyncio
    async def test_health_with_configured_manager(self):
        """Test health check with configured manager."""
        client_manager = AsyncMock()
        client_manager.health_check.return_value = {
            "status": "healthy",
            "latency_ms": 25.5,
        }
        client_manager.is_healthy.return_value = True

        store = HybridMemoryStore(
            chroma_client_manager=client_manager,
            deployment_mode="hybrid",
        )

        health = await store.get_chroma_health()
        assert health["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_sync_status_with_queue(self):
        """Test sync status with queue information."""
        collection_manager = Mock()
        config = ChromaConfig()
        sync_manager = ChromaMemorySyncManager(collection_manager, config)

        # Simulate some queued items
        sync_manager.sync_queue[uuid4()] = datetime.utcnow() - timedelta(
            seconds=5
        )  # Overdue
        sync_manager.sync_queue[uuid4()] = datetime.utcnow() + timedelta(
            seconds=60
        )  # Future

        store = HybridMemoryStore(
            sync_manager=sync_manager,
            deployment_mode="hybrid",
        )

        status = await store.get_chroma_sync_status(uuid4())
        assert status["queue_status"]["queue_size"] == 2
        assert status["queue_status"]["overdue_count"] == 1


class TestDeploymentModes:
    """Test deployment mode behavior."""

    def test_postgres_only_characteristics(self):
        """Test postgres_only mode characteristics."""
        store = HybridMemoryStore(deployment_mode="postgres_only")
        assert store.chroma_enabled is False
        assert store.deployment_mode == "postgres_only"

    def test_hybrid_characteristics(self):
        """Test hybrid mode characteristics."""
        store = HybridMemoryStore(deployment_mode="hybrid")
        assert store.chroma_enabled is True
        assert store.chroma_primary is False
        assert store.deployment_mode == "hybrid"

    def test_chroma_primary_characteristics(self):
        """Test chroma_primary mode characteristics."""
        store = HybridMemoryStore(deployment_mode="chroma_primary")
        assert store.chroma_enabled is True
        assert store.chroma_primary is True
        assert store.deployment_mode == "chroma_primary"

    def test_embedding_config(self):
        """Test embedding configuration."""
        store = HybridMemoryStore(
            embedding_provider="openai",
            embedding_model="text-embedding-3-large",
        )
        assert store.embedding_provider == "openai"
        assert store.embedding_model == "text-embedding-3-large"
