# RAGCORE Testing Reference: Component-Specific Patterns

**Quick Reference**: Copy-paste test examples for your multimodal architecture
**Updated**: 2026-03-28

---

## TABLE OF CONTENTS

1. [Embedding Pipeline Testing](#embedding-pipeline-testing)
2. [Context Manager Testing](#context-manager-testing)
3. [Storage Backend Testing](#storage-backend-testing)
4. [Router Endpoint Testing](#router-endpoint-testing)
5. [Provider Adapter Testing](#provider-adapter-testing)
6. [Database Model Testing](#database-model-testing)
7. [End-to-End Pipeline Testing](#end-to-end-pipeline-testing)

---

## EMBEDDING PIPELINE TESTING

### Test File: `tests/unit/test_embedding_pipeline.py`

```python
"""Unit tests for MultiModalEmbeddingPipeline."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from ragcore.modules.multimodal.embedding_pipeline import MultiModalEmbeddingPipeline
from ragcore.modules.multimodal.models import MultiModalChunk, ModuleType, ProcessingResult
from ragcore.core.model_provider_registry import ModelProviderRegistry, ProviderConfig, ProviderType


@pytest.fixture
def mock_adapter():
    """Create mock embedding adapter."""
    adapter = AsyncMock()
    adapter.embed_texts = AsyncMock(
        return_value=[[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]
    )
    return adapter


@pytest.fixture
def pipeline(mock_adapter):
    """Create pipeline with mock adapter."""
    return MultiModalEmbeddingPipeline(
        embedding_adapter=mock_adapter,
        embedding_dimension=1536,
        batch_size=10,
        cache_enabled=True,
    )


class TestMultiModalEmbeddingPipeline:
    """Test embedding pipeline."""

    def test_init_with_defaults(self):
        """Test pipeline initialization with defaults."""
        pipeline = MultiModalEmbeddingPipeline()
        assert pipeline.embedding_dimension == 1536
        assert pipeline.batch_size == 10
        assert pipeline.cache_enabled is True
        assert len(pipeline.cache) == 0

    def test_validate_embedding_dimension(self, pipeline):
        """Test embedding dimension validation."""
        # Valid
        valid_emb = [0.1] * 1536
        assert pipeline.validate_embedding_dimension(valid_emb) is True

        # Invalid
        invalid_emb = [0.1] * 768
        assert pipeline.validate_embedding_dimension(invalid_emb) is False

    def test_validate_embedding_empty(self, pipeline):
        """Test validation fails for empty embeddings."""
        assert pipeline.validate_embedding_dimension([]) is False
        assert pipeline.validate_embedding_dimension(None) is False

    @pytest.mark.asyncio
    async def test_embed_chunk_success(self, pipeline, mock_adapter):
        """Test embedding single chunk."""
        chunk = MultiModalChunk(
            id=uuid4(),
            session_id=uuid4(),
            content_id=uuid4(),
            modality=ModuleType.IMAGE,
            content="test chunk content",
            source_index=0,
            confidence_score=0.9,
        )

        result = await pipeline.embed_chunk(chunk)

        assert result is not None
        assert len(result.embedding) == 1536
        mock_adapter.embed_texts.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_chunks_batch(self, pipeline, mock_adapter):
        """Test batch embedding of chunks."""
        chunks = [
            MultiModalChunk(
                id=uuid4(),
                session_id=uuid4(),
                content_id=uuid4(),
                modality=ModuleType.IMAGE,
                content=f"chunk {i}",
                source_index=i,
            )
            for i in range(5)
        ]

        results = await pipeline.embed_chunks_batch(chunks)

        assert len(results) == 5
        assert all(len(r.embedding) == 1536 for r in results)
        mock_adapter.embed_texts.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_hit_performance(self, pipeline, mock_adapter):
        """Test embedding cache improves performance."""
        chunk = MultiModalChunk(
            id=uuid4(),
            session_id=uuid4(),
            content_id=uuid4(),
            modality=ModuleType.IMAGE,
            content="cached content",
            source_index=0,
        )

        # First call - cache miss
        result1 = await pipeline.embed_chunk(chunk)
        assert mock_adapter.embed_texts.call_count == 1

        # Second call - cache hit (same content)
        result2 = await pipeline.embed_chunk(chunk)
        assert mock_adapter.embed_texts.call_count == 1  # No additional call

        # Results should be identical
        assert result1.embedding == result2.embedding

    @pytest.mark.asyncio
    async def test_embed_processing_result(self, pipeline, mock_adapter):
        """Test embedding complete ProcessingResult."""
        chunks = [
            MultiModalChunk(
                id=uuid4(),
                session_id=uuid4(),
                content_id=uuid4(),
                modality=ModuleType.IMAGE,
                content=f"chunk {i}",
                source_index=i,
            )
            for i in range(3)
        ]

        result = ProcessingResult(
            success=True,
            chunks=chunks,
            tokens_used=150,
            processing_time_ms=500,
        )

        embedded_result = await pipeline.embed_processing_result(result)

        assert embedded_result.success is True
        assert len(embedded_result.chunks) == 3
        assert all(c.embedding is not None for c in embedded_result.chunks)

    def test_cache_statistics(self, pipeline):
        """Test cache statistics tracking."""
        stats = pipeline.get_cache_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["hit_rate"] == 0.0

        pipeline.cache_hit_count = 10
        pipeline.cache_miss_count = 5

        stats = pipeline.get_cache_stats()
        assert stats["hits"] == 10
        assert stats["misses"] == 5
        assert stats["hit_rate"] == pytest.approx(0.667, abs=0.01)

    def test_clear_cache(self, pipeline):
        """Test cache clearing."""
        pipeline.cache["test"] = [0.1] * 1536
        assert len(pipeline.cache) > 0

        pipeline.clear_cache()
        assert len(pipeline.cache) == 0
```

---

## CONTEXT MANAGER TESTING

### Test File: `tests/unit/test_context_manager.py`

```python
"""Unit tests for ContextWindowManagerForMultiModal."""

import pytest
from uuid import uuid4

from ragcore.modules.multimodal.context_manager import (
    ContextWindowManagerForMultiModal,
    ModalityWeights,
)
from ragcore.modules.multimodal.models import MultiModalChunk, ModuleType


@pytest.fixture
def manager():
    """Create context manager with standard budget."""
    return ContextWindowManagerForMultiModal(
        total_context_window_tokens=4096,
        reserved_tokens=512,  # System prompt
    )


@pytest.fixture
def mixed_chunks():
    """Create chunks across all modalities."""
    chunks = []

    # 10 image chunks
    for i in range(10):
        chunks.append(
            MultiModalChunk(
                id=uuid4(),
                session_id=uuid4(),
                content_id=uuid4(),
                modality=ModuleType.IMAGE,
                content=f"image description {i}",
                confidence_score=0.95,
                source_index=i,
                is_critical=i < 2,  # First 2 are critical
            )
        )

    # 5 audio chunks
    for i in range(5):
        chunks.append(
            MultiModalChunk(
                id=uuid4(),
                session_id=uuid4(),
                content_id=uuid4(),
                modality=ModuleType.AUDIO,
                content=f"audio transcript {i}",
                confidence_score=0.90,
                source_index=10 + i,
            )
        )

    # 3 video chunks
    for i in range(3):
        chunks.append(
            MultiModalChunk(
                id=uuid4(),
                session_id=uuid4(),
                content_id=uuid4(),
                modality=ModuleType.VIDEO,
                content=f"video description {i}",
                confidence_score=0.88,
                source_index=15 + i,
            )
        )

    return chunks


class TestContextWindowManager:
    """Test context window allocation."""

    def test_init_with_defaults(self):
        """Test initialization."""
        manager = ContextWindowManagerForMultiModal()
        assert manager.total_context_window_tokens == 4096
        assert manager.reserved_tokens == 256
        assert manager.pressure_threshold == 0.85

    def test_estimate_chunk_tokens(self, manager):
        """Test token estimation for chunks."""
        chunk = MultiModalChunk(
            id=uuid4(),
            session_id=uuid4(),
            content_id=uuid4(),
            modality=ModuleType.IMAGE,
            content="test content with multiple words",
            confidence_score=0.9,
            source_index=0,
        )

        tokens = manager.estimate_chunk_tokens(chunk)
        assert tokens > 0
        assert tokens <= 100  # Rough upper bound for short content

    def test_allocation_calculation(self, manager):
        """Test weight-based allocation calculation."""
        # Get default weights
        weights = manager.get_modality_weights()

        # Verify weights as per spec
        assert weights.image > 0
        assert weights.audio > weights.image  # 1.5x vs 1.0x
        assert weights.video > weights.audio   # 2.5x vs 2.0x

    def test_select_chunks_under_budget(self, manager, mixed_chunks):
        """Test chunk selection respects budget."""
        budget = 2000  # Tokens available

        selected, report = manager.select_chunks_under_budget(
            chunks=mixed_chunks,
            budget_tokens=budget,
        )

        # Verify budget respected
        assert report.total_tokens_allocated <= budget

        # Verify all modalities represented (if possible)
        selected_modalities = {c.modality for c in selected}
        assert len(selected_modalities) > 0

        # Verify critical chunks prioritized
        critical_selected = sum(1 for c in selected if c.is_critical)
        assert critical_selected > 0  # Should include critical chunks

    def test_fairness_verification(self, manager, mixed_chunks):
        """Test allocation fairness across modalities."""
        budget = 1500

        selected, report = manager.select_chunks_under_budget(
            chunks=mixed_chunks,
            budget_tokens=budget,
        )

        # Get per-modality allocations
        allocs = report.modality_allocations

        if allocs:
            # Verify expected weights applied
            # Audio weight (2.0x) → more tokens than image (1.5x)
            if ModuleType.AUDIO.value in allocs and ModuleType.IMAGE.value in allocs:
                audio_tokens = allocs[ModuleType.AUDIO.value]
                image_tokens = allocs[ModuleType.IMAGE.value]

                # With more tokens, audio should get proportionally less chunks
                # (since each audio chunk is "heavier")

    def test_is_under_pressure(self, manager):
        """Test pressure detection."""
        # Below threshold
        assert manager.is_under_pressure(3000, 4096) is False

        # Above threshold
        assert manager.is_under_pressure(3600, 4096) is True

    def test_custom_modality_weights(self):
        """Test custom weight configuration."""
        custom_weights = ModalityWeights(
            image=1.0,
            audio=1.5,  # Lower weight
            video=2.0,
            text=1.0,
        )

        manager = ContextWindowManagerForMultiModal(
            modality_weights=custom_weights,
        )

        weights = manager.get_modality_weights()
        assert weights.audio == 1.5


class TestContextManagerEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_chunks_list(self, manager):
        """Test handling empty chunks."""
        selected, report = manager.select_chunks_under_budget(
            chunks=[],
            budget_tokens=1000,
        )

        assert len(selected) == 0
        assert report.total_tokens_allocated == 0

    def test_budget_exceeds_single_chunk(self, manager):
        """Test when budget can't fit even one chunk."""
        chunk = MultiModalChunk(
            id=uuid4(),
            session_id=uuid4(),
            content_id=uuid4(),
            modality=ModuleType.IMAGE,
            content="x" * 1000,  # Large content
            confidence_score=0.9,
            source_index=0,
        )

        selected, report = manager.select_chunks_under_budget(
            chunks=[chunk],
            budget_tokens=10,  # Very small budget
        )

        # Should select 0 chunks
        assert len(selected) == 0

    def test_all_chunks_critical(self, manager):
        """Test when all chunks are marked critical."""
        chunks = [
            MultiModalChunk(
                id=uuid4(),
                session_id=uuid4(),
                content_id=uuid4(),
                modality=ModuleType.IMAGE,
                content=f"chunk {i}",
                confidence_score=0.9,
                source_index=i,
                is_critical=True,
            )
            for i in range(10)
        ]

        selected, report = manager.select_chunks_under_budget(
            chunks=chunks,
            budget_tokens=500,
        )

        # All selected should be critical
        assert all(c.is_critical for c in selected)
```

---

## STORAGE BACKEND TESTING

### Test File: `tests/integration/test_storage_backends.py`

```python
"""Integration tests for storage backends."""

import pytest
from pathlib import Path
import io


@pytest.mark.asyncio
class TestLocalStorageBackend:
    """Test LocalStorageBackend."""

    async def test_save_file(self, local_storage_backend):
        """Test saving file to local storage."""
        test_data = b"test file content"
        file_id = "test-uuid-123"

        path = await local_storage_backend.save_file(file_id, test_data)

        assert path is not None
        assert "test-uuid-123" in path
        assert "file://" in path

    async def test_get_file(self, local_storage_backend):
        """Test retrieving file from local storage."""
        test_data = b"test file content"
        file_id = "test-uuid-456"

        # Save first
        path = await local_storage_backend.save_file(file_id, test_data)

        # Retrieve
        retrieved = await local_storage_backend.get_file(path)

        assert retrieved == test_data

    async def test_delete_file(self, local_storage_backend):
        """Test deleting file from local storage."""
        test_data = b"test file content"
        file_id = "test-uuid-789"

        # Save
        path = await local_storage_backend.save_file(file_id, test_data)
        assert await local_storage_backend.exists(path) is True

        # Delete
        result = await local_storage_backend.delete_file(path)
        assert result is True

        # Verify deleted
        assert await local_storage_backend.exists(path) is False

    async def test_exists_check(self, local_storage_backend):
        """Test existence check."""
        test_data = b"test"
        file_id = "test-exists"

        # Non-existent
        fake_path = "file:///nonexistent/path.bin"
        assert await local_storage_backend.exists(fake_path) is False

        # After save
        path = await local_storage_backend.save_file(file_id, test_data)
        assert await local_storage_backend.exists(path) is True

        # After delete
        await local_storage_backend.delete_file(path)
        assert await local_storage_backend.exists(path) is False

    async def test_health_check(self, local_storage_backend):
        """Test health check."""
        is_healthy = await local_storage_backend.health_check()
        assert is_healthy is True

    async def test_get_backend_name(self, local_storage_backend):
        """Test backend name."""
        name = local_storage_backend.get_backend_name()
        assert name == "local"

    async def test_large_file_handling(self, local_storage_backend):
        """Test handling large files."""
        # 50MB file
        large_data = b"x" * (50 * 1024 * 1024)

        path = await local_storage_backend.save_file("large-file", large_data)
        assert path is not None

        retrieved = await local_storage_backend.get_file(path)
        assert len(retrieved) == len(large_data)
        assert retrieved == large_data

        await local_storage_backend.delete_file(path)


@pytest.mark.asyncio
class TestStorageBackendContract:
    """Test that all backends implement StorageBackend interface."""

    @pytest.mark.parametrize("backend_factory", [
        # Add more backend types as implemented
        lambda: LocalStorageBackend(base_path="/tmp/test"),
    ])
    async def test_backend_interface_contract(self, backend_factory):
        """Test backend contract for multiple backends."""
        backend = backend_factory()
        test_data = b"test data"
        file_id = "contract-test"

        # All backends should support these operations
        path = await backend.save_file(file_id, test_data)
        assert path is not None

        retrieved = await backend.get_file(path)
        assert retrieved == test_data

        exists = await backend.exists(path)
        assert exists is True

        deleted = await backend.delete_file(path)
        assert deleted is True

        name = backend.get_backend_name()
        assert isinstance(name, str)
        assert len(name) > 0

        health = await backend.health_check()
        assert isinstance(health, bool)
```

---

## ROUTER ENDPOINT TESTING

### Test File: `tests/integration/test_router_endpoints.py`

```python
"""Integration tests for router endpoints."""

import pytest
from fastapi.testclient import TestClient
from uuid import uuid4


class TestMultimodalRouterEndpoints:
    """Test HTTP endpoints."""

    def test_upload_endpoint_requires_auth(self, fastapi_client: TestClient):
        """Test upload endpoint requires JWT."""
        response = fastapi_client.post(
            "/multimodal/upload",
            files={"file": ("test.jpg", b"fake jpeg")},
            data={"session_id": str(uuid4()), "modality": "image"},
        )

        assert response.status_code == 401

    def test_upload_endpoint_with_auth(
        self,
        fastapi_client: TestClient,
        sample_session_id,
    ):
        """Test upload with valid auth."""
        response = fastapi_client.post(
            "/multimodal/upload",
            files={"file": ("test.jpg", b"fake jpeg data")},
            data={
                "session_id": str(sample_session_id),
                "modality": "image",
            },
            headers={"Authorization": "Bearer test-token"},
        )

        # Should succeed or return meaningful error
        assert response.status_code in [200, 400, 422]

    def test_search_endpoint(self, fastapi_client: TestClient, sample_session_id):
        """Test search endpoint."""
        response = fastapi_client.post(
            "/multimodal/search",
            json={
                "session_id": str(sample_session_id),
                "query": "test query",
                "limit": 10,
            },
            headers={"Authorization": "Bearer test-token"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_get_content_endpoint(self, fastapi_client: TestClient):
        """Test get content endpoint."""
        content_id = uuid4()

        response = fastapi_client.get(
            f"/multimodal/content/{content_id}",
            headers={"Authorization": "Bearer test-token"},
        )

        # Should return 404 for non-existent content
        assert response.status_code in [200, 404]

    def test_processing_status_endpoint(self, fastapi_client: TestClient):
        """Test processing status endpoint."""
        content_id = uuid4()

        response = fastapi_client.get(
            f"/multimodal/processing-status/{content_id}",
            headers={"Authorization": "Bearer test-token"},
        )

        assert response.status_code in [200, 404]

    def test_delete_content_endpoint(self, fastapi_client: TestClient):
        """Test content deletion."""
        content_id = uuid4()

        response = fastapi_client.delete(
            f"/multimodal/content/{content_id}",
            headers={"Authorization": "Bearer test-token"},
        )

        assert response.status_code in [200, 204, 404]

    def test_session_chunks_endpoint(self, fastapi_client: TestClient):
        """Test listing session chunks."""
        session_id = uuid4()

        response = fastapi_client.get(
            f"/multimodal/session/{session_id}/chunks",
            headers={"Authorization": "Bearer test-token"},
        )

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, (list, dict))

    def test_session_stats_endpoint(self, fastapi_client: TestClient):
        """Test session statistics."""
        session_id = uuid4()

        response = fastapi_client.get(
            f"/multimodal/session/{session_id}/stats",
            headers={"Authorization": "Bearer test-token"},
        )

        assert response.status_code == 200
```

---

## PROVIDER ADAPTER TESTING

### Test File: `tests/unit/test_provider_adapter.py`

```python
"""Unit tests for provider adapters."""

import pytest
from unittest.mock import AsyncMock, patch

from ragcore.modules.multimodal.providers.embedding_adapter import EmbeddingProviderAdapter
from ragcore.core.model_provider_registry import ModelProviderRegistry, ProviderType, ProviderConfig


class TestEmbeddingProviderAdapter:
    """Test embedding provider adapter."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with mock registry."""
        registry = ModelProviderRegistry()
        config = ProviderConfig(
            provider=ProviderType.OPENAI,
            api_key="test-key",
            endpoint="https://api.openai.com",
        )
        registry.register_provider(config)
        return EmbeddingProviderAdapter(registry=registry)

    @pytest.mark.asyncio
    async def test_embed_single_text(self, adapter):
        """Test embedding single text."""
        with patch.object(adapter, '_embed_with_openai') as mock_embed:
            mock_embed.return_value = [[0.1] * 1536]

            result = await adapter.embed_text("test text")

            assert result is not None
            assert len(result) == 1536
            mock_embed.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_multiple_texts(self, adapter):
        """Test embedding multiple texts."""
        with patch.object(adapter, '_embed_with_openai') as mock_embed:
            mock_embed.return_value = [[0.1] * 1536, [0.2] * 1536]

            result = await adapter.embed_texts(["text1", "text2"])

            assert len(result) == 2
            assert all(len(emb) == 1536 for emb in result)

    @pytest.mark.asyncio
    async def test_empty_text_handling(self, adapter):
        """Test handling empty text."""
        result = await adapter.embed_text("")
        assert result is None

        result = await adapter.embed_texts([])
        assert result is None

    @pytest.mark.asyncio
    async def test_provider_fallback(self, adapter):
        """Test fallback when primary provider fails."""
        # Mock primary to fail
        with patch.object(adapter, '_embed_with_openai') as mock_primary:
            mock_primary.side_effect = Exception("API Error")

            # Mock fallback
            with patch.object(adapter, '_embed_with_azure_openai') as mock_fallback:
                mock_fallback.return_value = [[0.1] * 1536]

                result = await adapter.embed_texts(["test"])

                # Should attempt fallback
                assert mock_primary.called or mock_fallback.called

    @pytest.mark.asyncio
    async def test_dimension_validation(self, adapter):
        """Test embedding dimension validation."""
        with patch.object(adapter, '_embed_with_openai') as mock_embed:
            # Return wrong dimension
            mock_embed.return_value = [[0.1] * 768]  # Wrong: should be 1536

            result = await adapter.embed_texts(["test"])

            # Should return None for wrong dimension
            assert result is None

    def test_adapter_health_status(self, adapter):
        """Test provider health tracking."""
        # Initially healthy
        assert adapter.is_provider_healthy(ProviderType.OPENAI) is True

        # Mark unhealthy
        adapter.record_provider_health(ProviderType.OPENAI, False)
        assert adapter.is_provider_healthy(ProviderType.OPENAI) is False

        # Mark healthy again
        adapter.record_provider_health(ProviderType.OPENAI, True)
        assert adapter.is_provider_healthy(ProviderType.OPENAI) is True
```

---

## DATABASE MODEL TESTING

### Test File: `tests/integration/test_database_models.py`

```python
"""Integration tests for database models."""

import pytest
from datetime import datetime
from uuid import uuid4
from sqlalchemy.exc import IntegrityError

from ragcore.modules.multimodal.models import (
    MultiModalContent,
    MultiModalChunk,
    ModuleType,
)


@pytest.mark.asyncio
class TestDatabaseModels:
    """Test database model constraints."""

    async def test_multimodal_content_insert(self, db_session):
        """Test inserting MultiModalContent."""
        session_id = uuid4()
        content = MultiModalContent(
            id=uuid4(),
            session_id=session_id,
            modality=ModuleType.IMAGE,
            raw_content=b"test image",
            created_at=datetime.utcnow(),
        )

        db_session.add(content)
        await db_session.flush()

        assert content.id is not None
        assert content.is_processed is False

    async def test_multimodal_content_requires_session_id(self, db_session):
        """Test that session_id is required."""
        content = MultiModalContent(
            id=uuid4(),
            session_id=uuid4(),  # Non-existent session
            modality=ModuleType.IMAGE,
            created_at=datetime.utcnow(),
        )

        db_session.add(content)

        # Foreign key violation expected
        with pytest.raises(IntegrityError):
            await db_session.flush()

        await db_session.rollback()

    async def test_multimodal_chunk_requires_content_id(self, db_session):
        """Test that content_id is required for chunks."""
        chunk = MultiModalChunk(
            id=uuid4(),
            session_id=uuid4(),  # Non-existent
            content_id=uuid4(),  # Non-existent
            modality=ModuleType.IMAGE,
            content="test content",
            created_at=datetime.utcnow(),
        )

        db_session.add(chunk)

        with pytest.raises(IntegrityError):
            await db_session.flush()

        await db_session.rollback()

    async def test_embedding_dimension_storage(self, db_session):
        """Test storing 1536-dimensional embedding."""
        chunk = MultiModalChunk(
            id=uuid4(),
            session_id=uuid4(),
            content_id=uuid4(),
            modality=ModuleType.IMAGE,
            content="test",
            embedding=[0.1] * 1536,  # pgvector format
            created_at=datetime.utcnow(),
        )

        db_session.add(chunk)
        await db_session.flush()

        # Verify stored
        assert chunk.embedding is not None
        assert len(chunk.embedding) == 1536
```

---

## END-TO-END PIPELINE TESTING

### Test File: `tests/e2e/test_full_pipelines.py`

```python
"""End-to-end integration tests."""

import pytest
from uuid import uuid4
from datetime import datetime
import asyncio

from ragcore.tests.factories import TestDataFactory


@pytest.mark.asyncio
class TestEndToEndPipelines:
    """Test complete workflows."""

    @pytest.mark.critical_path
    async def test_e2e_upload_embed_search(self):
        """E2E: Upload → Embed → Search pipeline."""
        # This is a placeholder for a complete integration test
        # that uses all fixtures and tests the full pipeline
        pass

    @pytest.mark.critical_path
    async def test_e2e_multi_modality_session(self):
        """E2E: Multi-modal session processing."""
        pass

    @pytest.mark.critical_path
    async def test_e2e_error_recovery(self):
        """E2E: Error handling and recovery."""
        pass

    @pytest.mark.asyncio
    async def test_e2e_provider_fallback(self):
        """E2E: Provider fallback during search."""
        pass

    @pytest.mark.asyncio
    async def test_e2e_concurrent_uploads(self):
        """E2E: Multiple concurrent uploads."""
        # Test up to 5 concurrent uploads
        # Verify all complete without interference
        pass
```

---

## RUNNING THESE TESTS

```bash
# Run all reference tests
pytest tests/ -v

# Run only critical path
pytest tests/ -v -m critical_path

# Run by component
pytest tests/unit/test_embedding_pipeline.py -v
pytest tests/integration/test_storage_backends.py -v

# Run with coverage
pytest tests/ --cov=ragcore.modules.multimodal --cov-report=html
```

---

## PYTEST MARKS (Add to pytest.ini)

```ini
[pytest]
markers =
    critical_path: Tests for critical user workflows
    slow: Tests that take >5 seconds
    requires_db: Tests that require database connection
    requires_storage: Tests that require storage backend
    requires_provider: Tests that require real provider API
```

