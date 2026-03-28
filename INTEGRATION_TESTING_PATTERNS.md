# Integration Testing Patterns for Multimodal RAG Systems

**Research Date**: 2026-03-28
**Scope**: RAGCORE architecture with PostgreSQL+pgvector, ChromaDB, S3/Azure storage
**Target Audience**: Phase 0 Critical Fixes & Phase 5 Sprint 6+ developers

---

## Executive Summary

Multimodal RAG systems require sophisticated integration testing to handle:
- **5 processing paths** (image, audio, video, text, mixed)
- **3 storage backends** (local, S3, Azure Blob)
- **6 embedding providers** (OpenAI, Azure, Anthropic, Vertex, etc.)
- **Dual database writes** (PostgreSQL + ChromaDB sync)
- **Large file handling** without OOM
- **Provider fallbacks** and error recovery

This document provides concrete patterns, code examples, and metrics for RAGCORE.

---

## PART 1: TEST PYRAMID STRUCTURE

### Recommended Distribution for Multimodal RAG

```
                    /\
                   /  \
                  /    \
                 / E2E  \           5-10% (20-50 tests)
                /   &    \          - Full pipeline (upload→embed→search)
               /  Load   /          - Multi-modality interaction
              /          /           - Error recovery
             /          /            - Provider fallback
            /          /
           /__________/
          /          \
         /    API     \              25-30% (80-120 tests)
        /  Integration /             - Router endpoints
       /             /               - Database constraints
      /             /                - Cross-layer contracts
     /             /
    /____________/
   /            \
  /   Unit      \                   60-70% (200-350 tests)
 /   Tests      \                   - Individual processors
/________________\                  - Adapter implementations
                                    - Utility functions
                                    - Business logic

Total: ~400-500 tests
Coverage Target: 85%+ (excluding mocks)
```

### RAGCORE Current State vs. Target

**Current**: 70 tests (mostly unit/low-level)
- ✅ 32 provider adapter tests
- ✅ 26 embedding pipeline tests
- ✅ 12 model/processor tests

**Target Next**: +150 integration tests
- 40-50 router/API tests
- 50-70 storage backend tests
- 40-60 E2E pipeline tests

---

## PART 2: CRITICAL PATH TESTS FOR RAGCORE

### Minimum Viable Coverage (Must Have)

These 15-20 tests represent the critical path. **All must pass before merge to main**.

#### 1. **Happy Path: Upload → Store → Embed → Search** (3 tests)

```
Test Case 1: Image Upload → Embed → Search
  - Upload 100KB PNG image
  - Verify multimodal_content record created
  - Verify image extracted to multimodal_chunks (confidence > 0.8)
  - Verify pgvector embedding stored (1536-dim)
  - Search with similar query
  - Verify results ranked by similarity_score

Test Case 2: Audio Upload → Transcribe → Embed
  - Upload 30-second MP3 audio
  - Verify audio transcribed to chunks
  - Verify chunks embedded
  - Verify processing_time_ms logged
  - Verify confidence_score reflects transcription quality

Test Case 3: Mixed Session (Image + Audio)
  - Upload image and audio to same session
  - Verify both processed independently
  - Cross-modal search returns top results from both
  - Verify context_manager allocates budget fairly
```

#### 2. **Storage Backend Consistency** (2 tests)

```
Test Case 4: Local Storage Backend
  - Configure storage to local filesystem
  - Upload file → verify saved to disk
  - Retrieve → verify matches original
  - Delete → verify removed from disk
  - health_check() returns True

Test Case 5: S3 Storage Backend (with LocalStack)
  - Start LocalStack S3
  - Configure credentials
  - Upload 5MB file → verify stored in bucket
  - Retrieve → verify matches original
  - Delete → verify removed
  - health_check() can reach S3 endpoint
```

#### 3. **Database Constraints** (2 tests)

```
Test Case 6: Foreign Key Integrity
  - Create content with session_id=<invalid>
  - Verify INSERT fails with foreign key error
  - Verify error message logged
  - Verify transaction rolled back

Test Case 7: Cascade Delete
  - Create session with 5 contents, 20 chunks
  - Delete session
  - Verify all multimodal_content, chunks, logs deleted
  - Verify referential integrity maintained
```

#### 4. **Embedding Provider Fallback** (2 tests)

```
Test Case 8: Primary Provider Fails → Fallback Succeeds
  - Mock OpenAI to return 500 error
  - Call embed_texts()
  - Verify EmbeddingProviderAdapter retries with Azure
  - Verify Azure returns valid 1536-dim embedding
  - Verify provider health marked False for OpenAI

Test Case 9: All Providers Unavailable
  - Mock all providers to fail
  - Call embed_texts()
  - Verify returns None gracefully
  - Verify error logged with all attempted providers
  - Verify status endpoint reflects unhealthy state
```

#### 5. **Large File Handling** (2 tests)

```
Test Case 10: File Streaming (No OOM)
  - Upload 500MB video file
  - Monitor process memory during upload
  - Verify file stored (not kept in memory)
  - Verify processing doesn't exceed 500MB delta memory
  - Verify chunks extracted without OOM

Test Case 11: Batch Processing Backpressure
  - Queue 100 images simultaneously
  - Verify batch processor uses configurable batch_size (default 10)
  - Verify batches processed sequentially
  - Verify queue status endpoint shows progress
```

#### 6. **Error Recovery** (2 tests)

```
Test Case 12: Processing Failure → Retry
  - Upload image, mock ImageProcessor to fail on first try
  - Verify processing_error logged in multimodal_content
  - Call POST /multimodal/session/{id}/process
  - Verify retry succeeds
  - Verify is_processed=true, processing_error cleared

Test Case 13: Partial Success (Multi-Modal)
  - Upload session with 3 images, 2 audio files
  - Mock one image processor to fail
  - Verify 2 images + 2 audio processed successfully
  - Verify 1 image logged as failed
  - Verify search returns results from successful modalities
```

#### 7. **ChromaDB Sync** (2 tests)

```
Test Case 14: Sync Multimodal Chunks → ChromaDB
  - Upload image, process to chunks
  - Trigger sync_all_memories_to_chroma()
  - Verify chunks exist in ChromaDB collection
  - Verify embeddings match PostgreSQL
  - Verify metadata preserved

Test Case 15: Dual-Write Consistency
  - Upload content
  - Query PostgreSQL AND ChromaDB
  - Verify same chunk count
  - Verify same top-5 search results from both
  - Verify similarity scores within 0.01 tolerance
```

#### 8. **Router Authentication & Authorization** (2 tests)

```
Test Case 16: Invalid JWT Token
  - Call POST /multimodal/upload without auth header
  - Verify 401 Unauthorized response
  - Verify error message "Missing or invalid JWT"

Test Case 17: Valid JWT → Success
  - Include valid JWT in Authorization header
  - Call POST /multimodal/upload
  - Verify 200 response
  - Verify current_api_key_id injected into request context
```

**Total Critical Path: 17 tests, ~3-5 hours implementation**

---

## PART 3: TESTING FRAMEWORK SETUP

### 3.1 Async Testing Patterns with pytest-asyncio

**Current Setup** (from pyproject.toml):
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"  # ✅ Already enabled
testpaths = ["tests"]
addopts = "-v --tb=short"
```

#### Pattern A: Async Fixtures

```python
import pytest
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

@pytest.fixture(scope="session")
async def db_engine():
    """Create async database engine for all tests."""
    engine = create_async_engine(
        "postgresql+asyncpg://ragcore:ragcore@localhost/ragcore_test",
        echo=False,
        pool_pre_ping=True,
    )
    async with engine.begin() as conn:
        # Create tables
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest.fixture
async def async_session(db_engine):
    """Create new session for each test."""
    async_session_local = sessionmaker(
        db_engine, class_=AsyncSession, expire_on_delete=False
    )
    async with async_session_local() as session:
        yield session
        await session.rollback()
```

#### Pattern B: Async Test Functions

```python
@pytest.mark.asyncio
async def test_embed_text_with_mock_provider():
    """Test embedding generation with mocked provider."""
    # Arrange
    adapter = EmbeddingProviderAdapter(embedding_dimension=1536)
    mock_provider = AsyncMock()
    adapter.registry.providers[ProviderType.OPENAI] = mock_provider

    # Mock embeddings
    mock_embeddings = [[0.1] * 1536 for _ in range(3)]
    mock_provider.embed.return_value = mock_embeddings

    # Act
    result = await adapter.embed_texts(["text1", "text2", "text3"])

    # Assert
    assert result == mock_embeddings
    mock_provider.embed.assert_called_once()
```

#### Pattern C: Async Context Managers in Tests

```python
@pytest.mark.asyncio
async def test_storage_backend_upload_download():
    """Test S3 backend save/retrieve cycle."""
    # Use context manager to ensure cleanup
    async with aiofiles.open("test_image.bin", "rb") as f:
        content = await f.read()

    backend = S3StorageBackend(
        bucket="ragcore-test",
        region="us-east-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )

    # Upload
    path = await backend.save_file("test-uuid", content)

    try:
        # Verify
        retrieved = await backend.get_file(path)
        assert retrieved == content
    finally:
        # Cleanup
        await backend.delete_file(path)
```

### 3.2 Pytest Fixtures for Multimodal Testing

Create `conftest.py` at `/e/Backup/pgwiz/rag/tests/conftest.py`:

```python
"""Shared pytest fixtures for integration tests."""

import pytest
import asyncio
from datetime import datetime
from uuid import UUID, uuid4
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ragcore.db.models import Base
from ragcore.modules.multimodal.models import MultiModalContent, ModuleType
from ragcore.modules.multimodal.storage.local import LocalStorageBackend
from ragcore.modules.multimodal.embedding_pipeline import MultiModalEmbeddingPipeline
from ragcore.core.model_provider_registry import ModelProviderRegistry


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create session-scoped event loop."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_db_engine():
    """Create in-memory SQLite engine for tests (fast, isolated)."""
    # Use SQLite for speed in unit/integration tests
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()


@pytest.fixture
async def db_session(test_db_engine):
    """Create new session for each test with rollback."""
    async_session_local = sessionmaker(
        test_db_engine,
        class_=AsyncSession,
        expire_on_delete=False,
    )

    async with async_session_local() as session:
        yield session
        await session.rollback()


# ============================================================================
# Storage Fixtures
# ============================================================================

@pytest.fixture
def temp_storage_dir(tmp_path):
    """Create temporary directory for local storage backend."""
    return tmp_path / "storage"


@pytest.fixture
def local_storage_backend(temp_storage_dir):
    """Create LocalStorageBackend for testing."""
    return LocalStorageBackend(base_path=str(temp_storage_dir))


# ============================================================================
# Mock Embedding Provider Fixtures
# ============================================================================

@pytest.fixture
def mock_embedding_provider():
    """Create mock embedding provider that returns deterministic vectors."""
    mock = AsyncMock()

    async def mock_embed(texts, **kwargs):
        """Return deterministic embeddings (1536-dim)."""
        # Deterministic: use hash of text for reproducibility
        embeddings = []
        for text in texts:
            # Simple deterministic vector: hash % 1536 repeated
            seed = hash(text) % 1000
            vec = [(seed + i) % 100 / 100.0 for i in range(1536)]
            embeddings.append(vec)
        return embeddings

    mock.embed.side_effect = mock_embed
    return mock


@pytest.fixture
def embedding_pipeline_with_mock(mock_embedding_provider):
    """Create embedding pipeline with mocked provider."""
    registry = ModelProviderRegistry()

    # Register mock provider
    from ragcore.core.model_provider_registry import ProviderType, ProviderConfig
    from ragcore.modules.multimodal.providers.embedding_adapter import EmbeddingProviderAdapter

    config = ProviderConfig(
        provider=ProviderType.OPENAI,
        api_key="test-key",
        endpoint="https://api.openai.com",
    )
    registry.register_provider(config)

    adapter = EmbeddingProviderAdapter(registry=registry)
    adapter.registry.providers[ProviderType.OPENAI] = mock_embedding_provider

    pipeline = MultiModalEmbeddingPipeline(
        embedding_adapter=adapter,
        embedding_dimension=1536,
        batch_size=10,
        cache_enabled=True,
    )

    return pipeline


# ============================================================================
# Model/Data Fixtures
# ============================================================================

@pytest.fixture
def sample_session_id():
    """Generate consistent session ID for tests."""
    return uuid4()


@pytest.fixture
def sample_multimodal_content(sample_session_id):
    """Create sample MultiModalContent."""
    return MultiModalContent(
        id=uuid4(),
        session_id=sample_session_id,
        modality=ModuleType.IMAGE,
        raw_content=b"fake image data",
        is_processed=False,
        storage_path=None,
        created_at=datetime.utcnow(),
    )


# ============================================================================
# FastAPI Test Client Fixtures
# ============================================================================

@pytest.fixture
def fastapi_client():
    """Create FastAPI TestClient for HTTP testing."""
    from fastapi.testclient import TestClient
    from ragcore.main import app

    # Override database dependency with test session
    # (Implementation depends on your app structure)

    return TestClient(app)


# ============================================================================
# Provider Mock Fixtures
# ============================================================================

@pytest.fixture
def mock_openai_provider():
    """Mock OpenAI provider with realistic responses."""
    mock = MagicMock()
    mock.embed.return_value = [[0.1] * 1536]  # 1536-dim embedding
    return mock


@pytest.fixture
def mock_azure_provider():
    """Mock Azure OpenAI provider."""
    mock = MagicMock()
    mock.embed.return_value = [[0.2] * 1536]
    return mock


@pytest.fixture
def mock_failing_provider():
    """Mock provider that fails."""
    mock = AsyncMock()
    mock.embed.side_effect = Exception("Provider unavailable")
    return mock
```

### 3.3 Test Data Generation Patterns

#### Pattern A: Factory for Minimal Test Files

```python
import io
from pathlib import Path

class TestDataFactory:
    """Generate minimal but realistic test files."""

    @staticmethod
    def create_test_image(size_kb: int = 5) -> bytes:
        """Create minimal PNG image."""
        # Use PIL if available, otherwise create fake PNG header
        try:
            from PIL import Image
            img = Image.new('RGB', (10, 10), color='red')
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')

            # Pad to target size
            data = buffer.getvalue()
            if len(data) < size_kb * 1024:
                data += b'\x00' * (size_kb * 1024 - len(data))
            return data[:size_kb * 1024]
        except ImportError:
            # Fallback: fake PNG (89504E47 = PNG signature)
            png_header = bytes.fromhex('89504E47')
            return png_header + b'\x00' * (size_kb * 1024 - len(png_header))

    @staticmethod
    def create_test_audio(size_kb: int = 10, duration_sec: int = 5) -> bytes:
        """Create minimal MP3 audio."""
        try:
            from pydub import AudioSegment
            silence = AudioSegment.silent(duration=duration_sec * 1000)
            buffer = io.BytesIO()
            silence.export(buffer, format="mp3")

            data = buffer.getvalue()
            if len(data) < size_kb * 1024:
                data += b'\x00' * (size_kb * 1024 - len(data))
            return data[:size_kb * 1024]
        except ImportError:
            # Fallback: fake MP3 (ID3 header)
            mp3_header = bytes.fromhex('FFE3')
            return mp3_header + b'\x00' * (size_kb * 1024 - len(mp3_header))

    @staticmethod
    def create_test_video(size_kb: int = 100, duration_sec: int = 3) -> bytes:
        """Create minimal MP4 video."""
        try:
            import cv2
            import numpy as np

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('temp_test.mp4', fourcc, 1.0, (10, 10))

            for _ in range(duration_sec):
                frame = np.zeros((10, 10, 3), dtype=np.uint8)
                out.write(frame)
            out.release()

            with open('temp_test.mp4', 'rb') as f:
                data = f.read()

            Path('temp_test.mp4').unlink()

            if len(data) < size_kb * 1024:
                data += b'\x00' * (size_kb * 1024 - len(data))
            return data[:size_kb * 1024]
        except (ImportError, OSError):
            # Fallback: fake MP4 (ftypisom header)
            mp4_header = bytes.fromhex('0000002066747970')
            return mp4_header + b'\x00' * (size_kb * 1024 - len(mp4_header))


# Usage in tests:
@pytest.mark.asyncio
async def test_image_upload():
    image_bytes = TestDataFactory.create_test_image(5)
    # ... use for upload test
```

---

## PART 4: MOCKING STRATEGIES

### 4.1 When to Mock vs. Use Real Services

| Component | Unit Tests | Integration Tests | E2E Tests |
|-----------|-----------|------------------|-----------|
| **Embedding Providers** | Always mock (deterministic) | Mock primary + test fallback | Real (staging key) |
| **Storage Backends** | Always mock or use LocalStack | Use LocalStack for S3, local FS | Real cloud buckets |
| **Database** | Mock queries | Use test DB (SQLite in-memory) | Real PostgreSQL |
| **HTTP Clients** | Mock aiohttp responses | Mock external API calls | Partial (test webhooks) |
| **File I/O** | Use tmp_path fixture | Use tmp_path fixture | Real files |

### 4.2 Async HTTP Mocking Pattern

```python
from unittest.mock import AsyncMock, patch
import aiohttp

@pytest.mark.asyncio
async def test_openai_embedding_with_mock():
    """Mock aiohttp responses for OpenAI API."""
    mock_response = {
        "data": [
            {"embedding": [0.1] * 1536},
            {"embedding": [0.2] * 1536},
        ],
        "model": "text-embedding-3-large",
    }

    # Patch aiohttp.ClientSession
    with patch('aiohttp.ClientSession.post') as mock_post:
        # Configure mock response
        mock_response_obj = AsyncMock()
        mock_response_obj.status = 200
        mock_response_obj.json = AsyncMock(return_value=mock_response)
        mock_post.return_value.__aenter__.return_value = mock_response_obj

        # Test code
        adapter = EmbeddingProviderAdapter()
        result = await adapter.embed_texts(["text1", "text2"])

        # Assertions
        assert len(result) == 2
        assert all(len(emb) == 1536 for emb in result)
        mock_post.assert_called_once()
```

### 4.3 Database Mocking Pattern (when integration DB unavailable)

```python
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_multimodal_content_insert_with_mock_db():
    """Mock database session for unit testing."""
    mock_session = AsyncMock()

    # Mock the add/flush sequence
    mock_session.add = MagicMock()
    mock_session.flush = AsyncMock()
    mock_session.commit = AsyncMock()

    content = MultiModalContent(
        id=uuid4(),
        session_id=uuid4(),
        modality=ModuleType.IMAGE,
        created_at=datetime.utcnow(),
    )

    mock_session.add(content)
    await mock_session.flush()
    await mock_session.commit()

    mock_session.add.assert_called_once_with(content)
```

### 4.4 Storage Backend Mocking

```python
@pytest.mark.asyncio
async def test_multimodal_router_upload_with_mock_storage():
    """Mock storage backend to avoid actual S3 calls."""
    mock_storage = AsyncMock()
    mock_storage.save_file = AsyncMock(return_value="s3://bucket/uuid.bin")
    mock_storage.exists = AsyncMock(return_value=True)

    # Patch the storage factory
    with patch('ragcore.modules.multimodal.storage.get_storage_backend_from_config',
               return_value=mock_storage):

        # Test upload endpoint
        client = TestClient(app)
        response = client.post(
            "/multimodal/upload",
            files={"file": ("test.jpg", b"fake jpeg data")},
            data={"session_id": str(uuid4()), "modality": "image"},
        )

        assert response.status_code == 200
        mock_storage.save_file.assert_called_once()
```

---

## PART 5: INTEGRATION TEST EXAMPLES

### 5.1 Complete Integration Test: Upload → Embed → Search

```python
"""
Integration test: Full multimodal pipeline from upload to search.
Tests: Database persistence, storage, embedding, search ranking.
"""

import pytest
from uuid import uuid4
from datetime import datetime
from fastapi.testclient import TestClient
import asyncio


@pytest.mark.asyncio
class TestMultimodalIntegration:
    """Integration tests for full multimodal pipeline."""

    async def test_upload_image_search_end_to_end(
        self,
        fastapi_client: TestClient,
        db_session,
        local_storage_backend,
        embedding_pipeline_with_mock,
        sample_session_id,
    ):
        """E2E: Upload image → store → embed → search."""
        from ragcore.modules.multimodal.models import ModuleType
        from ragcore.tests.factories import TestDataFactory

        # Arrange
        image_data = TestDataFactory.create_test_image(5)  # 5KB PNG
        session_id = sample_session_id

        # Act 1: Upload image
        response = fastapi_client.post(
            "/multimodal/upload",
            files={"file": ("test.jpg", image_data)},
            data={
                "session_id": str(session_id),
                "modality": "image",
                "metadata": '{"source": "test"}',
            },
            headers={"Authorization": "Bearer test-token"},
        )

        # Assert 1: Upload successful
        assert response.status_code == 200
        upload_data = response.json()
        content_id = upload_data["id"]

        # Act 2: Wait for processing (async)
        # In real test, use background job polling or immediate processing
        await asyncio.sleep(0.5)

        # Act 3: Check processing status
        response = fastapi_client.get(
            f"/multimodal/processing-status/{content_id}",
            headers={"Authorization": "Bearer test-token"},
        )

        assert response.status_code == 200
        status_data = response.json()
        assert status_data["is_processed"] is True

        # Act 4: Get content with chunks
        response = fastapi_client.get(
            f"/multimodal/content/{content_id}",
            headers={"Authorization": "Bearer test-token"},
        )

        assert response.status_code == 200
        content_data = response.json()
        assert content_data["chunks_count"] > 0

        # Act 5: Search across chunks
        response = fastapi_client.post(
            "/multimodal/search",
            json={
                "session_id": str(session_id),
                "query": "text from image",
                "limit": 5,
                "modalities": ["image"],
            },
            headers={"Authorization": "Bearer test-token"},
        )

        # Assert 5: Search returns results
        assert response.status_code == 200
        search_data = response.json()
        assert len(search_data["results"]) > 0

        # Verify results are sorted by similarity_score (descending)
        scores = [r["similarity_score"] for r in search_data["results"]]
        assert scores == sorted(scores, reverse=True)


    async def test_multi_modality_session_context_allocation(
        self,
        db_session,
        embedding_pipeline_with_mock,
        sample_session_id,
    ):
        """Integration: Multiple modalities share context window fairly."""
        from ragcore.modules.multimodal.context_manager import ContextWindowManagerForMultiModal
        from ragcore.modules.multimodal.models import (
            MultiModalChunk, ModuleType, MultiModalMetadata
        )

        session_id = sample_session_id

        # Create context manager with standard budget
        manager = ContextWindowManagerForMultiModal(
            total_context_window_tokens=4096,
            reserved_tokens=512,  # System prompt, etc
        )

        # Create mixed chunks: images (1.5x weight), audio (2.0x), video (2.5x)
        chunks = [
            # 10 image chunks
            *[
                MultiModalChunk(
                    content=f"image chunk {i}",
                    modality=ModuleType.IMAGE,
                    confidence_score=0.95,
                    source_index=i,
                )
                for i in range(10)
            ],
            # 5 audio chunks
            *[
                MultiModalChunk(
                    content=f"audio chunk {i}",
                    modality=ModuleType.AUDIO,
                    confidence_score=0.90,
                    source_index=i + 10,
                )
                for i in range(5)
            ],
            # 3 video chunks
            *[
                MultiModalChunk(
                    content=f"video chunk {i}",
                    modality=ModuleType.VIDEO,
                    confidence_score=0.88,
                    source_index=i + 15,
                )
                for i in range(3)
            ],
        ]

        # Act: Select chunks under budget
        selected, report = manager.select_chunks_under_budget(
            chunks=chunks,
            budget_tokens=3500,  # 3500 of 3584 available
        )

        # Assert 1: Selection respects budget
        assert report.total_tokens_allocated <= 3500

        # Assert 2: All modalities represented (fairness)
        selected_modalities = {c.modality for c in selected}
        assert ModuleType.IMAGE in selected_modalities
        assert ModuleType.AUDIO in selected_modalities or len(selected) > 0

        # Assert 3: Weighted allocation respected
        # Audio (2.0x) should have fewer chunks than image (1.5x)
        audio_count = sum(1 for c in selected if c.modality == ModuleType.AUDIO)
        image_count = sum(1 for c in selected if c.modality == ModuleType.IMAGE)

        # Audio chunks cost more, so fewer selected
        assert audio_count <= image_count

        # Assert 4: Report provides transparency
        assert report.modality_allocations is not None
        assert report.total_tokens_allocated > 0
        assert report.budget_remaining >= 0
```

### 5.2 Storage Backend Switching Test

```python
@pytest.mark.parametrize("storage_type", ["local", "s3", "azure_blob"])
@pytest.mark.asyncio
async def test_storage_backend_abstraction(storage_type):
    """Parametrized test: Same code works with different storage backends."""
    from ragcore.modules.multimodal.storage.factory import get_storage_backend_from_config
    from ragcore.config import settings

    # Arrange: Create appropriate backend
    if storage_type == "local":
        backend = get_storage_backend_from_config(
            settings,
            storage_type="local",
            base_path="/tmp/ragcore-test",
        )
    elif storage_type == "s3":
        # Use LocalStack for testing
        backend = get_storage_backend_from_config(
            settings,
            storage_type="s3",
            bucket="ragcore-test",
            region="us-east-1",
            endpoint_url="http://localhost:4566",  # LocalStack
        )
    else:  # azure_blob
        backend = get_storage_backend_from_config(
            settings,
            storage_type="azure_blob",
            container="ragcore-test",
        )

    # Act & Assert: Same operations work for all backends
    test_content = b"test file content"
    test_id = "test-file-123"

    # Save
    path = await backend.save_file(test_id, test_content)
    assert path is not None

    # Retrieve
    retrieved = await backend.get_file(path)
    assert retrieved == test_content

    # Exists
    exists = await backend.exists(path)
    assert exists is True

    # Delete
    deleted = await backend.delete_file(path)
    assert deleted is True

    # Verify deleted
    exists_after = await backend.exists(path)
    assert exists_after is False
```

---

## PART 6: CI/CD INTEGRATION CHECKLIST

### 6.1 GitHub Actions Workflow Template

```yaml
# .github/workflows/integration-tests.yml
name: Integration Tests

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: ankane/pgvector:latest
        env:
          POSTGRES_USER: ragcore
          POSTGRES_PASSWORD: ragcore
          POSTGRES_DB: ragcore_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

      redis:
        image: redis:7-alpine
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python 3.11
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
          pip install pytest-cov pytest-asyncio pytest-xdist

      - name: Run unit tests
        run: pytest tests/unit -v --cov=ragcore --cov-report=xml -n auto

      - name: Run integration tests
        env:
          DATABASE_URL: postgresql://ragcore:ragcore@localhost:5432/ragcore_test
          REDIS_URL: redis://localhost:6379/0
        run: pytest tests/integration -v --cov=ragcore --cov-report=xml

      - name: Run E2E tests (if critical path)
        if: github.event_name == 'pull_request'
        run: pytest tests/e2e -v -m critical_path

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml
          flags: integration-tests
          fail_ci_if_error: true

      - name: Performance regression check
        run: |
          # Store baseline metrics
          pytest tests/performance --benchmark-json=benchmark.json

          # Compare against baseline (if exists)
          if [ -f baseline.json ]; then
            pytest tests/performance --benchmark-compare=baseline.json
          fi
```

### 6.2 Pre-Commit Hook for Test Coverage

```yaml
# .pre-commit-config.yaml (add to existing)
- repo: local
  hooks:
    - id: pytest-cov-unit
      name: pytest coverage (unit)
      entry: bash -c 'pytest tests/unit --cov=ragcore --cov-fail-under=80'
      language: system
      types: [python]
      stages: [commit]
      pass_filenames: false

    - id: pytest-asyncio-check
      name: pytest async syntax check
      entry: bash -c 'pytest tests/integration --collect-only -q'
      language: system
      types: [python]
      stages: [commit]
      pass_filenames: false
```

### 6.3 Merge Checklist

```markdown
## Integration Tests Merge Checklist

- [ ] Unit tests passing (coverage ≥80%)
- [ ] Integration tests passing (coverage ≥75%)
- [ ] Critical path tests (17 minimum) all passing
- [ ] No async test warnings (pytest-asyncio)
- [ ] Database migrations tested (upgrade + downgrade)
- [ ] Storage backend tests passing (local + one cloud)
- [ ] Provider fallback tested (primary + fallback)
- [ ] Error recovery tests passing
- [ ] Performance baseline maintained (no >10% regression)
- [ ] Memory leak check passing (psutil monitoring)
- [ ] Code coverage report generated
- [ ] Documentation updated (if architecture changed)
```

---

## PART 7: PERFORMANCE & LOAD TESTING

### 7.1 Performance Baseline Metrics (Target)

| Operation | Target P50 | Target P95 | Target P99 |
|-----------|-----------|-----------|-----------|
| **Image Embed (5KB)** | 150ms | 300ms | 500ms |
| **Audio Embed (10KB, 5s)** | 200ms | 400ms | 800ms |
| **Video Embed (100KB, 3s)** | 500ms | 1000ms | 2000ms |
| **Semantic Search (100 chunks)** | 50ms | 100ms | 200ms |
| **Storage Upload (5MB)** | 200ms | 400ms | 600ms |
| **Storage Download (5MB)** | 100ms | 200ms | 400ms |
| **Database Insert (chunk)** | 10ms | 20ms | 50ms |
| **Batch Process (10 items)** | 800ms | 1500ms | 2500ms |

### 7.2 Load Test Template (using locust)

```python
# tests/load/locustfile.py
from locust import HttpUser, task, between
import random
from uuid import uuid4
import json


class MultimodalRagUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        """Setup: Create session and upload content."""
        self.session_id = str(uuid4())
        self.content_ids = []

        # Upload 3-5 sample contents
        for i in range(random.randint(3, 5)):
            with self.client.post(
                "/multimodal/upload",
                files={"file": ("test.jpg", b"x" * 5000)},
                data={
                    "session_id": self.session_id,
                    "modality": "image",
                },
                headers={"Authorization": "Bearer test-token"},
                catch_response=True,
            ) as resp:
                if resp.status_code == 200:
                    self.content_ids.append(resp.json()["id"])

    @task(3)
    def search_content(self):
        """Search (3x frequency)."""
        self.client.post(
            "/multimodal/search",
            json={
                "session_id": self.session_id,
                "query": "test query",
                "limit": 10,
            },
            headers={"Authorization": "Bearer test-token"},
        )

    @task(1)
    def upload_content(self):
        """Upload (1x frequency)."""
        self.client.post(
            "/multimodal/upload",
            files={"file": ("test.jpg", b"x" * 5000)},
            data={
                "session_id": self.session_id,
                "modality": "image",
            },
            headers={"Authorization": "Bearer test-token"},
        )

    @task(1)
    def get_status(self):
        """Status check (1x frequency)."""
        if self.content_ids:
            content_id = random.choice(self.content_ids)
            self.client.get(
                f"/multimodal/processing-status/{content_id}",
                headers={"Authorization": "Bearer test-token"},
            )


# Run: locust -f tests/load/locustfile.py -u 50 -r 10 -t 5m --host=http://localhost:8000
```

Run with:
```bash
# 50 users, 10 users spawned per second, run for 5 minutes
locust -f tests/load/locustfile.py -u 50 -r 10 -t 5m --host=http://localhost:8000

# Generate HTML report
locust -f tests/load/locustfile.py -u 50 -r 10 -t 5m --host=http://localhost:8000 --html=report.html
```

### 7.3 Memory Monitoring Test

```python
@pytest.mark.asyncio
async def test_large_file_memory_usage():
    """Test that 500MB upload doesn't cause OOM."""
    import psutil
    import os

    process = psutil.Process(os.getpid())
    baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Upload 500MB file
    large_content = b"x" * (500 * 1024 * 1024)

    # Peak memory should not exceed baseline + 600MB
    peak_memory = baseline_memory

    # ... upload logic ...

    peak_memory = max(peak_memory, process.memory_info().rss / 1024 / 1024)
    memory_delta = peak_memory - baseline_memory

    assert memory_delta < 600, f"Memory delta {memory_delta}MB exceeds 600MB limit"
```

---

## PART 8: RECOMMENDED IMPLEMENTATION ROADMAP

### Phase 0 (Week 1-2): Foundation
1. Create `tests/conftest.py` with async fixtures
2. Add test data factory
3. Implement 5-7 critical path tests (happy path)
4. Add mocking patterns for embedding providers

**Effort**: 20-30 hours

### Phase 1 (Week 3-4): Storage & Database
1. Add 40-50 storage backend tests (local + S3)
2. Add 20-30 database constraint tests
3. Add ChromaDB sync tests
4. Test cascade deletes and foreign keys

**Effort**: 25-35 hours

### Phase 2 (Week 5-6): Provider Resilience
1. Add 30-40 provider fallback tests
2. Add error recovery tests
3. Add health check tests
4. Load test embedding endpoints

**Effort**: 20-25 hours

### Phase 3 (Week 7-8): E2E & Performance
1. Add 20-30 E2E pipeline tests
2. Add performance benchmarking
3. Add load tests (locust)
4. Memory leak testing

**Effort**: 25-30 hours

**Total**: 90-120 hours for comprehensive integration suite

---

## PART 9: TESTING BEST PRACTICES SPECIFIC TO RAGCORE

### 9.1 Session Isolation Pattern

```python
@pytest.fixture
async def isolated_session(db_session):
    """Create isolated session for each test."""
    # Start transaction
    async with db_session.begin():
        yield db_session
        # Automatic rollback at end (no commit)
```

### 9.2 Async Cleanup Pattern

```python
@pytest.mark.asyncio
async def test_with_cleanup():
    """Test with guaranteed async cleanup."""
    resources = []

    try:
        # Setup
        resource = await create_async_resource()
        resources.append(resource)

        # Test
        assert await resource.is_valid()
    finally:
        # Async cleanup
        for resource in resources:
            await resource.cleanup()
```

### 9.3 Parametrized Testing for Multiple Modalities

```python
@pytest.mark.parametrize("modality,file_factory", [
    ("image", lambda: TestDataFactory.create_test_image()),
    ("audio", lambda: TestDataFactory.create_test_audio()),
    ("video", lambda: TestDataFactory.create_test_video()),
])
@pytest.mark.asyncio
async def test_modality_pipeline(modality, file_factory):
    """Test same pipeline with different modalities."""
    content = file_factory()
    # ... test logic ...
```

### 9.4 Fixture Composition

```python
@pytest.fixture
def complete_test_environment(
    db_session,
    local_storage_backend,
    embedding_pipeline_with_mock,
    sample_session_id,
):
    """Compose all fixtures for complex integration tests."""
    return {
        "db": db_session,
        "storage": local_storage_backend,
        "embeddings": embedding_pipeline_with_mock,
        "session_id": sample_session_id,
    }
```

---

## Summary & Next Steps

| Item | Current | Target | Gap |
|------|---------|--------|-----|
| **Total Tests** | 70 | 400-500 | +330-430 |
| **Integration Tests** | ~12 | 150-200 | +138-188 |
| **E2E Tests** | 0 | 20-50 | +20-50 |
| **Coverage** | 60% | 85%+ | +25% |
| **Async Patterns** | Basic | Advanced | Docs needed |
| **conftest.py** | Missing | Complete | Create immediately |

**Quick Wins (Next 1-2 days)**:
1. Create `/tests/conftest.py` with 10 core fixtures
2. Add TestDataFactory for minimal files
3. Implement 5 critical path tests
4. Document mocking patterns

---

## References

- pytest-asyncio: https://pytest-asyncio.readthedocs.io/
- pytest fixtures: https://docs.pytest.org/en/stable/fixture.html
- FastAPI testing: https://fastapi.tiangolo.com/advanced/testing-dependencies/
- hypothesis (property-based testing): https://hypothesis.readthedocs.io/
- locust (load testing): https://locust.io/
- psutil (memory monitoring): https://psutil.readthedocs.io/

