# RAGCORE Integration Testing Implementation Guide

**Target Audience**: Phase 0 Sprint developers
**Focus**: Practical step-by-step setup for your codebase
**Timeframe**: Implement in next 2-3 sprints

---

## 1. IMMEDIATE ACTIONS (First 2 Days)

### 1.1 Create conftest.py

File: `/e/Backup/pgwiz/rag/tests/conftest.py`

Key fixtures to implement:
- `db_engine` (async SQLAlchemy with in-memory SQLite)
- `db_session` (transaction-scoped for rollback)
- `local_storage_backend` (using tmp_path)
- `embedding_pipeline_with_mock` (deterministic test embeddings)
- `sample_session_id` and `sample_multimodal_content`
- `fastapi_client` (TestClient override)

**Time**: 2-3 hours
**Blocker**: None - can start immediately

### 1.2 Create test data factory

File: `/e/Backup/pgwiz/rag/tests/factories.py`

Key classes:
- `TestDataFactory.create_test_image()` (5KB PNG)
- `TestDataFactory.create_test_audio()` (10KB MP3)
- `TestDataFactory.create_test_video()` (100KB MP4)
- `SessionFactory` (generate test sessions with content)
- `ChunkFactory` (generate test chunks with embeddings)

**Time**: 1-2 hours
**Dependencies**: None

### 1.3 Add pytest-cov to dev dependencies

Update `/e/Backup/pgwiz/rag/pyproject.toml`:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.4",
    "pytest-asyncio>=0.23",
    "pytest-cov>=4.1",
    "pytest-xdist>=3.5",        # NEW: Parallel test execution
    "hypothesis>=6.92",          # NEW: Property-based testing
    "freezegun>=1.5",           # NEW: Time mocking
    "black>=24.1",
    "mypy>=1.8",
    "ruff>=0.3",
]
```

Then: `pip install -e ".[dev]"`

**Time**: 30 minutes

---

## 2. FIRST WEEK: CRITICAL PATH TESTS (17 tests)

### 2.1 Test File Structure

```
tests/
├── conftest.py                          # NEW: Shared fixtures
├── factories.py                         # NEW: Test data generators
├── unit/                                # Keep existing
│   ├── test_*.py
├── integration/                         # NEW: Integration tests
│   ├── conftest.py                      # Database fixtures (async)
│   ├── test_multimodal_pipeline.py      # NEW: 5 happy path tests
│   ├── test_storage_backends.py         # NEW: 3 storage tests
│   ├── test_database_integrity.py       # NEW: 2 DB constraint tests
│   ├── test_provider_fallback.py        # NEW: 2 provider tests
│   ├── test_error_recovery.py           # NEW: 2 error tests
│   ├── test_chromadb_sync.py            # NEW: 2 sync tests
│   └── test_router_auth.py              # NEW: 2 auth tests
└── e2e/                                 # NEW: End-to-end
    └── test_full_workflows.py           # NEW: 3 E2E tests
```

### 2.2 Implementation Priority

**Day 1-2**: Create conftest.py + factories.py
**Day 3-4**: Write 5 happy path tests (`test_multimodal_pipeline.py`)
**Day 5**: Write 3 storage tests (`test_storage_backends.py`)
**Day 6**: Write database + provider tests
**Day 7**: Write error recovery + sync tests

### 2.3 Critical Path Test Template

File: `/e/Backup/pgwiz/rag/tests/integration/test_multimodal_pipeline.py`

```python
"""Integration tests: Multimodal upload → embed → search pipeline."""

import pytest
from uuid import uuid4
from datetime import datetime
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from ragcore.modules.multimodal.models import ModuleType, MultiModalContent
from ragcore.tests.factories import TestDataFactory


@pytest.mark.asyncio
class TestMultimodalPipeline:
    """Test complete multimodal processing pipeline."""

    async def test_happy_path_image_upload_embed_search(
        self,
        db_session,
        local_storage_backend,
        sample_session_id,
    ):
        """
        CRITICAL PATH TEST 1: Image → Store → Embed → Search

        Steps:
        1. Upload 5KB PNG image
        2. Verify stored in database + storage backend
        3. Verify extracted to chunks with embeddings
        4. Search for similar content
        5. Verify ranked by similarity
        """
        # Arrange
        image_bytes = TestDataFactory.create_test_image(5)
        session_id = sample_session_id

        # Act 1: Create content record
        content = MultiModalContent(
            id=uuid4(),
            session_id=session_id,
            modality=ModuleType.IMAGE,
            raw_content=image_bytes,
            is_processed=False,
            created_at=datetime.utcnow(),
        )
        db_session.add(content)
        await db_session.flush()

        # Act 2: Store file
        storage_path = await local_storage_backend.save_file(
            str(content.id),
            image_bytes,
        )

        # Act 3: Update content with storage path
        content.storage_path = storage_path
        await db_session.flush()

        # Assert 1: Content persisted
        assert content.id is not None
        assert content.storage_path == storage_path

        # Assert 2: File retrievable
        retrieved = await local_storage_backend.get_file(storage_path)
        assert retrieved == image_bytes

    async def test_audio_pipeline_transcription(self, db_session):
        """CRITICAL PATH TEST 2: Audio → Transcribe → Embed."""
        # TODO: Implement after audio processor ready
        pass

    async def test_mixed_modality_session(self, db_session):
        """CRITICAL PATH TEST 3: Image + Audio in same session."""
        # TODO: Implement with both modalities
        pass

    async def test_search_cross_modality(self, db_session):
        """CRITICAL PATH TEST 4: Search returns results from all modalities."""
        # TODO: Implement search verification
        pass

    async def test_context_manager_token_allocation(self, db_session):
        """CRITICAL PATH TEST 5: ContextManager fairly allocates tokens."""
        # TODO: Implement fairness verification
        pass
```

---

## 3. STORAGE BACKEND TESTS

File: `/e/Backup/pgwiz/rag/tests/integration/test_storage_backends.py`

```python
"""Integration tests for storage backends (local, S3, Azure)."""

import pytest
from pathlib import Path


@pytest.mark.asyncio
class TestLocalStorageBackend:
    """Test LocalStorageBackend implementation."""

    async def test_save_retrieve_delete_file(self, local_storage_backend):
        """CRITICAL PATH TEST 6: Local storage CRUD."""
        test_data = b"test file content"
        file_id = "test-123"

        # Save
        path = await local_storage_backend.save_file(file_id, test_data)
        assert path.startswith("file://")

        # Retrieve
        retrieved = await local_storage_backend.get_file(path)
        assert retrieved == test_data

        # Delete
        deleted = await local_storage_backend.delete_file(path)
        assert deleted is True

        # Verify deleted
        exists = await local_storage_backend.exists(path)
        assert exists is False

    async def test_health_check(self, local_storage_backend):
        """CRITICAL PATH TEST 7: Local storage health check."""
        is_healthy = await local_storage_backend.health_check()
        assert is_healthy is True


@pytest.mark.asyncio
class TestS3StorageBackend:
    """Test S3StorageBackend with LocalStack."""

    @pytest.fixture(autouse=True)
    async def setup_localstack(self):
        """Start LocalStack S3 before each test."""
        # Use docker or moto for S3 mocking
        # TODO: Integrate with testcontainers or moto
        pass

    async def test_s3_save_retrieve_delete(self):
        """CRITICAL PATH TEST 8: S3 storage CRUD."""
        # TODO: Implement with LocalStack
        pass


@pytest.mark.parametrize("backend_type", ["local", "s3", "azure"])
@pytest.mark.asyncio
async def test_storage_backend_interface_contract(backend_type):
    """CRITICAL PATH TEST 9: All backends implement same interface."""
    # TODO: Parametrized test verifying StorageBackend contract
    pass
```

---

## 4. DATABASE INTEGRITY TESTS

File: `/e/Backup/pgwiz/rag/tests/integration/test_database_integrity.py`

```python
"""Integration tests for database constraints and referential integrity."""

import pytest
from uuid import uuid4
from datetime import datetime
from sqlalchemy.exc import IntegrityError

from ragcore.modules.multimodal.models import MultiModalContent, ModuleType


@pytest.mark.asyncio
class TestDatabaseIntegrity:
    """Test database constraints and cascading behavior."""

    async def test_foreign_key_constraint_violated(self, db_session):
        """CRITICAL PATH TEST 10: Invalid foreign key rejected."""
        # Try to insert content with non-existent session_id
        invalid_content = MultiModalContent(
            id=uuid4(),
            session_id=uuid4(),  # Non-existent session
            modality=ModuleType.IMAGE,
            created_at=datetime.utcnow(),
        )

        db_session.add(invalid_content)

        with pytest.raises(IntegrityError):
            await db_session.flush()

        await db_session.rollback()

    async def test_cascade_delete_on_session_deletion(self, db_session):
        """CRITICAL PATH TEST 11: Deleting session deletes all content."""
        # TODO: Implement with actual session deletion
        pass
```

---

## 5. PROVIDER FALLBACK TESTS

File: `/e/Backup/pgwiz/rag/tests/integration/test_provider_fallback.py`

```python
"""Integration tests for provider fallback and error recovery."""

import pytest
from unittest.mock import AsyncMock, patch

from ragcore.modules.multimodal.providers.embedding_adapter import EmbeddingProviderAdapter
from ragcore.core.model_provider_registry import ModelProviderRegistry, ProviderType, ProviderConfig


@pytest.mark.asyncio
class TestProviderFallback:
    """Test provider fallback mechanisms."""

    async def test_primary_provider_fails_fallback_succeeds(self):
        """CRITICAL PATH TEST 12: Fallback when primary provider fails."""
        registry = ModelProviderRegistry()

        # Register both providers
        config_primary = ProviderConfig(
            provider=ProviderType.OPENAI,
            api_key="test-key",
            endpoint="https://api.openai.com",
        )
        config_fallback = ProviderConfig(
            provider=ProviderType.AZURE_OPENAI,
            api_key="test-key",
            endpoint="https://test.openai.azure.com",
        )

        registry.register_provider(config_primary)
        registry.register_provider(config_fallback)

        adapter = EmbeddingProviderAdapter(registry=registry)

        # Mock primary to fail
        primary_mock = AsyncMock()
        primary_mock.embed.side_effect = Exception("API Error")

        # Mock fallback to succeed
        fallback_mock = AsyncMock()
        fallback_mock.embed.return_value = [[0.1] * 1536]

        with patch.dict(
            registry.providers,
            {ProviderType.OPENAI: primary_mock, ProviderType.AZURE_OPENAI: fallback_mock},
        ):
            result = await adapter.embed_texts(["test"])

        # Should use fallback
        assert result == [[0.1] * 1536]
        fallback_mock.embed.assert_called_once()

    async def test_all_providers_unavailable(self):
        """CRITICAL PATH TEST 13: Graceful failure when all providers down."""
        # TODO: Mock all providers to fail
        pass
```

---

## 6. ERROR RECOVERY TESTS

File: `/e/Backup/pgwiz/rag/tests/integration/test_error_recovery.py`

```python
"""Integration tests for error recovery and retry logic."""

import pytest


@pytest.mark.asyncio
class TestErrorRecovery:
    """Test error handling and recovery."""

    async def test_processing_failure_retry_succeeds(self):
        """CRITICAL PATH TEST 14: Retry after processing failure."""
        # TODO: Mock processor to fail once, succeed on retry
        pass

    async def test_partial_success_multimodal_session(self):
        """CRITICAL PATH TEST 15: Process what can be processed."""
        # TODO: Upload mixed content, fail 1/5 processors
        pass
```

---

## 7. CHROMADB SYNC TESTS

File: `/e/Backup/pgwiz/rag/tests/integration/test_chromadb_sync.py`

```python
"""Integration tests for ChromaDB synchronization."""

import pytest


@pytest.mark.asyncio
class TestChromaDBSync:
    """Test ChromaDB sync and dual-write consistency."""

    async def test_sync_multimodal_chunks_to_chromadb(self):
        """CRITICAL PATH TEST 16: Chunks sync to ChromaDB."""
        # TODO: Upload, process, sync, verify in both DBs
        pass

    async def test_dual_write_consistency(self):
        """CRITICAL PATH TEST 17: PostgreSQL + ChromaDB consistent."""
        # TODO: Compare search results from both
        pass
```

---

## 8. ROUTER AUTHENTICATION TESTS

File: `/e/Backup/pgwiz/rag/tests/integration/test_router_auth.py`

```python
"""Integration tests for router authentication and authorization."""

import pytest
from fastapi.testclient import TestClient


@pytest.mark.asyncio
class TestRouterAuth:
    """Test router JWT authentication."""

    def test_upload_without_auth_rejected(self, fastapi_client: TestClient):
        """Reject requests without JWT token."""
        response = fastapi_client.post(
            "/multimodal/upload",
            files={"file": ("test.jpg", b"data")},
            data={"session_id": "123", "modality": "image"},
        )

        assert response.status_code == 401
        assert "JWT" in response.json()["detail"]

    def test_upload_with_valid_auth_succeeds(self, fastapi_client: TestClient):
        """Accept requests with valid JWT."""
        response = fastapi_client.post(
            "/multimodal/upload",
            files={"file": ("test.jpg", b"data")},
            data={"session_id": "123", "modality": "image"},
            headers={"Authorization": "Bearer valid-test-token"},
        )

        assert response.status_code == 200
```

---

## 9. RUNNING TESTS

### 9.1 Run All Tests

```bash
cd /e/Backup/pgwiz/rag

# Install dev dependencies
pip install -e ".[dev]"

# Run all tests with coverage
pytest tests/ -v --cov=ragcore --cov-report=html

# Run only integration tests
pytest tests/integration -v

# Run specific test
pytest tests/integration/test_multimodal_pipeline.py::TestMultimodalPipeline::test_happy_path_image_upload_embed_search -v

# Run with parallel execution
pytest tests/ -n auto -v
```

### 9.2 Run by Category

```bash
# Unit tests only
pytest tests/unit -v --tb=short

# Integration tests only
pytest tests/integration -v --tb=long

# Critical path tests only
pytest -m critical_path -v

# By file size (quick tests first)
pytest tests/ -v --durations=10  # Show 10 slowest tests
```

### 9.3 Coverage Analysis

```bash
# Generate HTML coverage report
pytest tests/ --cov=ragcore --cov-report=html
open htmlcov/index.html

# Find untested files
pytest tests/ --cov=ragcore --cov-report=term-missing | grep "0%"

# Only test specific module
pytest tests/ --cov=ragcore.modules.multimodal --cov-report=html
```

---

## 10. NEXT STEPS AFTER WEEK 1

### Week 2-3: Provider API Tests (30 tests)
- Real embedding API calls (using test keys)
- Rate limiting handling
- Retry logic with exponential backoff
- Error message parsing

### Week 4-5: Load & Performance (20 tests)
- 100+ concurrent uploads
- Memory profiling during batch processing
- Storage backend performance comparison
- Database connection pooling behavior

### Week 6+: E2E & Regression (50+ tests)
- Full multi-day sessions
- Cross-session search
- Long-running background jobs
- Upgrade/downgrade database migrations

---

## 11. INTEGRATION WITH MAIN WORKFLOW

### Update .gitignore

```
tests/.pytest_cache/
tests/__pycache__/
htmlcov/
coverage.xml
.coverage
test-data/
```

### Update CI/CD (.github/workflows/tests.yml)

```yaml
- name: Run critical path tests
  run: pytest tests/integration -m critical_path -v --tb=short

- name: Report coverage
  run: pytest tests/ --cov=ragcore --cov-report=term-missing --cov-fail-under=75
```

### Pre-commit Hook

```bash
# tests/pre-commit-hook.sh
#!/bin/bash
pytest tests/unit --cov=ragcore --cov-fail-under=80 || exit 1
```

---

## 12. TROUBLESHOOTING COMMON ISSUES

### Issue: "pytest-asyncio not properly configured"

**Solution**: Ensure `pyproject.toml` has:
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

### Issue: "Database connection refused during tests"

**Solution**: Use in-memory SQLite for speed:
```python
engine = create_async_engine("sqlite+aiosqlite:///:memory:")
```

### Issue: "Async test fixture errors"

**Solution**: Mark fixture as async:
```python
@pytest.fixture
async def my_fixture():
    await setup()
    yield result
    await cleanup()
```

### Issue: "Mock provider returning wrong dimension"

**Solution**: Ensure mock returns exactly 1536-dim:
```python
mock.embed.return_value = [[0.1] * 1536 for _ in range(len(texts))]
```

---

## 13. SUCCESS CRITERIA

- [x] conftest.py created with 10+ core fixtures
- [ ] 17 critical path tests passing
- [ ] 85%+ line coverage on multimodal module
- [ ] All async tests pass without warnings
- [ ] Database migrations tested (up/down)
- [ ] Storage backends tested (2+ backends)
- [ ] Provider fallback verified
- [ ] CI/CD integration working
- [ ] Performance baseline established
- [ ] Documentation updated

**Estimated Timeline**: 3-4 weeks for comprehensive suite

