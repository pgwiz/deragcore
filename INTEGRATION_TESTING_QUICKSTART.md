# RAGCORE Integration Testing - Quick Start Guide

**Start Here**: This guide connects all 3 documents and shows you exactly what to do today.

---

## 📋 What You Got

| Document | Purpose | Length | When to Use |
|----------|---------|--------|------------|
| **INTEGRATION_TESTING_PATTERNS.md** | Complete theory & architecture | 40KB | Read end-to-end for context |
| **INTEGRATION_TESTING_IMPLEMENTATION.md** | Step-by-step implementation roadmap | 17KB | Follow during development |
| **INTEGRATION_TESTING_REFERENCE.md** | Copy-paste code examples | 30KB | Reference while coding |

---

## 🚀 DAY 1: SETUP (3 hours)

### Step 1: Create conftest.py (60 min)

**File**: `/e/Backup/pgwiz/rag/tests/conftest.py`

Copy from: INTEGRATION_TESTING_PATTERNS.md → Section 3.2

```bash
# Install new dev dependencies
pip install -e ".[dev]"  # pytest-asyncio, pytest-cov already there
pip install pytest-xdist hypothesis freezegun  # Parallel + property tests
```

**Key fixtures to implement**:
- `event_loop` (session-scoped)
- `test_db_engine` (in-memory SQLite)
- `db_session` (per-test with rollback)
- `local_storage_backend` (tmp_path)
- `mock_embedding_provider` (deterministic)
- `embedding_pipeline_with_mock`

### Step 2: Create factories.py (45 min)

**File**: `/e/Backup/pgwiz/rag/tests/factories.py`

Copy from: INTEGRATION_TESTING_PATTERNS.md → Section 3.3

**Implement**:
- `TestDataFactory.create_test_image()`
- `TestDataFactory.create_test_audio()`
- `TestDataFactory.create_test_video()`
- `SessionFactory` (generates test sessions)
- `ChunkFactory` (generates test chunks)

### Step 3: Verify Setup Works (30 min)

```bash
cd /e/Backup/pgwiz/rag

# Test that pytest finds fixtures
pytest --fixtures -q | grep "db_session\|local_storage"

# Run existing tests to verify nothing broke
pytest tests/unit/test_health.py -v
```

**Success Criteria**:
- ✅ conftest.py imports without errors
- ✅ factories.py creates minimal test files (5KB, 10KB, 100KB)
- ✅ Existing tests still pass

---

## 📝 DAY 2-3: WRITE 5 CRITICAL PATH TESTS (6 hours)

### Create test file

**File**: `/e/Backup/pgwiz/rag/tests/integration/test_multimodal_pipeline.py`

Copy from: INTEGRATION_TESTING_REFERENCE.md → Test 1-5

```python
# Structure:
class TestMultimodalPipeline:
    async def test_happy_path_image_upload_embed_search(...)  # Critical 1
    async def test_audio_pipeline_transcription(...)            # Critical 2
    async def test_mixed_modality_session(...)                  # Critical 3
    async def test_search_cross_modality(...)                   # Critical 4
    async def test_context_manager_token_allocation(...)        # Critical 5
```

### Run tests

```bash
pytest tests/integration/test_multimodal_pipeline.py -v --tb=short
```

**Expected Status After Day 3**:
- ✅ 5 integration tests written
- ✅ 3-4 passing (some dependencies may not exist yet)
- ✅ Clear error messages for failures

---

## 🔧 WEEK 2: COMPLETE 17 CRITICAL PATH TESTS (30 hours)

### Test Schedule

| Days | Tests | Files |
|------|-------|-------|
| Mon-Tue | Tests 1-5 (Happy Path) | `test_multimodal_pipeline.py` |
| Wed | Tests 6-8 (Storage) | `test_storage_backends.py` |
| Thu | Tests 9-11 (Database) | `test_database_integrity.py` |
| Fri | Tests 12-17 (Providers, Auth, Sync) | `test_provider_fallback.py`, `test_error_recovery.py`, `test_chromadb_sync.py`, `test_router_auth.py` |

### Each Test File Template

All follow this structure (from INTEGRATION_TESTING_REFERENCE.md):

```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.fixture
def specific_fixture():
    """Fixture for this test group."""
    pass

@pytest.mark.asyncio
class TestGroup:
    """Tests for specific component."""

    async def test_critical_case_1(self, specific_fixture):
        """CRITICAL PATH TEST N: Description."""
        # Arrange
        # Act
        # Assert
        pass

    async def test_critical_case_2(self, specific_fixture):
        pass
```

### Commands to Track Progress

```bash
# Count tests
pytest tests/integration --collect-only -q | tail -1

# Run critical path only
pytest tests/integration -m critical_path -v

# Coverage
pytest tests/integration --cov=ragcore.modules.multimodal --cov-report=term-missing
```

---

## 🎯 CRITICAL PATH TEST CHECKLIST

- [ ] **TEST 1**: Image upload → store → embed → search
- [ ] **TEST 2**: Audio transcription pipeline
- [ ] **TEST 3**: Mixed modality session (image + audio)
- [ ] **TEST 4**: Cross-modal search results
- [ ] **TEST 5**: Context manager token allocation fairness
- [ ] **TEST 6**: Local storage CRUD operations
- [ ] **TEST 7**: Local storage health check
- [ ] **TEST 8**: S3 storage backend (or skip if no LocalStack)
- [ ] **TEST 9**: Storage backend interface contract
- [ ] **TEST 10**: Foreign key constraint violation
- [ ] **TEST 11**: Cascade delete behavior
- [ ] **TEST 12**: Primary provider fails → fallback succeeds
- [ ] **TEST 13**: All providers unavailable → graceful failure
- [ ] **TEST 14**: Processing failure → retry succeeds
- [ ] **TEST 15**: Partial success in multi-modal session
- [ ] **TEST 16**: Chunks sync to ChromaDB
- [ ] **TEST 17**: PostgreSQL + ChromaDB consistency

---

## 🔍 COMMON ISSUES & SOLUTIONS

### Issue: "AttributeError: No attribute 'embed_texts'"

**Solution**: You haven't implemented the embedding pipeline yet. Mock it:

```python
from unittest.mock import AsyncMock

@pytest.fixture
def mock_embedding_adapter():
    adapter = AsyncMock()
    adapter.embed_texts.return_value = [[0.1] * 1536]
    return adapter
```

### Issue: "pytest-asyncio not auto mode"

**Solution**: Update pyproject.toml:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"  # ← Add this line
testpaths = ["tests"]
```

### Issue: "Database IntegrityError on valid insert"

**Solution**: You're missing a migration. Run:

```bash
cd /e/Backup/pgwiz/rag
alembic upgrade head
```

### Issue: "Test hangs forever"

**Solution**: Missing `await` on async call:

```python
# ✗ Wrong
result = backend.save_file("id", b"data")

# ✓ Correct
result = await backend.save_file("id", b"data")
```

---

## 📊 SUCCESS METRICS

After completing all 17 tests:

| Metric | Target | How to Check |
|--------|--------|-------------|
| **Tests Passing** | 17/17 | `pytest tests/integration -v \| grep PASSED \| wc -l` |
| **Coverage** | ≥75% | `pytest --cov=ragcore.modules.multimodal --cov-report=term` |
| **Async Warnings** | 0 | `pytest tests/integration -v 2>&1 \| grep -i "async"` |
| **Execution Time** | <30s | `pytest tests/integration -v --durations=5` |
| **Mock Usage** | 100% of providers | All embed calls use mocks, no real API calls |

---

## 🚀 WEEK 3+: SCALE TO 150+ TESTS

Once the 17 critical tests pass:

### Week 3: Provider API Tests (30 tests)

```bash
# Create
tests/integration/test_provider_apis.py
tests/integration/test_provider_rate_limiting.py
tests/integration/test_provider_error_handling.py

# Test
pytest tests/integration/test_provider* -v
```

### Week 4: Load & Performance (20 tests)

```bash
# Create
tests/performance/test_throughput.py
tests/performance/test_memory.py
tests/load/locustfile.py

# Run
pytest tests/performance -v
locust -f tests/load/locustfile.py -u 50 -r 10 -t 5m
```

### Week 5+: E2E & Regression (50+ tests)

```bash
# Create
tests/e2e/test_full_workflows.py
tests/e2e/test_multi_day_sessions.py
tests/e2e/test_database_migrations.py

# Run
pytest tests/e2e -v -m critical_path
```

---

## 📚 DOCUMENTATION REFERENCES

### Inside Each Test File (Copy These Comments)

```python
"""Integration tests for [Component].

CRITICAL PATH TESTS:
  1. [Test Name] - [User Action]
  2. [Test Name] - [User Action]

NON-CRITICAL (Future):
  1. [Test Name] - [Lower priority action]

Dependencies:
  - [External Service]
  - [Database Table]
  - [Async Provider]

Architecture:
  [Draw diagram of what's being tested]
"""
```

### In pytest.ini

```ini
[pytest]
markers =
    critical_path: Tests for critical user workflows
    slow: Tests that take >5 seconds
    requires_db: Tests that require database
    requires_cloud: Tests that require S3/Azure
```

---

## 🎓 RECOMMENDED READING ORDER

1. **Start**: This file (15 min)
2. **Theory**: INTEGRATION_TESTING_PATTERNS.md (30 min) → Section 1-2
3. **Implement**: INTEGRATION_TESTING_IMPLEMENTATION.md (20 min) → Sections 1-3
4. **Code**: INTEGRATION_TESTING_REFERENCE.md (reference as needed)
5. **Deep Dive**: INTEGRATION_TESTING_PATTERNS.md (60 min) → Sections 3-7

---

## 🔗 QUICK LINKS

| Need | Location |
|------|----------|
| Async fixture pattern | PATTERNS.md § 3.2 |
| Test data factory | PATTERNS.md § 3.3 |
| Mock strategies | PATTERNS.md § 4.1-4.4 |
| Complete E2E example | PATTERNS.md § 5.1 |
| Parametrized storage test | PATTERNS.md § 5.2 |
| CI/CD checklist | PATTERNS.md § 6.2 |
| Performance baseline | PATTERNS.md § 7.1 |
| Embedding pipeline test | REFERENCE.md § 1 |
| Context manager test | REFERENCE.md § 2 |
| Storage backend test | REFERENCE.md § 3 |
| Router endpoint test | REFERENCE.md § 4 |
| Provider adapter test | REFERENCE.md § 5 |
| Database model test | REFERENCE.md § 6 |

---

## ✅ YOUR FIRST 2-DAY CHECKLIST

**Day 1 (Monday)**:
- [ ] Read this Quick Start guide (15 min)
- [ ] Install dev dependencies (10 min)
- [ ] Create conftest.py from PATTERNS.md § 3.2 (60 min)
- [ ] Create factories.py from PATTERNS.md § 3.3 (45 min)
- [ ] Run `pytest --fixtures -q | grep db_session` (5 min)
- [ ] Verify existing tests still pass (10 min)

**Day 2 (Tuesday)**:
- [ ] Create integration/test_multimodal_pipeline.py (120 min)
- [ ] Implement tests 1-5 from REFERENCE.md (120 min)
- [ ] Run tests and fix errors (60 min)
- [ ] Document any blockers/dependencies found (30 min)

**Success Criteria**: `pytest tests/integration/test_multimodal_pipeline.py -v` shows at least 3 passing tests.

---

## 💬 QUESTIONS?

If any test fails, follow this troubleshooting order:

1. Check the error message
2. Search your error in PATTERNS.md § 12 "Troubleshooting"
3. Check if dependency is implemented (Database table? Provider API?)
4. Mock the dependency temporarily to make test pass
5. Create a follow-up task to implement that dependency

---

## 🎯 NORTH STAR

**After Week 2 (17 critical tests passing)**:
- You can merge confidence-building changes
- You have a reliable test harness for Phase 0 critical fixes
- New developers can run tests to understand architecture

**After Week 4 (150+ tests passing)**:
- Full integration test coverage of multimodal pipeline
- Performance baselines established
- Load testing infrastructure in place
- Ready for production hardening

