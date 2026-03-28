"""
Phase 0 Critical Fixes - Complete Implementation Plan
Based on online research and codebase analysis
Generated: 2026-03-28
"""

# PHASE 0 CRITICAL FIXES - IMPLEMENTATION PLAYBOOK

## EXECUTIVE SUMMARY

Phase 0 involves 4 remaining critical tasks (after storage + embeddings + router completed):
- Task 4: ChromaDB Sync Integration (2 hours)
- Task 5: Smart Audio/Video Chunking (4 hours)
- Task 6: Integration Test Suite (6 hours)
- Task 7: Success Criteria Verification (2 hours)

**Total Timeline**: 1 week (40-45 hours), 15 hours completed (43%)

---

## TASK 4: ChromaDB SYNC INTEGRATION (2 hours)

### What's Already Implemented ✅
- ChromaMemorySyncManager: `sync_single_memory()`, `batch_sync_memories()`, `process_sync_queue()`
- CircuitBreaker in client.py: 5-failure threshold, 10-min reset window
- Exponential backoff: 1, 2, 4, 8, 16 seconds
- Collection manager with type-based partitioning
- Performance router: P50/P95 latency tracking
- Database tables: `chroma_sync_state`, `chroma_sync_queue`
- Hybrid deployment mode (PostgreSQL primary, ChromaDB cache)

**Files**:
- `/ragcore/modules/memory/chroma/sync_manager.py` (300 lines)
- `/ragcore/modules/memory/chroma/client.py` (220 lines)
- `/ragcore/modules/memory/chroma/collection_manager.py` (350 lines)
- `/ragcore/modules/memory/chroma/performance_router.py` (340 lines)
- Migration: `/alembic/versions/005_phase5_chroma.py` (82 lines)

### What's Missing ❌
1. **Multimodal chunks not synced to ChromaDB** (critical)
   - MultiModalRouter persists to database but doesn't call sync_manager
   - EmbeddingProviderAdapter generates vectors but doesn't enqueue for ChromaDB
   - NO integration between multimodal and memory subsystems

2. **HybridMemoryStore._get_embedding() returns None** (returns None line 79)
   - Should use EmbeddingProviderAdapter
   - Causes dual-write failures (embeddings missing)

3. **Production Persistence Issues** (not in codebase)
   - sync_queue is in-memory (lost on restart)
   - No dead-letter queue for permanent failures
   - No jitter on exponential backoff (thundering herd)

### Implementation Plan (120 minutes)

#### Step 1: Wire Multimodal to ChromaDB (45 min)
**File**: `/ragcore/modules/multimodal/router.py` - upload endpoint

```python
# After saving to storage/database, queue ChromaDB sync:
if chunk and chunk.embedding:
    # Import sync manager
    from ragcore.modules.memory.chroma.sync_manager import ChromaMemorySyncManager
    from ragcore.modules.memory.chroma.client import ChromaClientFactory

    # Get sync manager instance (singleton or from DI)
    client_mgr = ChromaClientFactory.get_client()
    sync_mgr = ChromaMemorySyncManager(client_mgr.collection_manager, config=settings)

    # Queue sync for each extracted chunk
    for chunk in processing_result.chunks:
        await sync_mgr.sync_single_memory(
            session_id=session_id,
            memory_id=chunk.id,
            embedding=chunk.embedding,
            memory_type="multimodal_chunk",
            document=chunk.content[:500],  # Summary for search
            metadata={
                "modality": chunk.modality.value,
                "source": content_id,
                "confidence": chunk.confidence_score,
                "storage_path": storage_path,
            },
            operation="insert"
        )
```

**Time**: 45 minutes (read sync_manager, understand flow, integration)

#### Step 2: Implement HybridMemoryStore._get_embedding() (30 min)
**File**: `/ragcore/modules/memory/hybrid/hybrid_store.py` line 65-83

```python
async def _get_embedding(self, text: str) -> Optional[List[float]]:
    """Get embedding using EmbeddingProviderAdapter."""
    try:
        from ragcore.modules.multimodal.providers.embedding_adapter import EmbeddingProviderAdapter
        from ragcore.core.model_provider_registry import ModelProviderRegistry

        registry = ModelProviderRegistry()
        adapter = EmbeddingProviderAdapter(registry=registry)
        embedding = await adapter.embed_text(text)

        if embedding and len(embedding) == self.embedding_dimension:
            return embedding

        logger.warning(f"Invalid embedding dimension: got {len(embedding) if embedding else 0}")
        return None

    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        return None
```

**Time**: 30 minutes (copy from EmbeddingProviderAdapter, add dimension check)

#### Step 3: Database Persistence for Retry Queue (30 min)
**File**: `/ragcore/modules/memory/chroma/sync_manager.py` - modify sync_queue storage

Currently: `self.sync_queue = {}` (in-memory)
Fix: Persist to `chroma_sync_queue` table

```python
# Instead of in-memory queue, query database:
async def process_sync_queue(self) -> dict:
    """Process queued syncs from database."""
    from sqlalchemy import select, and_
    from datetime import datetime

    # Get ready-to-retry items
    stmt = select(ChromaSyncQueue).where(
        and_(
            ChromaSyncQueue.scheduled_for <= datetime.utcnow(),
            ChromaSyncQueue.retry_count < ChromaSyncQueue.max_retries
        )
    ).order_by(ChromaSyncQueue.scheduled_for)

    results = await self.db.execute(stmt)
    queue_items = results.scalars().all()

    # Process each item
    for item in queue_items:
        await self.sync_single_memory(
            session_id=item.session_id,
            memory_id=item.memory_id,
            embedding=item.embedding,
            memory_type=item.memory_type,
            document=item.document,
            metadata=item.metadata,
            operation=item.operation
        )
```

**Time**: 30 minutes (add DB query, update retry logic)

### Success Criteria
- [ ] Upload multimodal file → chunks extracted → synced to ChromaDB
- [ ] HybridMemoryStore._get_embedding() returns real 1536-dim vectors
- [ ] sync_queue persisted to database (survives restarts)
- [ ] Test: `pytest tests/test_chroma_multimodal_sync.py -v` (4+ tests pass)

---

## TASK 5: SMART CHUNKING PER MODALITY (4 hours)

### What's Already Implemented ✅
- Image processor: Extracts text via Claude Vision or Azure Vision
- Audio processor: Transcribes via Azure Speech or Whisper
- Video processor: Extracts frames + narration
- MultiModalChunk model with: content, modality, confidence_score, source_index
- TimeRange tracking: start_time, end_time available in metadata

**Files**:
- `/ragcore/modules/multimodal/processors/image_processor.py` (200 lines)
- `/ragcore/modules/multimodal/processors/audio_processor.py` (180 lines)
- `/ragcore/modules/multimodal/processors/video_processor.py` (220 lines)
- `/ragcore/modules/multimodal/models.py` (350 lines)

### What's Missing ❌
1. **Audio chunking**: All audio transcripts chunked as single blob (slow search)
   - No silence detection
   - No speaker diarization

2. **Video chunking**: Single narration text (no scene awareness)
   - No scene boundary detection
   - No temporal mapping

3. **Temporal metadata**: Metadata exists but not populated
   - start_time_sec, end_time_sec fields empty
   - boundary_type not used

4. **Configuration**: No strategy selection
   - No `chunking_strategy` parameter in config

### Research-Based Recommendations

**For Audio** (per research doc):
- Use **librosa** (0.10.0) for silence detection
- Use **pyannote.audio** (3.0.1) for speaker diarization
- **Hybrid approach**: Silence detection (fast) + diarization (accurate)
- Processing time: 30-120s per hour of audio

**For Video** (per research doc):
- Use **PySceneDetect** (0.6.1) for scene detection
- Extract adaptive keyframes within scenes
- Processing time: 3-10 minutes per hour of video

### Implementation Plan (240 minutes)

#### Step 1: Audio Silences Detection (60 min)
**File**: Create new `/ragcore/modules/multimodal/chunking/audio_chunker.py`

```python
"""Smart audio chunking with silence detection."""
import librosa
import numpy as np
from typing import List, Tuple

class AudioSilenceChunker:
    """Chunk audio by detecting silence intervals."""

    def __init__(self, energy_threshold: float = 0.02, min_chunk_duration_s: float = 2.0):
        self.energy_threshold = energy_threshold
        self.min_chunk_duration_s = min_chunk_duration_s

    async def chunk_transcript(
        self,
        audio_path: str,
        transcript: str,
        sample_rate: int = 16000
    ) -> List[dict]:
        """Chunk transcript by silence boundaries in audio.

        Returns:
            [{start_sec, end_sec, content, confidence}, ...]
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=sample_rate)

        # Detect silence frames
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        energy = librosa.power_to_db(S).mean(axis=0)

        # Find silent frames (below threshold)
        silent_frames = energy < np.percentile(energy, 20)  # Bottom 20%

        # Convert to time boundaries
        frame_times = librosa.frames_to_time(np.arange(len(silent_frames)), sr=sr)
        silence_boundaries = self._find_silence_boundaries(silent_frames, frame_times)

        # Map transcript to timing (simple: uniform distribution)
        chunks = self._split_transcript_by_silence(transcript, silence_boundaries)

        return chunks
```

**Dependencies to add**: `librosa>=0.10.0`

**Time**: 60 minutes (implement silence detection, timestamp mapping)

#### Step 2: Speaker Diarization (90 min)
**File**: Create `/ragcore/modules/multimodal/chunking/speaker_chunker.py`

```python
"""Smart audio chunking with speaker diarization."""
from pyannote.audio import Pipeline, Inference
from typing import List, Tuple

class SpeakerDiarizationChunker:
    """Chunk audio by speaker changes."""

    def __init__(self):
        # Load pretrained model
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.0",
            use_auth_token="hf_auth_token"  # Requires HF token
        )

    async def chunk_by_speakers(
        self,
        audio_path: str,
        transcript: str
    ) -> List[dict]:
        """Chunk transcript by speaker boundaries.

        Returns:
            [{start_sec, end_sec, speaker_id, content}, ...]
        """
        # Run diarization
        diarization = self.pipeline(audio_path)

        # Convert to speaker segments
        segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start_sec": turn.start,
                "end_sec": turn.end,
                "speaker": speaker,
            })

        # Map transcript to speaker segments
        chunks = self._map_transcript_to_speakers(transcript, segments)
        return chunks
```

**Dependencies to add**: `pyannote-audio>=3.0.1`
**Note**: Requires Hugging Face token + GPU recommended

**Time**: 90 minutes (model loading, speaker mapping, error handling)

#### Step 3: Video Scene Detection (60 min)
**File**: Create `/ragcore/modules/multimodal/chunking/video_chunker.py`

```python
"""Smart video chunking with scene detection."""
import cv2
from scenedetect import detect, AdaptiveDetector
from typing import List

class VideoSceneChunker:
    """Chunk video by scene boundaries."""

    async def chunk_by_scenes(
        self,
        video_path: str,
        narration_transcript: str,
        max_frames: int = 30
    ) -> List[dict]:
        """Chunk video into scenes with keyframes.

        Returns:
            [{start_sec, end_sec, keyframe_indices, narrative_content}, ...]
        """
        # Detect scenes
        scenes = detect(video_path, AdaptiveDetector())

        # Extract keyframes per scene
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        chunks = []
        for i, (start, end) in enumerate(zip(scenes[:-1], scenes[1:])):
            start_sec = start.get_seconds()
            end_sec = end.get_seconds()

            # Extract 2-3 representative frames per scene
            keyframe_indices = self._select_keyframes_in_range(
                video_path, start_sec, end_sec, fps, max_frames=3
            )

            chunks.append({
                "start_sec": start_sec,
                "end_sec": end_sec,
                "scene_id": i,
                "keyframe_indices": keyframe_indices,
                "narrative_content": narration_transcript,  # Simplified
            })

        return chunks
```

**Dependencies to add**: `scenedetect>=0.6.1`, `opencv-python>=4.8.1`

**Time**: 60 minutes (scene detection, keyframe extraction, temporal mapping)

#### Step 4: Configuration & Integration (30 min)
**File**: Updates to `/ragcore/config.py` and processor classes

```python
# Add to Settings class:
class Settings(BaseSettings):
    # Audio chunking
    audio_chunking_strategy: str = "hybrid"  # silence, diarization, hybrid
    audio_silence_threshold: float = 0.02
    audio_min_chunk_duration_s: float = 2.0

    # Video chunking
    video_chunking_strategy: str = "scene_detection"
    video_max_keyframes_per_scene: int = 3

    # Speaker diarization (optional, requires HF token)
    huggingface_token: Optional[str] = None
    enable_speaker_diarization: bool = False  # Default off (expensive)
```

**Processor updates**:
```python
# In AudioProcessor.__init__():
self.chunker = SpeakerDiarizationChunker() if enable_diarization else AudioSilenceChunker()

# In VideoProcessor.__init__():
self.chunker = VideoSceneChunker()
```

**Time**: 30 minutes (config integration, error handling)

### Success Criteria
- [ ] Upload 1-hour audio → splits into 6-8 semantic chunks
- [ ] Upload 10-min video → splits into 8-12 scene-based chunks
- [ ] Each chunk has start_time, end_time, boundary_type populated
- [ ] Processing time: <2min for audio, <5min for video
- [ ] Memory usage stays <500MB even for 1GB files

---

## TASK 6: INTEGRATION TEST SUITE (6 hours)

### What's Already Implemented ✅
- Basic async test setup with pytest-asyncio
- 70 unit tests (mostly provider adapters)
- Pydantic models for request/response validation
- Database models with relationships
- Storage backend abstraction with working implementations

**Files**:
- `/tests/conftest.py` (exists but minimal)
- `/tests/test_provider_adapters.py` (32 tests, all passing)
- `/tests/test_phase5_sprint5_multimodal_part1.py` (44 tests)
- `/tests/test_phase5_sprint5_multimodal_part2.py` (26 tests)

### What's Missing ❌
1. **Critical path tests** (0 of 17): Upload → embed → search → result
2. **Multimodal integration** (0 tests): Cross-modality workflows
3. **Storage backend tests** (0 tests): All 3 backends (local, S3, Azure)
4. **Database consistency** (0 tests): PostgreSQL + ChromaDB dual-write
5. **Error recovery** (0 tests): Retry logic, circuit breaker, fallback
6. **Large file handling** (0 tests): 500MB+ uploads with memory tracking
7. **E2E workflows** (0 tests): Full pipeline from upload to search result

### Research-Based Test Pyramid

**Current**: 70 unit tests (100% unit)
**Target**: 400-500 tests distributed as:
- Unit (60-70%): 240-280 tests
- Integration (25-30%): 100-150 tests
- E2E (5-10%): 20-50 tests

### Implementation Plan (360 minutes)

#### Step 1: Async Test Infrastructure (60 min)
**File**: Update `/tests/conftest.py`

```python
"""Pytest configuration and fixtures for RAGCORE."""
import pytest
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.pool import StaticPool

# Async test marker
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )

# Database fixture
@pytest.fixture
async def db_session():
    """In-memory SQLite for tests (fast)."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        echo=False,
    )

    async with engine.begin() as conn:
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)

    async_session = AsyncSession(engine, expire_on_commit=False)

    yield async_session

    await async_session.close()
    await engine.dispose()

# Client fixtures
@pytest.fixture
async def embedding_adapter():
    """Mock embedding adapter."""
    from ragcore.core.model_provider_registry import ModelProviderRegistry
    from ragcore.modules.multimodal.providers.embedding_adapter import EmbeddingProviderAdapter

    registry = ModelProviderRegistry()
    return EmbeddingProviderAdapter(registry=registry)

@pytest.fixture
async def storage_backend():
    """Local storage backend for tests."""
    from ragcore.modules.multimodal.storage import LocalStorage
    import tempfile

    tmp_dir = tempfile.mkdtemp()
    return LocalStorage(base_path=tmp_dir)

# Factory fixtures for test data
@pytest.fixture
def sample_image_bytes():
    """1KB PNG for tests."""
    # PNG header + minimal data
    return b'\x89PNG\r\n\x1a\n...'

@pytest.fixture
def sample_audio_bytes():
    """1KB WAV for tests."""
    # WAV header + minimal data
    return b'RIFF...'
```

**Time**: 60 minutes (fixtures, async setup, mocking)

#### Step 2: Critical Path Tests (120 min)
**File**: Create `/tests/integration/test_critical_paths.py`

```python
"""Critical path tests for Phase 0 verification."""
import pytest
from aioresponses import aioresponses

@pytest.mark.asyncio
async def test_upload_image_to_search(db_session, storage_backend, sample_image_bytes):
    """Test: Upload image → extract → embed → search."""
    # 1. Upload
    session_id = UUID()
    content_id = UUID()

    # 2. Process (mock Claude Vision response)
    with aioresponses() as mocked:
        mocked.post(
            "https://api.openai.com/v1/embeddings",
            payload={"data": [{"embedding": [0.1] * 1536}]}
        )
        # Test code

    # 3. Search
    # Assert results found

@pytest.mark.asyncio
async def test_upload_audio_to_chunks(db_session):
    """Test: Upload audio → transcribe → chunk → embed."""
    # Test audio transcription and chunking

@pytest.mark.asyncio
async def test_mixed_modality_session(db_session):
    """Test: Upload image + audio → cross-modal search."""
    # Test that both modalities searchable together

@pytest.mark.asyncio
async def test_large_file_no_oom(storage_backend, tmp_path):
    """Test: Upload 500MB file without OOM."""
    # Memory profiling: delta < 600MB

@pytest.mark.asyncio
async def test_database_constraints(db_session):
    """Test: Foreign key relationships enforced."""
    # Test cascade deletes, unique constraints

@pytest.mark.asyncio
async def test_provider_fallback():
    """Test: OpenAI down → fallback to Azure."""
    # Circuit breaker pattern

# ... 11 more critical path tests
```

**Time**: 120 minutes (implement 17 critical tests, all async)

#### Step 3: Storage Backend Tests (60 min)
**File**: Create `/tests/integration/test_storage_backends.py`

```python
"""Test all 3 storage backends with same interface."""
import pytest
from ragcore.modules.multimodal.storage import StorageBackend

@pytest.mark.parametrize("backend_fixture", ["local_storage", "s3_storage", "azure_storage"])
@pytest.mark.asyncio
async def test_save_and_retrieve(storage_backend):
    """Test CRUD operations across backends."""
    # Save
    file_id = "test_file"
    content = b"test content"
    path = await storage_backend.save_file(file_id, content)

    # Retrieve
    retrieved = await storage_backend.get_file(path)
    assert retrieved == content

    # Delete
    deleted = await storage_backend.delete_file(path)
    assert deleted is True

@pytest.mark.asyncio
async def test_large_file_streaming(s3_storage):
    """Test 500MB file doesn't load to memory."""
    # Memory profiling with large file

@pytest.mark.asyncio
async def test_backend_health_check(storage_backend):
    """Test health_check() method."""
    health = await storage_backend.health_check()
    assert health is True
```

**Time**: 60 minutes (parametrized tests, all backends)

#### Step 4: ChromaDB Sync Tests (60 min)
**File**: Create `/tests/integration/test_chroma_sync.py`

```python
"""Test ChromaDB sync and dual-write consistency."""
import pytest

@pytest.mark.asyncio
async def test_sync_single_memory(db_session):
    """Test: Save to PostgreSQL → sync to ChromaDB."""
    # Verify memory in both DBs within 0.01 timeout

@pytest.mark.asyncio
async def test_batch_sync(db_session):
    """Test: Batch sync multiple memories."""
    # Verify all synced or all failed (atomic)

@pytest.mark.asyncio
async def test_sync_retry_on_failure(db_session):
    """Test: Failed sync queued for retry."""
    # Verify exponential backoff timing

@pytest.mark.asyncio
async def test_circuit_breaker(db_session):
    """Test: 5 failures → circuit open."""
    # Verify fallback to PostgreSQL only

@pytest.mark.asyncio
async def test_full_session_resync(db_session):
    """Test: Force resync all PostgreSQL data to ChromaDB."""
    # Verify consistency between DBs
```

**Time**: 60 minutes (sync tests, retry logic, circuit breaker)

#### Step 5: CI/CD Integration (60 min)
**File**: Create/update `.github/workflows/test.yml`

```yaml
name: Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test

      # ... more services

    steps:
      - uses: actions/checkout@v3

      - name: Run critical path tests
        run: pytest tests/integration -v -m "critical"

      - name: Run storage backend tests
        run: pytest tests/integration/test_storage_backends.py -v

      - name: Run sync tests
        run: pytest tests/integration/test_chroma_sync.py -v

      - name: Coverage report
        run: pytest --cov=ragcore --cov-report=html
```

**Time**: 60 minutes (CI/CD setup, GitHub Actions)

### Success Criteria
- [ ] 17 critical path tests passing
- [ ] 150+ integration tests total
- [ ] 85%+ code coverage
- [ ] All storage backends tested
- [ ] ChromaDB sync verified
- [ ] Large file handling tested
- [ ] CI/CD pipeline green

---

## TASK 7: SUCCESS CRITERIA VERIFICATION (2 hours)

### Verification Checklist

#### Critical Issue 1: File Uploads Don't Persist
- [ ] Upload 100MB file
- [ ] Verify saved to storage backend (S3/Blob or local)
- [ ] Restart system
- [ ] File still exists and retrievable
- **Success**: Files survive system restart

#### Critical Issue 2: Embeddings Are Fake
- [ ] Upload text file
- [ ] Embed via EmbeddingProviderAdapter
- [ ] Verify real 1536-dim vectors (not deterministic placeholders)
- [ ] Different texts produce different embeddings
- **Success**: Semantic search returns relevant results

#### Critical Issue 3: Large Files Cause OOM
- [ ] Upload 500MB file
- [ ] Monitor memory during processing
- [ ] Memory delta < 600MB
- [ ] File completes without crash
- **Success**: Memory usage bounded by configuration

#### Critical Issue 4: External Storage Not Implemented
- [ ] Configure S3backend
- [ ] Upload file > 100KB
- [ ] Verify stored in S3, not database
- [ ] Verify retrieval works
- **Success**: Large files bypass database

#### Critical Issue 5: ChromaDB Sync Incomplete
- [ ] Upload video file
- [ ] Extract chunks with embeddings
- [ ] Query ChromaDB
- [ ] Verify results returned
- [ ] Verify PostgreSQL + ChromaDB consistency
- **Success**: Dual-write working, both DBs in sync

#### Phase 0 Complete When:
- [ ] All 5 critical issues resolved
- [ ] 17 critical path tests passing
- [ ] 150+ integration tests passing
- [ ] Manual verification completed
- [ ] Performance baselines met:
  - Image embed: P50=150ms
  - Search (100 chunks): P50=50ms
  - Large file (500MB): <2min upload, no OOM

### Performance Baselines

| Operation | P50 | P95 | P99 |
|-----------|-----|-----|-----|
| Image upload (20MB) | 500ms | 1000ms | 2000ms |
| Audio transcription (1 hour) | 90s | 120s | 150s |
| Video extraction (10 min) | 60s | 90s | 120s |
| Text embedding | 50ms | 100ms | 200ms |
| Search (100 chunks) | 50ms | 100ms | 200ms |
| ChromaDB sync (50 items) | 100ms | 200ms | 500ms |

### Rollout Plan

**Day 1**: Critical fixes deployed to staging
**Day 2**: Integration tests passing in CI/CD
**Day 3**: Performance baselines verified
**Day 4**: Manual QA sign-off
**Day 5**: Deploy to production

---

## SUMMARY: 14-DAY SPRINT ROADMAP

### Week 1
- **Mon**: ChromaDB sync integration + HybridMemoryStore._get_embedding()
- **Tue**: Audio silence detection + speaker diarization
- **Wed**: Video scene detection + keyframe extraction
- **Thu**: Integration test infrastructure (conftest, fixtures)
- **Fri**: Critical path tests (17 tests, all passing)

### Week 2
- **Mon**: Storage backend tests + ChromaDB sync tests
- **Tue**: E2E workflow tests + error recovery tests
- **Wed**: Performance baseline measurements
- **Thu**: CI/CD pipeline setup + GitHub Actions
- **Fri**: Success criteria verification + sign-off

---

## DEPENDENCIES CHECKLIST

### New Libraries to Install
```bash
# Task 4: ChromaDB sync (already installed)
# Task 5: Smart chunking
pip install librosa>=0.10.0
pip install pyannote-audio>=3.0.1
pip install scenedetect>=0.6.1
pip install opencv-python>=4.8.1

# Task 6: Integration tests (already installed)
pip install pytest-asyncio>=0.21.0
pip install aioresponses>=0.7.4
```

### System Dependencies
- FFmpeg (for video processing)
- PostgreSQL 13+ (for testing)
- Hugging Face token (for speaker diarization)

---

Generated: 2026-03-28
Based on research of: ChromaDB patterns, smart chunking techniques, integration testing best practices
