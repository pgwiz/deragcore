# Smart Chunking: Practical Implementation Guide

**Purpose**: Concrete code examples for integrating audio & video chunking into RAGCORE
**Status**: Ready for development phase
**Complexity**: Medium (library integration + temporal tracking)

---

## FILE 1: Extended Data Models (models_update.py)

Add to `/ragcore/modules/multimodal/models.py`:

```python
# ADDITIONS TO EXISTING models.py

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class BoundaryType(str, Enum):
    """Types of chunk boundaries."""
    SILENCE = "silence"                    # Audio: silence detection
    SPEAKER_CHANGE = "speaker_change"      # Audio: speaker diarization
    SCENE_CUT = "scene_cut"               # Video: hard cut
    FADE = "fade"                         # Video: fade transition
    DISSOLVE = "dissolve"                 # Video: cross-fade
    KEYFRAME = "keyframe"                 # Video: adaptive sampling
    TIMESTAMP = "timestamp"               # Fixed interval


@dataclass
class TemporalMetadata:
    """Temporal information for audio/video chunks."""

    start_time_sec: float                 # Absolute time in source media
    end_time_sec: float                   # Absolute time in source media
    duration_sec: float = field(init=False)  # Computed: end - start

    boundary_type: BoundaryType = BoundaryType.TIMESTAMP
    boundary_confidence: float = 1.0      # 0.0-1.0: confidence in boundary detection

    # Audio-specific
    speaker_id: Optional[str] = None      # Speaker identifier (e.g., "Speaker_1")
    speaker_label: Optional[str] = None   # Human label if available
    speech_rate_wpm: Optional[float] = None  # Words per minute in segment

    # Video-specific
    visual_complexity: Optional[float] = None  # Entropy score (0.0-1.0)
    motion_magnitude: Optional[float] = None   # Optical flow magnitude
    is_scene_boundary: bool = False       # Transition between scenes
    is_keyframe: bool = False             # Representative frame

    def __post_init__(self):
        self.duration_sec = self.end_time_sec - self.start_time_sec

    def to_dict(self) -> dict:
        """Convert to database-storable format."""
        return {
            "start_time_sec": self.start_time_sec,
            "end_time_sec": self.end_time_sec,
            "duration_sec": self.duration_sec,
            "boundary_type": self.boundary_type.value,
            "boundary_confidence": self.boundary_confidence,
            "speaker_id": self.speaker_id,
            "speaker_label": self.speaker_label,
            "speech_rate_wpm": self.speech_rate_wpm,
            "visual_complexity": self.visual_complexity,
            "motion_magnitude": self.motion_magnitude,
            "is_scene_boundary": self.is_scene_boundary,
            "is_keyframe": self.is_keyframe,
        }


# MODIFY: MultiModalChunk to include temporal metadata

@dataclass
class MultiModalChunk:
    """Represents a chunk extracted from multi-modal content."""

    id: UUID
    session_id: UUID
    memory_id: Optional[UUID] = None
    modality: ModuleType = ModuleType.TEXT
    content: str = ""
    embedding: List[float] = field(default_factory=list)

    # NEW: Temporal metadata for audio/video chunks
    temporal_metadata: Optional[TemporalMetadata] = None

    metadata: MultiModalMetadata = field(...)
    source_index: int = 0
    confidence_score: float = 1.0
    is_critical: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)

    def get_time_range(self) -> tuple[float, float]:
        """Return (start_sec, end_sec) for audio/video chunks."""
        if self.temporal_metadata:
            return (
                self.temporal_metadata.start_time_sec,
                self.temporal_metadata.end_time_sec,
            )
        return (0.0, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = {
            "id": str(self.id),
            "session_id": str(self.session_id),
            "memory_id": str(self.memory_id) if self.memory_id else None,
            "modality": self.modality.value,
            "content": self.content,
            "embedding": self.embedding,
            "source_index": self.source_index,
            "confidence_score": self.confidence_score,
            "is_critical": self.is_critical,
            "created_at": self.created_at.isoformat(),
            "temporal_metadata": (
                self.temporal_metadata.to_dict()
                if self.temporal_metadata
                else None
            ),
        }
        return data
```

---

## FILE 2: Chunking Strategy Interfaces (chunking_strategies.py)

Create `/ragcore/modules/multimodal/chunking/strategies.py`:

```python
"""Chunking strategy interfaces for audio & video."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChunkBoundary:
    """Single chunk boundary specification."""
    start_sec: float
    end_sec: float
    boundary_type: str  # "silence", "speaker_change", "scene_cut", etc.
    confidence: float = 1.0


class ChunkingStrategy(ABC):
    """Base class for chunking strategies."""

    @abstractmethod
    async def chunk(
        self,
        source_path: str,
        min_duration_sec: float = 15.0,
        max_duration_sec: float = 90.0,
        **kwargs
    ) -> List[ChunkBoundary]:
        """
        Compute chunk boundaries for source media.

        Args:
            source_path: Path to audio/video file (or URL for streaming)
            min_duration_sec: Minimum chunk duration
            max_duration_sec: Maximum chunk duration
            **kwargs: Strategy-specific parameters

        Returns:
            List of ChunkBoundary objects in temporal order
        """
        pass

    def _enforce_duration_limits(
        self,
        boundaries: List[ChunkBoundary],
        min_sec: float,
        max_sec: float,
    ) -> List[ChunkBoundary]:
        """Merge short chunks, split long chunks to meet duration constraints."""
        adjusted = []

        for boundary in boundaries:
            duration = boundary.end_sec - boundary.start_sec

            # Skip very short chunks (merge with next)
            if duration < min_sec:
                continue

            # Split long chunks
            if duration > max_sec:
                num_splits = int(duration / max_sec) + 1
                split_duration = duration / num_splits
                for i in range(num_splits):
                    start = boundary.start_sec + i * split_duration
                    end = min(start + split_duration, boundary.end_sec)
                    adjusted.append(
                        ChunkBoundary(
                            start_sec=start,
                            end_sec=end,
                            boundary_type=f"{boundary.boundary_type}_split_{i}",
                            confidence=boundary.confidence * 0.95,  # Slightly lower confidence
                        )
                    )
            else:
                adjusted.append(boundary)

        return adjusted


# ============================================================================
# AUDIO CHUNKING STRATEGIES
# ============================================================================

class SilenceDetectionChunking(ChunkingStrategy):
    """Silence-based chunking using energy threshold."""

    def __init__(self, threshold_multiplier: float = 0.5, min_silence_ms: int = 300):
        self.threshold_multiplier = threshold_multiplier  # 0.4-0.6
        self.min_silence_ms = min_silence_ms

    async def chunk(
        self,
        source_path: str,
        min_duration_sec: float = 15.0,
        max_duration_sec: float = 90.0,
        **kwargs
    ) -> List[ChunkBoundary]:
        """Detect silence regions and create chunks."""
        try:
            import librosa
            import numpy as np
        except ImportError:
            raise ImportError("Install librosa: pip install librosa")

        logger.info(f"Silence detection chunking: {source_path}")

        # Load audio
        y, sr = librosa.load(source_path, sr=16000)
        hop_length = 512

        # Compute energy
        S = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length)
        log_S = librosa.power_to_db(S, ref=np.max)
        energy = np.mean(log_S, axis=0)

        # Dynamic thresholding
        threshold = np.mean(energy) * self.threshold_multiplier
        silence_mask = energy < threshold

        # Find silence regions
        min_frames = int(self.min_silence_ms * sr / 1000 / hop_length)

        # Convert to time boundaries
        boundaries = []
        in_silence = False
        silence_start = 0.0

        for frame_idx, is_silent in enumerate(silence_mask):
            time_sec = librosa.frames_to_time(frame_idx, sr=sr, hop_length=hop_length)

            if is_silent and not in_silence:
                silence_start = time_sec
                in_silence = True
            elif not is_silent and in_silence:
                # Silence ended -> use as chunk boundary
                boundaries.append(
                    ChunkBoundary(
                        start_sec=silence_start,
                        end_sec=time_sec,
                        boundary_type="silence",
                        confidence=0.85,
                    )
                )
                in_silence = False

        # Convert silence regions to speech regions (chunks)
        chunks = []
        prev_end = 0.0

        for boundary in boundaries:
            if boundary.start_sec > prev_end + min_duration_sec:
                chunks.append(
                    ChunkBoundary(
                        start_sec=prev_end,
                        end_sec=boundary.start_sec,
                        boundary_type="silence",
                        confidence=0.85,
                    )
                )
            prev_end = boundary.end_sec

        # Final chunk to end of audio
        total_duration = len(y) / sr
        if total_duration > prev_end + min_duration_sec:
            chunks.append(
                ChunkBoundary(
                    start_sec=prev_end,
                    end_sec=total_duration,
                    boundary_type="silence",
                    confidence=0.85,
                )
            )

        return self._enforce_duration_limits(chunks, min_duration_sec, max_duration_sec)


class SpeakerDiarizationChunking(ChunkingStrategy):
    """Speaker diarization-based chunking."""

    def __init__(self, use_huggingface_token: bool = True):
        self.use_huggingface_token = use_huggingface_token

    async def chunk(
        self,
        source_path: str,
        min_duration_sec: float = 15.0,
        max_duration_sec: float = 90.0,
        **kwargs
    ) -> List[ChunkBoundary]:
        """Detect speaker changes and create chunks."""
        try:
            from pyannote.audio import Pipeline
        except ImportError:
            raise ImportError("Install pyannote.audio: pip install pyannote.audio")

        logger.info(f"Speaker diarization chunking: {source_path}")

        # Load model (downloads on first use)
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.0",
            use_auth_token=kwargs.get("hf_token"),
        )

        # Process audio
        diarization = pipeline(source_path, num_speakers=kwargs.get("num_speakers"))

        # Extract speaker turns
        boundaries = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            boundaries.append(
                ChunkBoundary(
                    start_sec=turn.start,
                    end_sec=turn.end,
                    boundary_type=f"speaker_change:{speaker}",
                    confidence=0.9,
                )
            )

        return self._enforce_duration_limits(boundaries, min_duration_sec, max_duration_sec)


class HybridAudioChunking(ChunkingStrategy):
    """Combines silence detection + speaker diarization."""

    def __init__(self, use_diarization: bool = True, use_silence: bool = True):
        self.use_diarization = use_diarization
        self.use_silence = use_silence
        self.silence_strategy = SilenceDetectionChunking()
        self.diarization_strategy = SpeakerDiarizationChunking()

    async def chunk(
        self,
        source_path: str,
        min_duration_sec: float = 15.0,
        max_duration_sec: float = 90.0,
        **kwargs
    ) -> List[ChunkBoundary]:
        """Combine silence + speaker detection for robust chunking."""

        boundaries = []

        # Get speaker boundaries if available
        if self.use_diarization:
            try:
                speaker_boundaries = await self.diarization_strategy.chunk(
                    source_path, min_duration_sec, max_duration_sec, **kwargs
                )
                boundaries.extend(speaker_boundaries)
                logger.info(f"Found {len(speaker_boundaries)} speaker segments")
            except Exception as e:
                logger.warning(f"Speaker diarization failed: {e}, falling back to silence")

        # Get silence boundaries
        if self.use_silence:
            try:
                silence_boundaries = await self.silence_strategy.chunk(
                    source_path, min_duration_sec, max_duration_sec, **kwargs
                )
                # Merge silence boundaries with speaker boundaries
                boundaries = self._merge_boundaries(boundaries, silence_boundaries)
                logger.info(f"Found {len(silence_boundaries)} silence-based segments")
            except Exception as e:
                logger.warning(f"Silence detection failed: {e}")

        # Sort by time and remove duplicates
        boundaries.sort(key=lambda b: b.start_sec)
        boundaries = self._deduplicate_boundaries(boundaries)

        return self._enforce_duration_limits(boundaries, min_duration_sec, max_duration_sec)

    def _merge_boundaries(
        self, primary: List[ChunkBoundary], secondary: List[ChunkBoundary]
    ) -> List[ChunkBoundary]:
        """Merge boundaries from multiple strategies."""
        # Simple merge: combine and let enforce_duration_limits clean up
        return primary + secondary

    def _deduplicate_boundaries(self, boundaries: List[ChunkBoundary]) -> List[ChunkBoundary]:
        """Remove duplicate/overlapping boundaries."""
        if not boundaries:
            return []

        unique = [boundaries[0]]
        for boundary in boundaries[1:]:
            # Skip if overlaps with previous (within 1 second)
            if boundary.start_sec - unique[-1].end_sec < 1.0:
                continue
            unique.append(boundary)

        return unique


# ============================================================================
# VIDEO CHUNKING STRATEGIES
# ============================================================================

class SceneDetectionChunking(ChunkingStrategy):
    """Scene detection using PySceneDetect library."""

    def __init__(self, threshold: float = 8.0):
        self.threshold = threshold  # 1-30 (higher = fewer scenes)

    async def chunk(
        self,
        source_path: str,
        min_duration_sec: float = 2.0,
        max_duration_sec: float = 300.0,
        **kwargs
    ) -> List[ChunkBoundary]:
        """Detect scene boundaries using adaptive detector."""
        try:
            from scenedetect import detect, AdaptiveDetector
        except ImportError:
            raise ImportError("Install scenedetect: pip install scenedetect[opencv]")

        logger.info(f"Scene detection chunking: {source_path}")

        # Detect scenes using adaptive detector
        scenes = detect(
            source_path,
            AdaptiveDetector(threshold=self.threshold),
        )

        # Convert scenes to boundaries
        boundaries = []
        for i in range(len(scenes) - 1):
            start_sec = scenes[i].get_seconds()
            end_sec = scenes[i + 1].get_seconds()

            boundaries.append(
                ChunkBoundary(
                    start_sec=start_sec,
                    end_sec=end_sec,
                    boundary_type="scene_cut",
                    confidence=0.85,
                )
            )

        return self._enforce_duration_limits(boundaries, min_duration_sec, max_duration_sec)


class AdaptiveKeyframeChunking(ChunkingStrategy):
    """Adaptive keyframe extraction based on visual entropy."""

    def __init__(self, max_frames: int = 30, entropy_threshold: float = 5.0):
        self.max_frames = max_frames
        self.entropy_threshold = entropy_threshold

    async def chunk(
        self,
        source_path: str,
        min_duration_sec: float = 2.0,
        max_duration_sec: float = 300.0,
        **kwargs
    ) -> List[ChunkBoundary]:
        """Extract adaptive keyframes based on visual complexity."""
        try:
            import cv2
            import numpy as np
        except ImportError:
            raise ImportError("Install opencv: pip install opencv-python")

        logger.info(f"Adaptive keyframe chunking: {source_path}")

        cap = cv2.VideoCapture(source_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / fps

        # Divide video into windows
        window_duration = 5.0  # 5-second windows
        window_frames = int(fps * window_duration)

        boundaries = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # At window boundaries, mark as keyframe
            if frame_idx % window_frames == 0 and frame_idx > 0:
                time_sec = frame_idx / fps
                boundaries.append(
                    ChunkBoundary(
                        start_sec=max(0, time_sec - window_duration),
                        end_sec=time_sec,
                        boundary_type="keyframe",
                        confidence=0.9,
                    )
                )

            frame_idx += 1

        cap.release()

        # Add final chunk
        if boundaries:
            boundaries.append(
                ChunkBoundary(
                    start_sec=boundaries[-1].end_sec,
                    end_sec=total_duration,
                    boundary_type="keyframe",
                    confidence=0.9,
                )
            )

        return self._enforce_duration_limits(boundaries, min_duration_sec, max_duration_sec)


class HybridVideoChunking(ChunkingStrategy):
    """Combines scene detection + adaptive keyframe extraction."""

    def __init__(self, use_scenes: bool = True, use_keyframes: bool = True):
        self.use_scenes = use_scenes
        self.use_keyframes = use_keyframes
        self.scene_strategy = SceneDetectionChunking()
        self.keyframe_strategy = AdaptiveKeyframeChunking()

    async def chunk(
        self,
        source_path: str,
        min_duration_sec: float = 2.0,
        max_duration_sec: float = 300.0,
        **kwargs
    ) -> List[ChunkBoundary]:
        """Combine scene + keyframe detection."""

        boundaries = []

        if self.use_scenes:
            try:
                scene_boundaries = await self.scene_strategy.chunk(
                    source_path, min_duration_sec, max_duration_sec
                )
                boundaries.extend(scene_boundaries)
                logger.info(f"Found {len(scene_boundaries)} scenes")
            except Exception as e:
                logger.warning(f"Scene detection failed: {e}")

        if self.use_keyframes:
            try:
                keyframe_boundaries = await self.keyframe_strategy.chunk(
                    source_path, min_duration_sec, max_duration_sec
                )
                # Merge with scene boundaries
                boundaries = self._merge_boundaries(boundaries, keyframe_boundaries)
                logger.info(f"Found {len(keyframe_boundaries)} keyframes")
            except Exception as e:
                logger.warning(f"Keyframe extraction failed: {e}")

        # Sort and deduplicate
        boundaries.sort(key=lambda b: b.start_sec)
        boundaries = self._deduplicate_boundaries(boundaries)

        return self._enforce_duration_limits(boundaries, min_duration_sec, max_duration_sec)

    def _merge_boundaries(
        self, primary: List[ChunkBoundary], secondary: List[ChunkBoundary]
    ) -> List[ChunkBoundary]:
        """Merge boundaries, preferring scene cuts over keyframes."""
        return primary + secondary

    def _deduplicate_boundaries(self, boundaries: List[ChunkBoundary]) -> List[ChunkBoundary]:
        """Remove overlapping boundaries, keeping higher-confidence ones."""
        if not boundaries:
            return []

        # Sort by start time, then by confidence (descending)
        boundaries.sort(key=lambda b: (b.start_sec, -b.confidence))

        unique = [boundaries[0]]
        for boundary in boundaries[1:]:
            # Skip if starts before previous one ends (overlap)
            if boundary.start_sec < unique[-1].end_sec:
                continue
            unique.append(boundary)

        return unique
```

---

## FILE 3: Integration with AudioProcessor

Update `/ragcore/modules/multimodal/processors/audio_processor.py`:

```python
# ADD TO AudioProcessor class

async def process(
    self,
    content: MultiModalContent,
    session_id: UUID,
    chunking_strategy: Optional['HybridAudioChunking'] = None,
    **kwargs,
) -> ProcessingResult:
    """Process audio with intelligent chunking."""
    from ragcore.modules.multimodal.chunking.strategies import HybridAudioChunking

    start_time = time.time()
    await self._log_processing_start(content)

    # Validate
    if not self.validate_content(content):
        return ProcessingResult(
            success=False,
            modality=ModuleType.AUDIO,
            error_message="Invalid audio content",
        )

    try:
        # Use hybrid chunking by default
        if chunking_strategy is None:
            chunking_strategy = HybridAudioChunking(
                use_diarization=kwargs.get("use_diarization", True),
                use_silence=kwargs.get("use_silence", True),
            )

        # Save audio temporarily for chunking (if needed)
        temp_audio_path = await self._save_temp_audio(content.raw_content)

        # Get chunk boundaries
        chunk_boundaries = await chunking_strategy.chunk(
            temp_audio_path,
            min_duration_sec=kwargs.get("min_chunk_sec", 15),
            max_duration_sec=kwargs.get("max_chunk_sec", 90),
            hf_token=kwargs.get("hf_token"),
            num_speakers=kwargs.get("num_speakers"),
        )

        logger.info(f"Created {len(chunk_boundaries)} audio chunks")

        # Process each boundary
        chunks = []
        for boundary_idx, boundary in enumerate(chunk_boundaries):
            chunk = await self._process_chunk_boundary(
                content,
                session_id,
                boundary,
                boundary_idx,
            )
            chunks.append(chunk)

        return ProcessingResult(
            success=True,
            modality=ModuleType.AUDIO,
            chunks=chunks,
            extracted_text="\n\n".join([c.content for c in chunks]),
            tokens_used=sum(self.estimate_tokens_used_for_chunk(c) for c in chunks),
            processing_time_ms=(time.time() - start_time) * 1000,
        )

    except Exception as e:
        logger.error(f"Audio processing failed: {e}")
        return ProcessingResult(
            success=False,
            modality=ModuleType.AUDIO,
            error_message=str(e),
            processing_time_ms=(time.time() - start_time) * 1000,
        )

async def _process_chunk_boundary(
    self,
    content: MultiModalContent,
    session_id: UUID,
    boundary,
    boundary_idx: int,
) -> MultiModalChunk:
    """Process single chunk boundary."""
    from ragcore.modules.multimodal.models import TemporalMetadata, BoundaryType

    # Extract audio segment
    segment_audio = await self._extract_audio_segment(
        content.raw_content,
        start_ms=boundary.start_sec * 1000,
        end_ms=boundary.end_sec * 1000,
    )

    # Transcribe
    transcription = await self._transcribe_segment(segment_audio, content.metadata.language or "en-US")

    # Create chunk with temporal metadata
    chunk = MultiModalChunk(
        id=uuid4(),
        session_id=session_id,
        modality=ModuleType.AUDIO,
        content=transcription["text"],
        embedding=[],
        metadata=content.metadata,
        source_index=boundary_idx,
        confidence_score=transcription.get("confidence", 0.9),
        temporal_metadata=TemporalMetadata(
            start_time_sec=boundary.start_sec,
            end_time_sec=boundary.end_sec,
            boundary_type=BoundaryType(boundary.boundary_type.split(":")[0]),
            boundary_confidence=boundary.confidence,
            speaker_id=boundary.boundary_type.split(":")[1] if ":" in boundary.boundary_type else None,
            speech_rate_wpm=await self._estimate_speech_rate(transcription["text"], boundary),
        ),
    )

    return chunk

async def _extract_audio_segment(
    self, audio_bytes: bytes, start_ms: float, end_ms: float
) -> bytes:
    """Extract time range from audio."""
    import librosa
    import soundfile as sf
    import numpy as np
    import io

    # Load full audio
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)

    # Extract segment by sample indices
    start_sample = int(start_ms * sr / 1000)
    end_sample = int(end_ms * sr / 1000)
    segment = y[start_sample:end_sample]

    # Convert back to bytes
    output = io.BytesIO()
    sf.write(output, segment, sr, format='WAV')
    return output.getvalue()

async def _estimate_speech_rate(self, text: str, boundary) -> float:
    """Estimate words per minute for segment."""
    duration_sec = boundary.end_sec - boundary.start_sec
    word_count = len(text.split())
    wpm = (word_count / duration_sec) * 60 if duration_sec > 0 else 0
    return wpm
```

---

## FILE 4: Updated Database Schema (Alembic Migration)

Create `/ragcore/db/migrations/versions/007_add_temporal_metadata.py`:

```python
"""Add temporal metadata to multimodal_chunks."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '007'
down_revision = '006'
branch_labels = None
depends_on = None


def upgrade():
    """Add temporal_metadata JSON column."""
    op.add_column(
        'multimodal_chunks',
        sa.Column(
            'temporal_metadata',
            postgresql.JSON,
            nullable=True,
            doc='Temporal data: start_sec, end_sec, boundary_type, etc.',
        ),
    )

    # Add indexes for temporal queries
    op.create_index(
        'idx_chunks_temporal_range',
        'multimodal_chunks',
        [
            sa.func.to_jsonb(sa.column('temporal_metadata')).op('->')('start_time_sec'),
            sa.func.to_jsonb(sa.column('temporal_metadata')).op('->')('end_time_sec'),
        ],
        postgresql_using='btree',
    )

    # Add boundary_type index
    op.create_index(
        'idx_chunks_boundary_type',
        'multimodal_chunks',
        [sa.func.to_jsonb(sa.column('temporal_metadata')).op('->>')('boundary_type')],
        postgresql_using='hash',
    )


def downgrade():
    """Remove temporal metadata columns."""
    op.drop_index('idx_chunks_boundary_type')
    op.drop_index('idx_chunks_temporal_range')
    op.drop_column('multimodal_chunks', 'temporal_metadata')
```

---

## Updated requirements.txt

```txt
# ... existing requirements ...

# Audio Processing (Smart Chunking)
librosa==0.10.0
pyannote.audio==3.0.1
pyannote.core==5.0.0
soundfile==0.12.1
scipy>=1.10

# Video Processing (Smart Chunking)
scenedetect[opencv]==0.6.1
opencv-python==4.8.1.78
ffmpeg-python==0.2.1
av==11.0.0

# Additional for chunk processing
numpy>=1.24
```

---

## Testing Setup

Create `/tests/test_chunking_strategies.py`:

```python
"""Tests for smart chunking strategies."""

import pytest
import asyncio
from pathlib import Path
from ragcore.modules.multimodal.chunking.strategies import (
    SilenceDetectionChunking,
    HybridAudioChunking,
    SceneDetectionChunking,
    HybridVideoChunking,
)


@pytest.mark.asyncio
async def test_silence_detection():
    """Test silence-based chunking."""
    strategy = SilenceDetectionChunking(threshold_multiplier=0.5)

    # Use sample audio (download/generate for test)
    boundaries = await strategy.chunk(
        "test_data/sample_audio.mp3",
        min_duration_sec=10,
        max_duration_sec=60,
    )

    assert len(boundaries) > 0
    assert boundaries[0].start_sec == 0.0
    assert boundaries[-1].end_sec > boundaries[0].start_sec


@pytest.mark.asyncio
async def test_hybrid_audio():
    """Test hybrid audio chunking."""
    strategy = HybridAudioChunking(use_diarization=False, use_silence=True)

    boundaries = await strategy.chunk(
        "test_data/podcast_30s.mp3",
        min_duration_sec=5,
        max_duration_sec=60,
    )

    # Verify chunk continuity
    for i in range(len(boundaries) - 1):
        assert boundaries[i].end_sec <= boundaries[i + 1].start_sec


@pytest.mark.asyncio
async def test_scene_detection():
    """Test scene detection for video."""
    strategy = SceneDetectionChunking(threshold=8.0)

    boundaries = await strategy.chunk(
        "test_data/sample_video.mp4",
        min_duration_sec=2,
        max_duration_sec=60,
    )

    assert len(boundaries) > 0
    assert all(b.boundary_type == "scene_cut" for b in boundaries)


@pytest.mark.asyncio
async def test_hybrid_video():
    """Test hybrid video chunking."""
    strategy = HybridVideoChunking(use_scenes=True, use_keyframes=True)

    boundaries = await strategy.chunk(
        "test_data/sample_video.mp4",
        min_duration_sec=2,
        max_duration_sec=60,
    )

    assert len(boundaries) > 0
```

---

**Status**: Ready for Phase 0 Sprint Implementation
**Priority**: High (blocking other multimodal features)
**Estimated Implementation Time**: 2-3 weeks
**Dependencies**: Libraries must be installed via requirements.txt
