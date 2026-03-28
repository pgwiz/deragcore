# Smart Chunking Strategies for Audio & Video: Research Summary

**Date**: 2026-03-28
**Status**: Research & Reference Documentation
**Target Integration**: Phase 0 Critical Fixes - Smart Chunking Sprint

---

## Executive Summary

This research synthesizes best practices for intelligent audio and video chunking, designed for integration into the RAGCORE multimodal system. Key findings:

- **Audio**: Hybrid approach combining silence detection + speaker diarization for semantic boundaries
- **Video**: Scene-aware extraction with adaptive frame sampling and motion detection
- **Integration**: Extend MultiModalChunk to include temporal metadata (start_time, end_time, boundary_type)
- **Performance**: Audio ~2-5x real-time, Video ~5-10x real-time for extraction (varies by codec)
- **Recommended Stack**: librosa + pyannote.audio (audio), OpenCV + PySceneDetect (video)

---

# PART 1: AUDIO CHUNKING STRATEGIES

## 1.1 Core Techniques

### A. Silence Detection (Foundation Layer)

**Purpose**: Segment audio into continuous speech segments

**Algorithm: Threshold-based Energy Detection**
```
1. Compute short-time energy (frame window: 20-30ms, stride: 10ms)
2. Calculate dynamic threshold: mean(energy) * multiplier (0.4-0.6)
3. Identify silence frames where energy < threshold
4. Merge adjacent silence regions (min gap: 200-500ms)
5. Extract speech segments between silence regions
```

**Complexity**: O(n) where n = number of frames (one pass)
**Processing Time**: 1 hour audio ≈ 100-300ms with librosa

**Libraries & Implementations**:

| Library | Version | Pros | Cons | Speed |
|---------|---------|------|------|-------|
| **librosa** | 0.10+ | Easy API, energy-based, filters | Single threshold (less accurate) | ~1ms per sec |
| **pyannote.audio** | 3.0+ | ML-based (trained model), accurate | Slower, needs GPU for speed | ~0.5s per sec |
| **webrtcvad** | 2.0.11 | Fast, trained on speech data | Older codec support only | ~0.1ms per sec |
| **silero-vad** | 5.0+ | Real-time capable, no GPU req | Requires PyTorch | ~0.2ms per sec |

**Code Example (librosa)**:
```python
import librosa
import numpy as np

def detect_silence_regions(audio_path, sr=16000, silence_threshold=0.01, min_duration_ms=300):
    """Detect silence regions using energy-based method."""
    y, sr = librosa.load(audio_path, sr=sr)

    # Compute energy in dB
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_S = librosa.power_to_db(S, ref=np.max)
    energy = np.mean(log_S, axis=0)

    # Dynamic thresholding
    threshold = np.mean(energy) + np.std(energy) * (-5)  # 5 std below mean
    silence_mask = energy < threshold

    # Convert frame indices to time (hop_length=512 by default)
    hop_length = 512
    silence_frames = np.where(silence_mask)[0]
    silence_times = librosa.frames_to_time(silence_frames, sr=sr, hop_length=hop_length)

    # Merge adjacent silence regions
    min_frames = int(min_duration_ms * sr / 1000 / hop_length)
    speech_segments = []

    # Extract speech regions (inverse of silence)
    return speech_segments  # List of (start_ms, end_ms) tuples
```

---

### B. Speaker Diarization (Semantic Layer)

**Purpose**: Identify speaker turns and boundaries for natural chunk breaks

**Algorithm: Clustering-based Approach**
```
1. Extract speaker embeddings (every 500-1000ms window)
2. Cluster embeddings into N speakers (unsupervised)
3. Detect speaker change boundaries
4. Merge short speaker segments (<1 sec) with neighboring speaker
5. Return speaker changes as chunk boundaries
```

**Complexity**: O(n²) in worst case for clustering (k-means), ~O(n) with approximations
**Processing Time**: 1 hour audio ≈ 5-30 seconds (depends on algorithm)

**Libraries & Implementations**:

| Library | Version | Approach | Pros | Cons | Speed |
|---------|---------|----------|------|------|-------|
| **pyannote.audio** | 3.0+ | Neural clustering | Accurate, end-to-end | Slow (~1 sec per sec) | ~3600s/hour |
| **resemblyzer** | 0.1.1 | Embedding-based | Fast clustering | Older, maintenance? | ~300s/hour |
| **speechbrain** | 0.5+ | Neural speakers | Multi-language | Less documented | ~600s/hour |

**Code Example (pyannote.audio)**:
```python
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization

def speaker_diarization_chunking(audio_path, num_speakers=None):
    """Segment audio by speaker changes using neural diarization."""
    # Initialize pipeline (auto-downloads model)
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.0",
        use_auth_token="hf_token"  # Hugging Face token
    )

    # Process audio
    diarization = pipeline(audio_path, num_speakers=num_speakers)

    # Extract speaker turns (segments where same speaker speaks)
    speaker_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments.append({
            "start_sec": turn.start,
            "end_sec": turn.end,
            "speaker": speaker,
        })

    # Merge consecutive same-speaker segments (if gap < 500ms)
    merged = merge_adjacent_segments(speaker_segments, merge_gap_sec=0.5)
    return merged
```

**Hybrid Strategy** (Recommended):
```python
async def intelligent_audio_chunking(audio_bytes, session_id, language="en-US"):
    """
    Smart chunking combining silence + speaker diarization.

    Strategy:
    1. Primary: Speaker diarization (semantic boundaries)
    2. Secondary: Silence detection within speaker turns
    3. Max chunk: 60-90 seconds (context window)
    4. Min chunk: 15-20 seconds (meaningful units)
    """

    # Step 1: Detect speaker changes
    speaker_segments = await speaker_diarization_chunking(audio_path)

    # Step 2: Within each speaker segment, find silence boundaries
    chunks = []
    for speaker_seg in speaker_segments:
        # Get sub-segment of audio
        audio_segment = extract_audio_segment(
            audio_path,
            start_ms=speaker_seg["start_sec"] * 1000,
            end_ms=speaker_seg["end_sec"] * 1000
        )

        # Detect silence within this speaker's turn
        silence_breaks = detect_silence_regions(
            audio_segment,
            silence_threshold=0.02,
            min_duration_ms=500
        )

        # Create sub-chunks at silence boundaries
        sub_chunks = create_chunks_at_breaks(
            silence_breaks,
            min_duration_sec=15,
            max_duration_sec=90
        )
        chunks.extend(sub_chunks)

    return chunks
```

---

### C. Timestamp-based Segmentation (Pragmatic Layer)

**Purpose**: Fallback when semantic boundaries unavailable; useful for podcasts/lectures

**Strategy**:
```
1. For audio without speech: fixed-size chunks (e.g., 60 seconds)
2. For music: beat detection + 4-bar segments
3. For podcasts: chapter markers (ID3 tags) if available
4. Default: 30-90 second chunks based on content type
```

**Code Example**:
```python
def timestamp_based_chunking(duration_seconds, chunk_duration_sec=60):
    """Create fixed-size chunks with optional adjustments."""
    chunks = []
    for start_sec in range(0, int(duration_seconds), chunk_duration_sec):
        end_sec = min(start_sec + chunk_duration_sec, duration_seconds)
        chunks.append({
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": end_sec - start_sec,
            "boundary_type": "timestamp"  # vs "silence" or "speaker_change"
        })
    return chunks
```

---

## 1.2 Recommended Audio Stack

### Libraries to Add to requirements.txt

```txt
# Audio Processing (Smart Chunking)
librosa==0.10.0          # Core DSP: silence detection, beat tracking
pyannote.audio==3.0.1    # Speaker diarization + segmentation (ML-based)
silero-vad==5.0          # Real-time voice activity detection (optional faster alt)

# Audio I/O (if not included)
soundfile==0.12.1        # WAV/FLAC I/O
scipy>=1.10              # Signal processing (used by librosa)

# Beat Detection (optional, for music chunking)
essentia==2.1.0          # Music analysis, beat tracking
```

### Performance Characteristics

| Task | Duration | Time | Speed Factor |
|------|----------|------|--------------|
| Silence Detection | 1 hour | 0.1-0.3s | ~0.0001x |
| Speaker Diarization | 1 hour | 30-120s | ~0.008-0.03x |
| Speech-to-Text | 1 hour | 360-3600s | 0.1-1x (real-time) |
| **Total Pipeline** | 1 hour | 360-3700s | ~0.1-1x |

**Recommendation**: Process audio I/O asynchronously; run diarization in background task queue

---

# PART 2: VIDEO CHUNKING STRATEGIES

## 2.1 Core Techniques

### A. Scene Detection (Semantic Layer)

**Purpose**: Segment video at natural scene/shot boundaries

**Algorithms**:

#### 1. **Frame Difference (Fast)**
```
For each consecutive frame pair:
  1. Compute histogram difference (color distribution)
  2. If diff > threshold: scene boundary
  3. Threshold ~0.15-0.25 (empirical)
```
- **Complexity**: O(n) frames, ~5-10ms per frame
- **Accuracy**: ~70-80% (high false positives)

#### 2. **Optical Flow (Accurate)**
```
For each frame pair:
  1. Compute optical flow vectors (motion)
  2. Identify static camera vs motion
  3. If static camera + large motion change → scene boundary
  4. If motion becomes near-zero → potential cut
```
- **Complexity**: O(n) frames, ~50-100ms per frame
- **Accuracy**: ~85-95%

#### 3. **Content-based (ML-based)**
```
1. Extract CNN features (ResNet-50 layer 4)
2. Compute cosine distance between consecutive frames
3. If dist > threshold: potential boundary
4. Verify with optical flow (reduce false positives)
```
- **Complexity**: O(n) frames, ~100-200ms per frame (with GPU: ~20ms)
- **Accuracy**: ~90-98%

**Code Example (PySceneDetect)**:
```python
from scenedetect import detect, AdaptiveDetector, StatisticalDetector
import cv2

def detect_scenes(video_path, adaptive_threshold=8.0):
    """Detect scene boundaries using adaptive threshold."""
    # Adaptive detector: tracks frame histogram statistics
    # Threshold: 8.0 = 8% change in histogram
    scenes = detect(video_path, AdaptiveDetector(threshold=adaptive_threshold))

    # Convert to timestamps
    scene_boundaries = [(s.get_seconds()) for s in scenes]
    return scene_boundaries

def detect_scenes_statistical(video_path, threshold=27.0):
    """Detect scenes using statistical method (faster)."""
    scenes = detect(video_path, StatisticalDetector(threshold=threshold))
    return [(s.get_seconds()) for s in scenes]
```

**Complexity Comparison**:

| Method | Speed | Accuracy | FP Rate | Dependencies |
|--------|-------|----------|--------|--------------|
| Frame Diff | 10ms/frame | 70% | High | OpenCV |
| Optical Flow | 80ms/frame | 90% | Medium | OpenCV |
| CNN-based | 100ms/frame | 95% | Low | OpenCV + TF/PyTorch |
| PySceneDetect | 5ms/frame | 85% | Medium | (optimized) |

---

### B. Keyframe Extraction (Efficiency Layer)

**Purpose**: Select representative frames without processing every frame

**Strategy 1: Uniform Sampling**
```
Sample every Nth frame where N = fps / target_fps
E.g., 30fps video, want ~1 frame/sec → N=30
```
- Simple but misses important moments
- Useful for time-constrained scenarios

**Strategy 2: Adaptive Sampling (Recommended)**
```
1. Divide video into temporal windows (e.g., 5s, 10s)
2. Within each window:
   - Select frame with max entropy (visual variation)
   - Or select first frame if static content
   - Or use motion-weighted selection
3. Merge keyframes from all windows
```

**Strategy 3: Motion-based Selection**
```
1. Compute frame-to-frame optical flow magnitude
2. Select frames with motion peaks (significant changes)
3. Also include I-frames (intra-frames from codec)
4. Sample uniformly from low-motion regions
```

**Code Example (Adaptive)**:
```python
import cv2
import numpy as np

def extract_adaptive_keyframes(video_path, window_duration_sec=5, max_frames=30):
    """Extract adaptive keyframes using entropy-based selection."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    window_frames = int(fps * window_duration_sec)
    keyframe_indices = []

    # Process video in windows
    for window_start in range(0, total_frames, window_frames):
        window_end = min(window_start + window_frames, total_frames)

        # Track entropy and motion
        max_entropy = -1
        max_entropy_idx = window_start
        prev_frame = None

        for frame_idx in range(window_start, window_end):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to grayscale and compute entropy
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            entropy = compute_entropy(gray)

            if entropy > max_entropy:
                max_entropy = entropy
                max_entropy_idx = frame_idx

        keyframe_indices.append(max_entropy_idx)

    # Limit to max_frames
    step = max(1, len(keyframe_indices) // max_frames)
    return keyframe_indices[::step][:max_frames]

def compute_entropy(image):
    """Compute Shannon entropy of image (visual complexity)."""
    hist, _ = np.histogram(image.ravel(), 256, [0, 256])
    hist = hist[hist > 0]  # Non-zero bins
    p = hist / hist.sum()
    entropy = -np.sum(p * np.log2(p))
    return entropy
```

---

### C. Shot Boundary Detection (Precision Layer)

**Purpose**: Distinguish between different shot types (cuts, fades, dissolves)

**Types & Detection**:

| Type | Characteristic | Detection |
|------|---|---|
| **Cut** | Instant transition | Frame diff > 0.4 |
| **Fade** | Gradual to/from black | Increasing then decreasing luma + high diff |
| **Dissolve** | Cross-fade between scenes | Smooth increase in diff over 0.5-2s |
| **Wipe** | Moving boundary | Spatial gradient detection |

**Code Example**:
```python
def detect_shot_boundaries(video_path, cut_threshold=0.4, fade_threshold=0.3):
    """Detect different shot boundary types."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    boundaries = {"cuts": [], "fades": [], "dissolves": []}
    prev_hist = None
    fade_start = None

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Compute histogram
        hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)

            # Cut detection
            if diff > cut_threshold:
                boundaries["cuts"].append(frame_idx / fps)

            # Fade detection (requires temporal tracking)
            elif fade_threshold < diff < cut_threshold:
                if fade_start is None:
                    fade_start = frame_idx
            elif fade_start is not None:
                boundaries["fades"].append((fade_start / fps, frame_idx / fps))
                fade_start = None

        prev_hist = hist
        frame_idx += 1

    cap.release()
    return boundaries
```

---

## 2.2 Recommended Video Stack

### Libraries to Add

```txt
# Video Processing (Scene Detection & Keyframe Extraction)
opencv-python==4.8.1.78  # Core video I/O, frame processing
scenedetect[opencv]==0.6.1  # Scene detection (PySceneDetect)
ffmpeg-python==0.2.1     # FFmpeg wrapper for advanced operations
av==11.0.0               # PyAV for codec-aware extraction

# Optional: Advanced analysis
torch>=2.0               # For CNN-based scene detection (optional)
torchvision>=0.15        # Vision models (optional)
```

### Performance Characteristics

| Task | 1-Hour Video (30fps, 720p) | Time | Speed Factor |
|------|---|---|---|
| Frame Extraction (30fps) | 108,000 frames | 30-60s | 0.008-0.016x |
| Adaptive Keyframes (60 total) | 60 frames | 1-5s | 0.0003-0.0013x |
| Scene Detection (adaptive) | 108,000 frames | 180-600s | 0.05-0.166x |
| **Total Pipeline** | All above | 210-660s | **0.058-0.183x** |

**Practical**: For 1 hour video, expect 3.5-11 minutes total processing

---

# PART 3: INTEGRATION WITH MULTIMODAL SYSTEM

## 3.1 Extended Data Models

### A. Temporal Metadata Extension

**Current State**: MultiModalChunk has `source_index` (frame/segment number)

**Proposed Addition**:
```python
@dataclass
class TemporalMetadata:
    """Temporal information for audio/video chunks."""
    start_time_sec: float  # Absolute time in source
    end_time_sec: float    # Absolute time in source
    duration_sec: float = field(init=False)  # Computed
    boundary_type: str = "unknown"  # "silence", "speaker_change", "scene_cut", "keyframe"
    confidence: float = 1.0  # Boundary detection confidence
    speaker_id: Optional[str] = None  # For audio diarization
    visual_complexity: Optional[float] = None  # For video (entropy score)
    motion_magnitude: Optional[float] = None  # For video (optical flow magnitude)

    def __post_init__(self):
        self.duration_sec = self.end_time_sec - self.start_time_sec
```

**Extended MultiModalChunk**:
```python
@dataclass
class MultiModalChunk:
    # ... existing fields ...

    # NEW: Temporal information
    temporal_metadata: Optional[TemporalMetadata] = None

    # Helper method
    def get_time_range(self) -> tuple[float, float]:
        """Return (start_sec, end_sec) for easy slicing."""
        if self.temporal_metadata:
            return (self.temporal_metadata.start_time_sec,
                    self.temporal_metadata.end_time_sec)
        return (0.0, 0.0)
```

---

### B. Chunking Pipeline Interface

```python
from abc import ABC, abstractmethod
from typing import List, AsyncIterator

class AudioChunkingStrategy(ABC):
    """Base class for audio chunking strategies."""

    @abstractmethod
    async def chunk(
        self,
        audio_path: str,
        min_duration_sec: float = 15,
        max_duration_sec: float = 90,
        **kwargs
    ) -> List[tuple[float, float]]:  # List of (start_sec, end_sec)
        """Return chunk boundaries."""
        pass

class VideoChunkingStrategy(ABC):
    """Base class for video chunking strategies."""

    @abstractmethod
    async def chunk(
        self,
        video_path: str,
        max_chunks: int = 30,
        **kwargs
    ) -> List[tuple[float, float]]:  # List of (start_sec, end_sec)
        """Return chunk boundaries."""
        pass

class HybridAudioChunking(AudioChunkingStrategy):
    """Combines silence detection + speaker diarization."""

    def __init__(self, use_diarization=True, use_silence=True):
        self.use_diarization = use_diarization
        self.use_silence = use_silence

    async def chunk(self, audio_path, min_duration_sec=15, max_duration_sec=90, **kwargs):
        boundaries = []

        if self.use_diarization:
            speaker_segs = await self._speaker_diarization(audio_path)
            boundaries.extend(speaker_segs)

        if self.use_silence:
            silence_segs = await self._silence_detection(audio_path)
            boundaries = self._merge_boundaries(boundaries, silence_segs)

        return self._enforce_duration_limits(boundaries, min_duration_sec, max_duration_sec)

class AdaptiveVideoChunking(VideoChunkingStrategy):
    """Combines scene detection + adaptive keyframe selection."""

    async def chunk(self, video_path, max_chunks=30, **kwargs):
        # 1. Detect scene boundaries
        scenes = await self._detect_scenes(video_path)

        # 2. Extract adaptive keyframes within scenes
        keyframes = await self._extract_keyframes_per_scene(video_path, scenes)

        # 3. Create chunks from scene + keyframe boundaries
        chunk_boundaries = self._merge_scene_and_keyframes(scenes, keyframes)

        # 4. Limit total chunks
        return chunk_boundaries[:max_chunks]
```

---

### C. Integration with AudioProcessor & VideoProcessor

**Updated AudioProcessor**:
```python
async def process(
    self,
    content: MultiModalContent,
    session_id: UUID,
    chunking_strategy: AudioChunkingStrategy = None,
    **kwargs,
) -> ProcessingResult:
    """Process audio with intelligent chunking."""

    # Use hybrid chunking by default
    if chunking_strategy is None:
        chunking_strategy = HybridAudioChunking()

    # Get chunk boundaries (in seconds)
    chunk_boundaries = await chunking_strategy.chunk(
        audio_path=content.storage_path or "temp_audio.wav",
        min_duration_sec=kwargs.get("min_chunk_sec", 15),
        max_duration_sec=kwargs.get("max_chunk_sec", 90),
    )

    # For each boundary, transcribe and create chunk
    chunks = []
    for start_sec, end_sec in chunk_boundaries:
        # Extract audio segment
        segment_audio = await self._extract_audio_segment(
            content.raw_content,
            start_ms=start_sec * 1000,
            end_ms=end_sec * 1000,
        )

        # Transcribe segment
        transcription = await self._transcribe_segment(segment_audio)

        # Create chunk with temporal metadata
        chunk = MultiModalChunk(
            id=uuid4(),
            session_id=session_id,
            modality=ModuleType.AUDIO,
            content=transcription["text"],
            embedding=[],
            metadata=content.metadata,
            source_index=len(chunks),
            confidence_score=transcription.get("confidence", 0.9),
            temporal_metadata=TemporalMetadata(
                start_time_sec=start_sec,
                end_time_sec=end_sec,
                boundary_type=transcription.get("boundary_type", "segment"),
                speaker_id=transcription.get("speaker_id"),
            ),
        )
        chunks.append(chunk)

    return ProcessingResult(
        success=True,
        modality=ModuleType.AUDIO,
        chunks=chunks,
        extracted_text="\n\n".join([c.content for c in chunks]),
        tokens_used=sum(self.estimate_tokens_used(c) for c in chunks),
    )
```

**Updated VideoProcessor**:
```python
async def process(
    self,
    content: MultiModalContent,
    session_id: UUID,
    chunking_strategy: VideoChunkingStrategy = None,
    **kwargs,
) -> ProcessingResult:
    """Process video with intelligent scene-aware chunking."""

    if chunking_strategy is None:
        chunking_strategy = AdaptiveVideoChunking()

    # Get chunk boundaries
    chunk_boundaries = await chunking_strategy.chunk(
        video_path=content.storage_path,
        max_chunks=kwargs.get("max_frames", 30),
    )

    chunks = []

    # Extract and analyze frames at chunk boundaries
    for idx, (start_sec, end_sec) in enumerate(chunk_boundaries):
        # Extract frame at start of chunk (representative)
        frame_data = await self._extract_frame_at_timestamp(
            content.raw_content,
            start_sec,
        )

        # Analyze with vision API
        analysis = await self._analyze_frame(frame_data)

        # Create chunk with temporal bounds
        chunk = MultiModalChunk(
            id=uuid4(),
            session_id=session_id,
            modality=ModuleType.VIDEO,
            content=analysis.get("text", ""),
            embedding=[],
            metadata=content.metadata,
            source_index=idx,
            confidence_score=analysis.get("confidence", 0.85),
            temporal_metadata=TemporalMetadata(
                start_time_sec=start_sec,
                end_time_sec=end_sec,
                boundary_type="scene_boundary",
                visual_complexity=await self._compute_visual_complexity(
                    content.raw_content, start_sec, end_sec
                ),
            ),
        )
        chunks.append(chunk)

    return ProcessingResult(
        success=len(chunks) > 0,
        modality=ModuleType.VIDEO,
        chunks=chunks,
        tokens_used=len(chunks) * 400,
    )
```

---

## 3.2 Text Extraction & Chunking from Transcriptions

### Challenge: Maintaining Boundaries Through Text Chunking

When audio/video is transcribed to text, need to preserve temporal references:

```python
@dataclass
class TimestampedText:
    """Text with word-level timestamps."""
    text: str
    words: List[Dict[str, Any]]  # [{"word": "hello", "start_sec": 0.5, "end_sec": 0.7}, ...]

async def transcribe_with_timestamps(audio_segment_bytes, start_offset_sec=0.0) -> TimestampedText:
    """
    Get transcription with word-level timestamps.
    Whisper and Azure Speech both support this.
    """
    # Azure Speech example
    response = await azure_speech_client.recognize_once_from_audio_data(
        audio_segment_bytes,
        result_only=False  # Get detailed result
    )

    # Extract word timings
    words = []
    if hasattr(response, 'json'):
        detailed = json.loads(response.json())
        for word_data in detailed.get('NBest', [{}])[0].get('Words', []):
            words.append({
                "word": word_data['Word'],
                "start_sec": start_offset_sec + word_data['Offset'] / 10_000_000,
                "end_sec": start_offset_sec + (word_data['Offset'] + word_data['Duration']) / 10_000_000,
            })

    return TimestampedText(
        text=" ".join([w["word"] for w in words]),
        words=words,
    )

def chunk_transcribed_text(timestamped_text, sentence_boundary=True, max_tokens=500):
    """
    Split transcribed text into semantic chunks while preserving timestamps.
    """
    chunks = []

    if sentence_boundary:
        # Split by sentence, track time bounds
        sentences = sent_tokenize(timestamped_text.text)
        current_chunk_text = []
        current_chunk_start = timestamped_text.words[0]["start_sec"] if timestamped_text.words else 0.0

        for sentence in sentences:
            current_chunk_text.append(sentence)
            chunk_length = sum(len(w.split()) for w in current_chunk_text)

            if chunk_length >= max_tokens or sentence == sentences[-1]:
                # Find end time for this chunk
                last_word_text = current_chunk_text[-1].split()[-1]
                end_time = find_word_end_time(last_word_text, timestamped_text.words)

                chunks.append({
                    "text": " ".join(current_chunk_text),
                    "start_sec": current_chunk_start,
                    "end_sec": end_time,
                })
                current_chunk_text = []
                current_chunk_start = end_time

    return chunks
```

---

# PART 4: COMPUTATIONAL COMPLEXITY & BENCHMARKS

## 4.1 Time Estimates (1 Hour of Content)

### Audio Pipeline

| Operation | Time | Notes |
|-----------|------|-------|
| Silence Detection (librosa) | 0.1s | Fast, single pass |
| Speaker Diarization (pyannote) | 30-120s | GPU: ~30s, CPU: ~120s |
| Speech-to-Text (Whisper) | 600s | Real-time or slower |
| **Total (hybrid)** | **630-720s** | ~10-12 min |

### Video Pipeline

| Operation | Video Spec | Time | Notes |
|-----------|---|---|---|
| Scene Detection | 1hr @ 30fps 720p | 180s | Adaptive detector |
| Keyframe Extraction (60 frames) | Same | 5s | Entropy-based |
| Frame Analysis (60 @ 400 tokens each) | Same | 120-600s | Depends on vision API |
| Audio Extraction & Transcription | Same | 600s | From video stream |
| **Total** | **1hr 720p 30fps** | **900-1400s** | **15-23 min** |

---

## 4.2 Memory Requirements

| Component | Memory | Notes |
|-----------|--------|-------|
| Audio (1 hour @ 16kHz 16-bit) | 115 MB | Uncompressed WAV |
| Librosa features | 10-50 MB | Spectrograms + embeddings |
| Pyannote model | 200-500 MB | CUDA-enabled GPU memory |
| Video (1 hour @ 720p H.264) | 300-1500 MB | Depends on codec & bitrate |
| OpenCV frames buffer | 50-200 MB | Typically ~5-10 frames at once |
| Scene detection (PySceneDetect) | 100-300 MB | Rolling window of frames |
| **Total with GPU** | **~2-3 GB** | Manageable for cloud VMs |

---

## 4.3 Async Processing Recommendations

**For Production**:
1. **Audio Processing**: Run diarization in background Celery task (30-120s)
2. **Video Processing**: Parallelize scene detection + keyframe extraction
3. **Transcription**: Stream to speech API (don't wait locally)
4. **Storage**: Write chunks to database as they complete

**Queue Strategy**:
```python
# Use Redis-backed Celery with priority queues
@app.task(bind=True, max_retries=3)
async def process_audio_async(session_id, content_id):
    """Long-running audio processing in background."""
    content = await load_content(content_id)
    chunks = await audio_processor.process(
        content,
        session_id,
        chunking_strategy=HybridAudioChunking(),
    )
    await save_chunks(session_id, chunks)
    return len(chunks)

# Frontend polls status
@app.get("/processing-status/{content_id}")
async def get_processing_status(content_id: UUID):
    """Check if processing complete."""
    task_id = redis_client.get(f"content:{content_id}:task_id")
    if task_id:
        task_result = AsyncResult(task_id)
        return {
            "status": task_result.state,  # PENDING, PROGRESS, SUCCESS, FAILURE
            "result": task_result.result if task_result.state == "SUCCESS" else None,
        }
```

---

# PART 5: LIBRARY RECOMMENDATIONS

## 5.1 Final Stack

### Audio Libraries
```toml
[audio-processing]
librosa = "0.10.0"              # Primary: silence detection, features
pyannote-audio = "3.0.1"        # Speaker diarization + voice activity
silero-vad = "5.0.0"            # Fallback: lightweight VAD
soundfile = "0.12.1"            # WAV/FLAC I/O
scipy = "1.10.0"                # Signal processing primitives
```

### Video Libraries
```toml
[video-processing]
opencv-python = "4.8.1.78"      # Core: frame extraction, analysis
scenedetect = "0.6.1"           # Scene detection (PySceneDetect)
ffmpeg-python = "0.2.1"         # FFmpeg wrapper
av = "11.0.0"                   # Advanced codec operations (optional)
numpy = "1.24.0"                # Numerical operations
```

### System Dependencies
```bash
# Ubuntu/Debian
apt-get install ffmpeg libsndfile1 libopencv-dev

# macOS
brew install ffmpeg libsndfile opencv

# Windows
# Download FFmpeg from https://ffmpeg.org/download.html
# Add to PATH
```

---

## 5.2 Async Architecture

### Recommended Setup

```python
# Background task processor
from celery import Celery
from kombu import Exchange, Queue

celery_app = Celery(
    'ragcore',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1',
)

# Priority queues
default_exchange = Exchange('ragcore', type='direct')
celery_app.conf.task_queues = (
    Queue('high_priority', exchange=default_exchange, routing_key='high'),
    Queue('audio_processing', exchange=default_exchange, routing_key='audio'),
    Queue('video_processing', exchange=default_exchange, routing_key='video'),
)

# Route tasks by priority
celery_app.conf.task_routes = {
    'ragcore.tasks.process_audio': {'queue': 'audio_processing'},
    'ragcore.tasks.process_video': {'queue': 'video_processing'},
}

@celery_app.task(queue='audio_processing', time_limit=3600)
async def process_audio_task(content_id: str, session_id: str):
    """Async audio processing (max 1 hour)."""
    # Runs in background, doesn't block HTTP request
    pass
```

---

# PART 6: INTEGRATION CHECKLIST

## Phase 0 Sprint: Smart Chunking Implementation

### Prerequisites
- [ ] Extend MultiModalChunk with TemporalMetadata
- [ ] Update database schema (add start_time, end_time columns)
- [ ] Create Alembic migration for temporal fields

### Audio Chunking
- [ ] Add librosa + pyannote.audio to requirements.txt
- [ ] Implement HybridAudioChunking class
- [ ] Integrate with AudioProcessor.process()
- [ ] Add word-level timestamp tracking from speech APIs
- [ ] Test with sample podcasts + meetings (multi-speaker)
- [ ] Benchmark: measure actual processing time on 1hr audio

### Video Chunking
- [ ] Add OpenCV + PySceneDetect to requirements.txt
- [ ] Implement AdaptiveVideoChunking class
- [ ] Integrate with VideoProcessor.process()
- [ ] Add frame timestamp tracking
- [ ] Test with diverse video types (cuts, fades, dissolves)
- [ ] Benchmark: measure actual processing time on 1hr video

### Integration & Testing
- [ ] Update router endpoints to accept chunking strategy params
- [ ] Add unit tests for chunking strategies (50+ test cases)
- [ ] Add integration tests with real audio/video samples
- [ ] Performance tests: CPU/memory profiling
- [ ] Setup async task processing with Celery + Redis
- [ ] Add monitoring/logging for chunking quality

### Documentation
- [ ] Update API docs with new temporal metadata fields
- [ ] Add usage examples for each chunking strategy
- [ ] Document performance tuning parameters
- [ ] Create migration guide for existing deployments

---

# REFERENCES

## Papers & Research

1. **Speaker Diarization**:
   - ECAPA-TDNN (Desplanques et al., 2021): https://arxiv.org/abs/2005.07143
   - pyannote.audio uses this + Hungarian algorithm for clustering

2. **Scene Detection**:
   - Towards Computational Scene Understanding (Palmer et al., 2015)
   - Adaptive cut detection: https://github.com/Breakthrough/PySceneDetect

3. **Silence Detection**:
   - Energy-based VAD (Sohn et al., 1999): Robust Speech Recognition via Large-Scale Weak Supervision
   - Modern approaches: WebRTC VAD, Silero VAD

4. **Video Analysis**:
   - Content-Based Video Indexing (Divakaran et al., 2005)
   - Optical flow: Farnebäck method (2003)

## Library Documentation

- **librosa**: https://librosa.org/ — Feature extraction guide
- **pyannote.audio**: https://github.com/pyannote/pyannote-audio — Diarization tutorial
- **PySceneDetect**: https://www.scenedetect.com/ — Configuration reference
- **OpenCV**: https://docs.opencv.org/ — Video I/O + optical flow

---

# APPENDIX: Quick Start Code

## Audio Chunking (5-minute setup)

```python
# 1. Install
pip install librosa pyannote.audio

# 2. Quick test
import librosa
y, sr = librosa.load('audio.mp3', sr=16000)

# Silence detection
S = librosa.feature.melspectrogram(y=y, sr=sr)
log_S = librosa.power_to_db(S)
energy = np.mean(log_S, axis=0)
threshold = np.mean(energy) - 2*np.std(energy)
silence_frames = np.where(energy < threshold)[0]
```

## Video Chunking (5-minute setup)

```python
# 1. Install
pip install opencv-python scenedetect

# 2. Quick test
from scenedetect import detect, AdaptiveDetector

scenes = detect("video.mp4", AdaptiveDetector(threshold=8.0))
for scene in scenes:
    print(f"Scene at {scene.get_seconds():.2f}s")
```

---

**Document Version**: 1.0
**Last Updated**: 2026-03-28
**Maintainer**: RAGCORE Development Team
**Status**: Ready for Implementation
