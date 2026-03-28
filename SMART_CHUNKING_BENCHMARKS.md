# Smart Chunking: Quick Reference & Benchmarks

---

## Quick Comparison Matrix

### Audio Chunking Techniques

| Technique | Accuracy | Speed | Library | Best For | False Positives |
|-----------|----------|-------|---------|----------|---|
| **Silence Detection** | 60-75% | ⚡ Fast (0.1s/hr) | librosa | Podcasts, interviews | Medium |
| **Speaker Diarization** | 85-95% | 🐢 Slow (30-120s/hr) | pyannote.audio | Multi-speaker meetings | Low |
| **Hybrid** (both) | 90-98% | 🟡 Medium (30-120s/hr) | Both libs | Production use | Lowest |
| **Fixed Interval** | N/A | ⚡ Instant | Native | Emergency fallback | N/A |

**Recommended**: Hybrid (silence + diarization) for production

---

### Video Chunking Techniques

| Technique | Accuracy | Speed | Library | Best For | False Positives |
|-----------|----------|-------|---------|----------|---|
| **Frame Diff** | 70% | ⚡ Fast (10ms/frame) | OpenCV | Rough cuts | High |
| **Optical Flow** | 85% | 🟡 Medium (80ms/frame) | OpenCV | Accurate detection | Low |
| **Scene Detection** | 85% | ⚡ Fast (5ms/frame) | PySceneDetect | Production use | Medium |
| **Adaptive Keyframes** | N/A | 🟡 Medium (1s/window) | OpenCV | Coverage | Low |
| **CNN-based** | 95% | 🐢 Slow (100ms/frame) | OpenCV+TF | High accuracy | Very Low |

**Recommended**: Scene detection + adaptive keyframes

---

## Performance Benchmarks

### 1 Hour of Content Processing Times

#### Audio (1 hour @ 16kHz)

```
┌──────────────────────────────────────┐
│  Silence Detection      ~0.5s         │ 0.0001x real-time
│  Speaker Diarization    ~60s (avg)    │ 0.017x real-time
│  Hybrid (combined)      ~60s          │ 0.017x real-time
│  Transcription (Whisper)~900-3600s    │ 0.25-1x real-time
│                                        │
│  TOTAL PIPELINE         ~960-3660s    │ 0.26-1x real-time
│  (~16-61 minutes)                     │
└──────────────────────────────────────┘
```

**Critical Path**: Speech-to-Text dominates (not chunking)

#### Video (1 hour @ 30fps 720p)

```
┌──────────────────────────────────────┐
│  Scene Detection        ~180s         │ 0.05x real-time
│  Keyframe Extraction    ~5s           │ 0.001x real-time
│  Frame Analysis (60x)   ~300-1500s    │ 0.08-0.4x real-time
│                                        │
│  TOTAL (w/o AI)         ~185-1685s    │ 0.05-0.47x real-time
│  TOTAL (w/ AI vision)   ~485-3185s    │ 0.13-0.88x real-time
│  (~8-53 minutes)                      │
└──────────────────────────────────────┘
```

**Bottleneck**: Frame analysis via Claude Vision API

---

## Memory Footprint

| Component | Size | Notes |
|-----------|------|-------|
| Audio (1hr @ 16kHz 16-bit) | 115 MB | Uncompressed in RAM |
| librosa Features | 20-50 MB | Spectrograms, MFCCs |
| pyannote Model | 200-500 MB | GPU memory required |
| **Audio Total** | **350-650 MB** | With diarization |
| --- | --- | --- |
| Video (1hr @ 720p H.264) | 300-1500 MB | Depends on bitrate |
| OpenCV Frame Buffer | 100-300 MB | ~5-10 frames at 1080p |
| **Video Total** | **400-1800 MB** | Without AI analysis |

**Recommendation**: Deploy on VM with 4-8GB RAM

---

## Async Processing Strategy

### Recommended Architecture

```python
# User uploads audio/video → immediately returns content_id
# Backend enqueues processing task

# High Priority (< 5 min): Image, small audio
# Normal Priority (5-30 min): Typical audio/video
# Low Priority: Large video, archive processing

from celery import Celery, states
from celery.result import AsyncResult

app = Celery('ragcore')

# Route long tasks to separate queue
@app.task(queue='long_processing', time_limit=7200)
async def process_long_audio(content_id):
    """1-hour audio: up to 2 hours timeout."""
    pass

# Short tasks run immediately
@app.task(queue='default', time_limit=600)
async def process_short_audio(content_id):
    """Podcasts < 15 min: up to 10 min timeout."""
    pass

# UI polls this endpoint
@app.route('/content/{content_id}/status')
def get_status(content_id):
    """Return processing status."""
    return {
        "status": "processing",  # pending, processing, success, failure
        "progress": 45,  # 0-100%
        "chunks_created": 12,
        "eta_seconds": 240,
    }
```

---

## Tuning Parameters

### Audio Chunking

```python
# Silence Detection
silence_threshold = 0.5      # 0.4-0.6: lower = more sensitive
min_silence_ms = 300         # 200-500: shorter = more chunks

# Speaker Diarization
num_speakers = None          # Auto-detect best (None = automatic)
min_speaker_duration_sec = 1 # Merge short speaker turns

# Hybrid
min_chunk_duration_sec = 15  # Don't create micro-chunks
max_chunk_duration_sec = 90  # Respect context window
```

### Video Chunking

```python
# Scene Detection
scene_threshold = 8.0        # 1-30: lower = more scenes
                             # 8.0 = balanced (recommended)

# Keyframe Extraction
window_duration_sec = 5      # Sample every 5 seconds
max_frames = 30              # Never exceed 30 frames

# Overall
max_video_chunks = 30        # Final limit on output
```

---

## Deployment Checklist

### Prerequisites
- [ ] Python 3.10+
- [ ] FFmpeg system binary installed
- [ ] 4-8 GB RAM available
- [ ] Redis for async task queue
- [ ] PostgreSQL with pgvector

### Installation
- [ ] `pip install -r requirements.txt` (with new libs)
- [ ] `alembic upgrade head` (apply migrations)
- [ ] Test imports: `python -c "import librosa; from pyannote.audio import Pipeline"`

### Configuration
- [ ] Set `HUGGING_FACE_TOKEN` env var (for pyannote)
- [ ] Configure Celery broker URL (Redis)
- [ ] Set task timeouts: audio=3600s, video=7200s
- [ ] Enable chunking in processor config

### Testing
- [ ] Unit tests for each strategy (100+ test cases)
- [ ] Integration test with sample audio/video
- [ ] Performance profiling: CPU/memory
- [ ] Latency benchmark: measure vs. target SLAs

### Monitoring
- [ ] Log chunk boundaries created
- [ ] Track processing time per content
- [ ] Monitor memory usage during diarization
- [ ] Alert on task timeout/failure

---

## Troubleshooting

### Issue: "No module named librosa"
```bash
pip install librosa==0.10.0
```

### Issue: pyannote.audio model download fails
```bash
# Need Hugging Face token
export HUGGING_FACE_TOKEN="hf_xxxxx"
python -c "from pyannote.audio import Pipeline; Pipeline.from_pretrained(...)"
```

### Issue: FFmpeg not found
```bash
# Linux
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download from ffmpeg.org, add to PATH
```

### Issue: Out of memory during diarization
```python
# Reduce model size or process in chunks
# Or run on GPU-enabled machine
```

### Issue: Scene detection too slow
```python
# Use lower quality for initial detection
# Then process at full quality only for detected boundaries
```

---

## Real-World Performance Examples

### Podcast (45 min, 1 speaker, MP3)
- **Silence Detection**: 0.2s
- **Speaker Diarization**: Skipped (mono speaker)
- **Transcription**: ~30-40s (Whisper)
- **Total**: 30-40s
- **Chunks**: ~6-8 (intervals ~6-7 min)
- **Boundary Types**: All "silence"

### Meeting Recording (90 min, 4 speakers, WAV)
- **Silence Detection**: 0.3s
- **Speaker Diarization**: ~90s (GPU: 30s)
- **Transcription**: ~60-90s (Whisper)
- **Total**: 90-150s
- **Chunks**: ~12-15 (speaker turns)
- **Boundary Types**: Mostly "speaker_change"

### Short Product Demo Video (10 min, 720p, MP4)
- **Scene Detection**: ~18s
- **Keyframe Extraction**: ~2s
- **Frame Analysis (30 frames)**: ~2-5min (Claude Vision)
- **Total**: 2-5 min
- **Chunks**: 8-12 (scene-aware)
- **Boundary Types**: "scene_cut", "keyframe"

---

## When to Use Which Strategy

| Use Case | Audio Strategy | Video Strategy | Notes |
|----------|---|---|---|
| **Podcast** | Silence only | Scene detection | Fast, simple |
| **Meeting** | Hybrid (diar+silence) | Scene + Keyframes | Accurate speaker attribution |
| **Lecture** | Silence | Scene detection | Long content, topic shifts |
| **Interview** | Diarization | Scene detection | Q&A format |
| **Unstructured** | Hybrid | Scene + Keyframes | Safe default |
| **Low Latency** | Fixed interval | Keyframes only | Sacrifice accuracy |

---

## Cost Optimization

### Cloud Processing Costs (Rough Estimates)

**Using AWS** (as reference):
- 1 hour audio processing: $0.50-1.00 (Lambda + Transcribe)
- 1 hour video processing: $2-5 (EC2 + Vision API calls)

**Using self-hosted** (this approach):
- 1 hour audio: $0.01-0.05 (compute only, no API calls)
- 1 hour video: $0.05-0.20 (compute only, cheaper if using local vision)

**Recommendation**:
- Use self-hosted diarization (free)
- Cache chunking results
- Re-use existing chunks for similar content
- Batch process low-priority items at night

---

## References & Further Reading

1. **Audio Processing**
   - pyannote.audio docs: https://github.com/pyannote/pyannote-audio
   - librosa guide: https://librosa.org/doc/latest/

2. **Video Processing**
   - PySceneDetect: https://www.scenedetect.com/
   - OpenCV: https://docs.opencv.org/

3. **Benchmarking**
   - Audio processing: O(n) for features, O(n²) worst case for clustering
   - Video processing: O(n) for frame analysis

4. **Production Deployment**
   - Async processing: Celery + Redis
   - Monitoring: Prometheus + Grafana
   - Storage: S3/Blob for raw media, PostgreSQL for metadata

---

**Document Version**: 1.0
**Last Updated**: 2026-03-28
**Ready for Implementation**: Yes
**Estimated Timeline**: 2-3 weeks (including testing & integration)
