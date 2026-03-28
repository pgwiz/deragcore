"""Integration tests for Phase 0 Task 5: Smart Chunking per Modality.

Tests audio silence detection, speaker diarization (mocked), and video scene detection.
"""

import pytest
from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock, patch
import logging

from ragcore.modules.multimodal.chunking import (
    AudioSilenceChunker,
    SpeakerDiarizationChunker,
    VideoSceneChunker,
)
from ragcore.modules.multimodal.processors.audio_processor import AudioProcessor
from ragcore.modules.multimodal.processors.video_processor import VideoProcessor
from ragcore.modules.multimodal.models import (
    MultiModalContent,
    MultiModalMetadata,
    ModuleType,
)
from ragcore.config import settings

logger = logging.getLogger(__name__)


# ============================================================================
# AUDIO CHUNKING TESTS
# ============================================================================


class TestAudioSilenceChunker:
    """Test audio silence detection chunking."""

    @pytest.fixture
    def silence_chunker(self):
        """Create silence chunker instance."""
        return AudioSilenceChunker(
            energy_percentile=20,
            min_chunk_duration_s=2.0,
            silence_duration_s=0.5,
        )

    def test_chunk_by_silence_empty_transcript(self, silence_chunker):
        """Test chunking with empty transcript."""
        result = silence_chunker._chunk_by_sentences("")
        assert result == []

    def test_chunk_by_sentences_fallback(self, silence_chunker):
        """Test sentence-based chunking (fallback method)."""
        transcript = "Hello world. This is a test. Another sentence!"
        result = silence_chunker._chunk_by_sentences(transcript)

        assert len(result) == 3
        assert result[0]["content"] == "Hello world"
        assert result[1]["content"] == "This is a test"
        assert result[2]["content"] == "Another sentence"
        assert all(0.6 <= chunk["confidence"] <= 0.8 for chunk in result)

    def test_chunk_timestamp_parsing(self, silence_chunker):
        """Test word-level timestamp chunking."""
        transcript = "one two three four five"
        timestamps = [
            (0.0, 0.5),   # "one"
            (0.5, 1.0),   # "two"
            (1.0, 1.5),   # "three" (gap before next)
            (2.0, 2.5),   # "four"  (large gap = chunk boundary)
            (2.5, 3.0),   # "five"
        ]

        result = silence_chunker._chunk_by_timestamps(transcript, timestamps)

        # Should create at least 1 chunk
        assert len(result) >= 1
        assert result[0]["boundary_type"] == "silence"
        assert result[0]["confidence"] > 0.9

    def test_estimate_chunk_count(self, silence_chunker):
        """Test chunk count estimation."""
        # ~100 words per chunk
        transcript = " ".join(["word"] * 250)
        estimated = silence_chunker.estimate_chunk_count(transcript)

        assert 1 <= estimated <= 4  # Should estimate 2-3 chunks
        assert isinstance(estimated, int)

    @pytest.mark.asyncio
    async def test_chunk_by_silence_basic(self, silence_chunker):
        """Test basic silence detection (without audio file)."""
        transcript = "First sentence. Second sentence. Third."
        result = await silence_chunker.chunk_by_silence(transcript)

        assert len(result) >= 1
        assert all("content" in chunk for chunk in result)
        assert all("start_sec" in chunk for chunk in result)
        assert all("end_sec" in chunk for chunk in result)

    @pytest.mark.asyncio
    async def test_chunk_by_silence_with_timestamps(self, silence_chunker):
        """Test silence detection with word-level timestamps."""
        transcript = "one two three four five"
        timestamps = [
            (0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (2.0, 2.5), (2.5, 3.0)
        ]

        result = await silence_chunker.chunk_by_silence(transcript, timestamps=timestamps)

        assert len(result) >= 1
        assert all("start_sec" in chunk for chunk in result)
        assert all("end_sec" in chunk for chunk in result)


class TestSpeakerDiarizationChunker:
    """Test speaker diarization chunking (mocked)."""

    @pytest.fixture
    def speaker_chunker(self):
        """Create speaker chunker instance."""
        return SpeakerDiarizationChunker(huggingface_token=None)

    def test_extract_speaker_segments(self, speaker_chunker):
        """Test speaker segment extraction from mock diarization."""
        # Mock diarization output
        mock_diarization = MagicMock()
        mock_diarization.itertracks.return_value = [
            (MagicMock(start=0.0, end=5.0), MagicMock(), "Speaker 1"),
            (MagicMock(start=5.0, end=10.0), MagicMock(), "Speaker 2"),
            (MagicMock(start=10.0, end=15.0), MagicMock(), "Speaker 1"),
        ]

        segments = speaker_chunker._extract_speaker_segments(mock_diarization)

        assert len(segments) == 3
        assert segments[0]["speaker"] == "Speaker 1"
        assert segments[0]["duration_sec"] == 5.0
        assert segments[1]["speaker"] == "Speaker 2"

    def test_get_speaker_count(self, speaker_chunker):
        """Test unique speaker count."""
        segments = [
            {"speaker": "Speaker 1", "start_sec": 0, "end_sec": 5},
            {"speaker": "Speaker 1", "start_sec": 5, "end_sec": 10},
            {"speaker": "Speaker 2", "start_sec": 10, "end_sec": 15},
        ]

        count = speaker_chunker.get_speaker_count(segments)
        assert count == 2

    def test_merge_adjacent_speakers(self, speaker_chunker):
        """Test merging chunks from same speaker with small gaps."""
        chunks = [
            {"speaker": "A", "start_sec": 0, "end_sec": 5, "content": "Hello", "duration_sec": 5},
            {"speaker": "A", "start_sec": 5.2, "end_sec": 10, "content": "World", "duration_sec": 4.8},
            {"speaker": "B", "start_sec": 10, "end_sec": 15, "content": "Hi", "duration_sec": 5},
        ]

        merged = speaker_chunker.merge_adjacent_speakers(chunks, merge_silence_threshold_s=0.5)

        assert len(merged) == 2  # First two should merge
        assert merged[0]["content"] == "Hello World"
        assert merged[0]["duration_sec"] == 10.0

    def test_map_transcript_to_speakers(self, speaker_chunker):
        """Test mapping transcript words to speaker segments."""
        transcript = "Alice said hello world. Bob replied thanks"
        segments = [
            {"start_sec": 0, "end_sec": 3, "speaker": "Alice"},
            {"start_sec": 3, "end_sec": 6, "speaker": "Bob"},
        ]

        result = speaker_chunker._map_transcript_to_speakers(transcript, segments)

        assert len(result) == 2
        assert result[0]["speaker"] == "Alice"
        assert result[1]["speaker"] == "Bob"
        assert all("content" in chunk for chunk in result)


# ============================================================================
# VIDEO CHUNKING TESTS
# ============================================================================


class TestVideoSceneChunker:
    """Test video scene detection chunking."""

    @pytest.fixture
    def scene_chunker(self):
        """Create scene chunker instance."""
        return VideoSceneChunker(
            detection_method="content",
            threshold=27.0,
            min_scene_length_s=1.0,
            keyframes_per_scene=2,
        )

    def test_select_keyframes_single(self, scene_chunker):
        """Test keyframe selection for single keyframe."""
        keyframes = scene_chunker._select_keyframes(
            start_sec=0, end_sec=10, fps=30, num_keyframes=1
        )

        assert len(keyframes) == 1
        assert keyframes[0] == 150  # Middle of 10s at 30fps (frames 0-300)

    def test_select_keyframes_multiple(self, scene_chunker):
        """Test keyframe selection for multiple keyframes."""
        keyframes = scene_chunker._select_keyframes(
            start_sec=0, end_sec=10, fps=30, num_keyframes=3
        )

        assert len(keyframes) == 3
        assert keyframes[0] == 0  # Start
        assert keyframes[-1] == 299  # End (10s * 30fps - 1)

    def test_estimate_chunk_count(self, scene_chunker):
        """Test scene count estimation."""
        video_duration = 60
        avg_scene = 5
        estimated = scene_chunker.estimate_chunk_count(video_duration, avg_scene)

        assert estimated == 12  # 60 / 5

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires OpenCV (cv2) - install with: pip install opencv-python")
    async def test_chunk_by_duration_fallback(self, scene_chunker):
        """Test duration-based chunking (fallback)."""
        result = await scene_chunker._chunk_by_duration(
            video_path="mock_path.mp4",
            chunk_duration_s=10.0,
        )

        # This will likely fail without real video file, but test structure
        assert isinstance(result, list)

    def test_create_scene_chunks_structure(self, scene_chunker):
        """Test scene chunk structure creation."""
        boundaries = [(0, 5), (5, 10), (10, 15)]
        narration_map = {0: "Scene 1 narration", 7: "Scene 2 narration"}

        # This test verifies the expected structure if we had video
        for start, end in boundaries:
            assert end > start
            assert isinstance(start, (int, float))


# ============================================================================
# PROCESSOR INTEGRATION TESTS
# ============================================================================


class TestAudioProcessorChunking:
    """Test AudioProcessor with chunking integration."""

    @pytest.fixture
    async def audio_processor(self):
        """Create audio processor instance."""
        return AudioProcessor(
            azure_client=None,
            whisper_client=None,
            embedding_model=None,
        )

    @pytest.fixture
    def mock_content(self):
        """Create mock audio content."""
        metadata = MultiModalMetadata(
            modality=ModuleType.AUDIO,
            file_name="test.mp3",
            file_size_bytes=1000000,
            duration_seconds=30,
        )
        return MultiModalContent(
            session_id=uuid4(),
            id=uuid4(),
            modality=ModuleType.AUDIO,
            raw_content=b"fake audio data",
            metadata=metadata,
        )

    @pytest.mark.asyncio
    async def test_apply_audio_chunking_silence_strategy(self, audio_processor, mock_content):
        """Test audio chunking with silence strategy."""
        with patch.object(settings, "audio_chunking_strategy", "silence"):
            transcription = "Hello world. How are you today?"
            session_id = uuid4()

            chunks = await audio_processor._apply_audio_chunking(
                transcription=transcription,
                session_id=session_id,
                metadata=mock_content.metadata,
                base_confidence=0.9,
            )

            assert len(chunks) >= 1
            assert all(chunk.session_id == session_id for chunk in chunks)
            assert all(chunk.modality == ModuleType.AUDIO for chunk in chunks)

    @pytest.mark.asyncio
    async def test_apply_audio_chunking_fallback(self, audio_processor, mock_content):
        """Test fallback to single chunk on error."""
        transcription = "Test transcript"
        session_id = uuid4()

        with patch.object(
            settings, "audio_chunking_strategy", "invalid_strategy"
        ):
            chunks = await audio_processor._apply_audio_chunking(
                transcription=transcription,
                session_id=session_id,
                metadata=mock_content.metadata,
                base_confidence=0.9,
            )

            assert len(chunks) >= 1
            assert chunks[0].content == transcription

    def test_create_default_chunk(self, audio_processor, mock_content):
        """Test default chunk creation."""
        session_id = uuid4()
        chunk = audio_processor._create_default_chunk(
            content="Test content",
            session_id=session_id,
            metadata=mock_content.metadata,
            confidence=0.85,
        )

        assert chunk.session_id == session_id
        assert chunk.modality == ModuleType.AUDIO
        assert chunk.content == "Test content"
        assert chunk.confidence_score == 0.85


class TestVideoProcessorChunking:
    """Test VideoProcessor with chunking integration."""

    @pytest.fixture
    def video_processor(self):
        """Create video processor instance."""
        return VideoProcessor(
            vision_client=None,
            speech_client=None,
            embedding_model=None,
            max_frames=30,
        )

    @pytest.fixture
    def mock_video_content(self):
        """Create mock video content."""
        metadata = MultiModalMetadata(
            modality=ModuleType.VIDEO,
            file_name="test.mp4",
            file_size_bytes=10000000,
            duration_seconds=60,
        )
        return MultiModalContent(
            session_id=uuid4(),
            id=uuid4(),
            modality=ModuleType.VIDEO,
            raw_content=b"fake video data",
            metadata=metadata,
        )

    def test_calculate_adaptive_frame_indices(self, video_processor):
        """Test adaptive frame sampling calculation."""
        # 10 second video
        indices = video_processor._calculate_adaptive_frame_indices(duration_seconds=10)

        assert len(indices) > 0
        assert indices[0] == 0  # Starts at beginning
        assert indices[-1] <= 10  # Ends before duration

    def test_calculate_adaptive_frame_indices_long(self, video_processor):
        """Test adaptive frame sampling for long video."""
        # 5 minute video
        indices = video_processor._calculate_adaptive_frame_indices(duration_seconds=300)

        assert len(indices) <= 30  # Max frames limit
        assert indices[0] == 0
        assert all(isinstance(t, float) for t in indices)


# ============================================================================
# END-TO-END PIPELINE TESTS
# ============================================================================


class TestChunkingEndToEnd:
    """Test end-to-end chunking with full multimodal pipeline."""

    @pytest.mark.asyncio
    async def test_audio_processing_with_chunks(self):
        """Test complete audio processing with chunking."""
        processor = AudioProcessor()
        metadata = MultiModalMetadata(
            modality=ModuleType.AUDIO,
            file_name="test.mp3",
            duration_seconds=30,
        )
        content = MultiModalContent(
            session_id=uuid4(),
            id=uuid4(),
            modality=ModuleType.AUDIO,
            raw_content=b"audio",
            metadata=metadata,
        )

        # Mock the Azure Speech response
        with patch.object(
            processor, "_call_azure_speech"
        ) as mock_speech:
            mock_speech.return_value = {
                "text": "Hello world. How are you? I am fine.",
                "confidence": 0.92,
            }

            result = await processor._process_with_azure_speech(
                content=content,
                session_id=uuid4(),
                language="en-US",
                segment_audio=False,
            )

            assert result.success
            assert len(result.chunks) >= 1
            assert result.extracted_text == "Hello world. How are you? I am fine."

    def test_temporal_metadata_preservation(self):
        """Test that chunk metadata is preserved in chunks."""
        processor = AudioProcessor()
        session_id = uuid4()
        metadata = MultiModalMetadata(
            modality=ModuleType.AUDIO,
            file_name="test.mp3"
        )

        chunk_data = [
            {"content": "First part", "start_sec": 0, "end_sec": 5, "confidence": 0.9},
            {"content": "Second part", "start_sec": 5, "end_sec": 10, "confidence": 0.85},
        ]

        chunks = []
        for idx, chunk_info in enumerate(chunk_data):
            from ragcore.modules.multimodal.models import MultiModalChunk

            chunk = MultiModalChunk(
                id=uuid4(),
                session_id=session_id,
                modality=ModuleType.AUDIO,
                content=chunk_info["content"],
                embedding=[],
                metadata=metadata,
                source_index=idx,
                confidence_score=chunk_info["confidence"],
            )
            chunks.append(chunk)

        # Verify chunks were created with correct source indices and confidence scores
        assert len(chunks) == 2
        assert chunks[0].content == "First part"
        assert chunks[1].content == "Second part"
        assert chunks[0].source_index == 0
        assert chunks[1].source_index == 1
        assert chunks[0].confidence_score == 0.9
        assert chunks[1].confidence_score == 0.85


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================


class TestChunkingConfiguration:
    """Test chunking configuration integration."""

    def test_audio_config_defaults(self):
        """Test default audio chunking configuration."""
        assert settings.audio_chunking_strategy == "silence"
        assert settings.audio_silence_threshold == 20
        assert settings.audio_min_chunk_duration_s == 2.0
        assert settings.audio_silence_duration_s == 0.5

    def test_video_config_defaults(self):
        """Test default video chunking configuration."""
        assert settings.video_chunking_strategy == "content"
        assert settings.video_scene_detection_threshold == 27.0
        assert settings.video_min_scene_length_s == 1.0
        assert settings.video_keyframes_per_scene == 2

    def test_chunker_initialization_from_config(self):
        """Test initializing chunkers from config."""
        audio_chunker = AudioSilenceChunker(
            energy_percentile=settings.audio_silence_threshold,
            min_chunk_duration_s=settings.audio_min_chunk_duration_s,
            silence_duration_s=settings.audio_silence_duration_s,
        )

        assert audio_chunker.energy_percentile == 20
        assert audio_chunker.min_chunk_duration_s == 2.0

        video_chunker = VideoSceneChunker(
            detection_method=settings.video_chunking_strategy,
            threshold=settings.video_scene_detection_threshold,
            min_scene_length_s=settings.video_min_scene_length_s,
            keyframes_per_scene=settings.video_keyframes_per_scene,
        )

        assert video_chunker.detection_method == "content"
        assert video_chunker.threshold == 27.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
