"""Integration tests for Sprint 5 Part 1: Multi-Modal Content Processing.

Tests cover:
- MultiModalContent and MultiModalChunk models
- BaseModalityProcessor abstract class
- ImageProcessor (Claude Vision + Azure fallback)
- AudioProcessor (Azure Speech-to-Text + Whisper fallback)
- VideoProcessor (adaptive frame sampling)
- ProcessingResult data class
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from uuid import uuid4

from ragcore.modules.multimodal.models import (
    ModuleType,
    MultiModalContent,
    MultiModalChunk,
    MultiModalMetadata,
    ProcessingResult,
    ImageFormat,
    AudioFormat,
    VideoFormat,
)
from ragcore.modules.multimodal.processors.base import BaseModalityProcessor
from ragcore.modules.multimodal.processors.image_processor import ImageProcessor
from ragcore.modules.multimodal.processors.audio_processor import AudioProcessor
from ragcore.modules.multimodal.processors.video_processor import VideoProcessor


# ============================================================================
# Tests for MultiModalMetadata
# ============================================================================


class TestMultiModalMetadata:
    """Test MultiModalMetadata dataclass."""

    def test_init_image_metadata(self):
        """Test metadata for image."""
        meta = MultiModalMetadata(
            modality=ModuleType.IMAGE,
            file_name="test.jpg",
            file_size_bytes=1024 * 100,  # 100KB
            width=1920,
            height=1080,
            extraction_method="claude_vision",
        )
        assert meta.modality == ModuleType.IMAGE
        assert meta.file_name == "test.jpg"
        assert meta.width == 1920
        assert meta.height == 1080

    def test_init_audio_metadata(self):
        """Test metadata for audio."""
        meta = MultiModalMetadata(
            modality=ModuleType.AUDIO,
            file_name="test.mp3",
            duration_seconds=120.5,
            sample_rate_hz=44100,
            channels=2,
            extraction_method="azure_speech",
        )
        assert meta.modality == ModuleType.AUDIO
        assert meta.duration_seconds == 120.5
        assert meta.sample_rate_hz == 44100

    def test_init_video_metadata(self):
        """Test metadata for video."""
        meta = MultiModalMetadata(
            modality=ModuleType.VIDEO,
            duration_seconds=300.0,
            width=1280,
            height=720,
            fps=30.0,
            extraction_method="adaptive_sampling",
        )
        assert meta.modality == ModuleType.VIDEO
        assert meta.fps == 30.0


# ============================================================================
# Tests for MultiModalChunk
# ============================================================================


class TestMultiModalChunk:
    """Test MultiModalChunk dataclass."""

    def test_init_basic(self):
        """Test basic chunk creation."""
        session_id = uuid4()
        chunk = MultiModalChunk(
            id=uuid4(),
            session_id=session_id,
            modality=ModuleType.IMAGE,
            content="[Image analysis text]",
            confidence_score=0.95,
        )
        assert chunk.modality == ModuleType.IMAGE
        assert chunk.confidence_score == 0.95
        assert chunk.is_critical is False

    def test_chunk_to_dict(self):
        """Test chunk serialization."""
        session_id = uuid4()
        chunk_id = uuid4()
        chunk = MultiModalChunk(
            id=chunk_id,
            session_id=session_id,
            modality=ModuleType.AUDIO,
            content="Transcribed text",
            confidence_score=0.88,
        )
        chunk_dict = chunk.to_dict()
        assert chunk_dict["id"] == str(chunk_id)
        assert chunk_dict["session_id"] == str(session_id)
        assert chunk_dict["modality"] == "audio"
        assert chunk_dict["confidence_score"] == 0.88

    def test_chunk_with_embedding(self):
        """Test chunk with embedding vector."""
        chunk = MultiModalChunk(
            id=uuid4(),
            session_id=uuid4(),
            modality=ModuleType.IMAGE,
            content="text",
            embedding=[0.1] * 1536,  # 1536-dim embedding
        )
        assert len(chunk.embedding) == 1536
        assert chunk.embedding[0] == 0.1


# ============================================================================
# Tests for MultiModalContent
# ============================================================================


class TestMultiModalContent:
    """Test MultiModalContent dataclass."""

    def test_init_image_content(self):
        """Test image content creation."""
        raw_bytes = b"fake_image_data"
        content = MultiModalContent(
            id=uuid4(),
            session_id=uuid4(),
            modality=ModuleType.IMAGE,
            raw_content=raw_bytes,
            metadata=MultiModalMetadata(
                modality=ModuleType.IMAGE,
                file_name="test.png",
            ),
        )
        assert content.modality == ModuleType.IMAGE
        assert content.is_processed is False
        assert len(content.processed_chunks) == 0

    def test_get_size_mb(self):
        """Test size calculation."""
        content = MultiModalContent(
            id=uuid4(),
            session_id=uuid4(),
            modality=ModuleType.AUDIO,
            raw_content=b"x" * (1024 * 1024 * 2),  # 2MB
            metadata=MultiModalMetadata(modality=ModuleType.AUDIO),
        )
        assert content.get_size_mb() == 2.0

    def test_should_inline_small_content(self):
        """Test inline storage for small content."""
        content = MultiModalContent(
            id=uuid4(),
            session_id=uuid4(),
            modality=ModuleType.IMAGE,
            raw_content=b"x" * (1024 * 50),  # 50KB
            metadata=MultiModalMetadata(modality=ModuleType.IMAGE),
        )
        assert content.should_inline() is True  # Default max 100KB

    def test_should_not_inline_large_content(self):
        """Test S3/Blob storage for large content."""
        content = MultiModalContent(
            id=uuid4(),
            session_id=uuid4(),
            modality=ModuleType.VIDEO,
            raw_content=b"x" * (1024 * 500),  # 500KB
            metadata=MultiModalMetadata(modality=ModuleType.VIDEO),
        )
        assert content.should_inline() is False


# ============================================================================
# Tests for ProcessingResult
# ============================================================================


class TestProcessingResult:
    """Test ProcessingResult dataclass."""

    def test_successful_result(self):
        """Test successful processing result."""
        chunks = [
            MultiModalChunk(
                id=uuid4(),
                session_id=uuid4(),
                modality=ModuleType.IMAGE,
                content="extracted",
            )
        ]
        result = ProcessingResult(
            success=True,
            modality=ModuleType.IMAGE,
            chunks=chunks,
            extracted_text="Full image analysis",
            processing_time_ms=2500.0,
            tokens_used=450,
            confidence_scores=[0.92],
        )
        assert result.success is True
        assert len(result.chunks) == 1
        assert result.tokens_used == 450

    def test_failure_result(self):
        """Test failed processing result."""
        result = ProcessingResult(
            success=False,
            modality=ModuleType.AUDIO,
            error_message="Audio file corrupted",
            processing_time_ms=500.0,
        )
        assert result.success is False
        assert result.error_message == "Audio file corrupted"
        assert len(result.chunks) == 0


# ============================================================================
# Tests for BaseModalityProcessor
# ============================================================================


class ConcreteModalityProcessor(BaseModalityProcessor):
    """Concrete implementation for testing."""

    async def process(self, content, session_id, **kwargs):
        """Placeholder process method."""
        return ProcessingResult(
            success=True,
            modality=self.modality,
            chunks=[],
        )

    def validate_content(self, content):
        """Placeholder validation."""
        return True


class TestBaseModalityProcessor:
    """Test base processor."""

    def test_init(self):
        """Test processor initialization."""
        processor = ConcreteModalityProcessor(ModuleType.IMAGE)
        assert processor.modality == ModuleType.IMAGE
        assert processor.logger is not None

    def test_get_supported_formats(self):
        """Test supported formats method."""
        processor = ConcreteModalityProcessor(ModuleType.AUDIO)
        formats = processor.get_supported_formats()
        assert isinstance(formats, list)

    def test_estimate_processing_time(self):
        """Test processing time estimation."""
        processor = ConcreteModalityProcessor(ModuleType.VIDEO)
        content = MultiModalContent(
            id=uuid4(),
            session_id=uuid4(),
            modality=ModuleType.VIDEO,
            raw_content=b"video",
            metadata=MultiModalMetadata(
                modality=ModuleType.VIDEO,
                duration_seconds=60.0,
            ),
        )
        time_ms = processor.estimate_processing_time_ms(content)
        assert time_ms >= 0

    def test_estimate_tokens_used(self):
        """Test token estimation."""
        processor = ConcreteModalityProcessor(ModuleType.IMAGE)
        content = MultiModalContent(
            id=uuid4(),
            session_id=uuid4(),
            modality=ModuleType.IMAGE,
            raw_content=b"x" * (1024 * 100),  # 100KB
            metadata=MultiModalMetadata(modality=ModuleType.IMAGE),
        )
        tokens = processor.estimate_tokens_used(content)
        assert tokens >= 0


# ============================================================================
# Tests for ImageProcessor
# ============================================================================


class TestImageProcessor:
    """Test image processing."""

    def test_init(self):
        """Test image processor initialization."""
        processor = ImageProcessor()
        assert processor.modality == ModuleType.IMAGE

    def test_get_supported_formats(self):
        """Test supported formats."""
        processor = ImageProcessor()
        formats = processor.get_supported_formats()
        assert "jpeg" in formats
        assert "png" in formats
        assert "webp" in formats

    def test_validate_content_invalid_modality(self):
        """Test validation fails for wrong modality."""
        processor = ImageProcessor()
        content = MultiModalContent(
            id=uuid4(),
            session_id=uuid4(),
            modality=ModuleType.AUDIO,  # Wrong!
            raw_content=b"data",
            metadata=MultiModalMetadata(modality=ModuleType.AUDIO),
        )
        assert processor.validate_content(content) is False

    def test_validate_content_invalid_format(self):
        """Test validation fails for unsupported format."""
        processor = ImageProcessor()
        content = MultiModalContent(
            id=uuid4(),
            session_id=uuid4(),
            modality=ModuleType.IMAGE,
            raw_content=b"data",
            metadata=MultiModalMetadata(
                modality=ModuleType.IMAGE,
                file_name="test.exe",  # Invalid!
            ),
        )
        assert processor.validate_content(content) is False

    def test_validate_content_too_large(self):
        """Test validation fails for oversized image."""
        processor = ImageProcessor()
        content = MultiModalContent(
            id=uuid4(),
            session_id=uuid4(),
            modality=ModuleType.IMAGE,
            raw_content=b"x" * (1024 * 1024 * 25),  # 25MB (max 20MB)
            metadata=MultiModalMetadata(modality=ModuleType.IMAGE),
        )
        assert processor.validate_content(content) is False

    def test_validate_content_valid(self):
        """Test validation passes for valid image."""
        processor = ImageProcessor()
        content = MultiModalContent(
            id=uuid4(),
            session_id=uuid4(),
            modality=ModuleType.IMAGE,
            raw_content=b"x" * (1024 * 100),  # 100KB
            metadata=MultiModalMetadata(
                modality=ModuleType.IMAGE,
                file_name="test.jpg",
            ),
        )
        assert processor.validate_content(content) is True

    def test_estimate_processing_time(self):
        """Test image processing time estimate."""
        processor = ImageProcessor()
        content = MultiModalContent(
            id=uuid4(),
            session_id=uuid4(),
            modality=ModuleType.IMAGE,
            raw_content=b"data",
            metadata=MultiModalMetadata(modality=ModuleType.IMAGE),
        )
        time_ms = processor.estimate_processing_time_ms(content)
        assert 2500 <= time_ms <= 3500  # Should be around 3 seconds

    def test_estimate_tokens_small_image(self):
        """Test token estimate for small image."""
        processor = ImageProcessor()
        content = MultiModalContent(
            id=uuid4(),
            session_id=uuid4(),
            modality=ModuleType.IMAGE,
            raw_content=b"x" * (1024 * 500),  # 500KB
            metadata=MultiModalMetadata(
                modality=ModuleType.IMAGE,
                file_size_bytes=1024 * 500,
            ),
        )
        tokens = processor.estimate_tokens_used(content)
        assert tokens == 300  # Small image

    def test_estimate_tokens_large_image(self):
        """Test token estimate for large image."""
        processor = ImageProcessor()
        content = MultiModalContent(
            id=uuid4(),
            session_id=uuid4(),
            modality=ModuleType.IMAGE,
            raw_content=b"x" * (1024 * 1024 * 10),  # 10MB
            metadata=MultiModalMetadata(
                modality=ModuleType.IMAGE,
                file_size_bytes=1024 * 1024 * 10,
            ),
        )
        tokens = processor.estimate_tokens_used(content)
        assert tokens == 1200  # Large image

    @pytest.mark.asyncio
    async def test_process_invalid_content(self):
        """Test processing invalid content returns failure."""
        processor = ImageProcessor()
        content = MultiModalContent(
            id=uuid4(),
            session_id=uuid4(),
            modality=ModuleType.AUDIO,  # Wrong modality
            raw_content=b"data",
            metadata=MultiModalMetadata(modality=ModuleType.AUDIO),
        )
        result = await processor.process(content, uuid4())
        assert result.success is False


# ============================================================================
# Tests for AudioProcessor
# ============================================================================


class TestAudioProcessor:
    """Test audio processing."""

    def test_init(self):
        """Test audio processor initialization."""
        processor = AudioProcessor()
        assert processor.modality == ModuleType.AUDIO

    def test_get_supported_formats(self):
        """Test supported formats."""
        processor = AudioProcessor()
        formats = processor.get_supported_formats()
        assert "mp3" in formats
        assert "wav" in formats
        assert "m4a" in formats

    def test_validate_content_invalid_modality(self):
        """Test validation fails for wrong modality."""
        processor = AudioProcessor()
        content = MultiModalContent(
            id=uuid4(),
            session_id=uuid4(),
            modality=ModuleType.VIDEO,
            raw_content=b"data",
            metadata=MultiModalMetadata(modality=ModuleType.VIDEO),
        )
        assert processor.validate_content(content) is False

    def test_validate_content_too_long(self):
        """Test validation fails for excessively long audio."""
        processor = AudioProcessor()
        content = MultiModalContent(
            id=uuid4(),
            session_id=uuid4(),
            modality=ModuleType.AUDIO,
            raw_content=b"data",
            metadata=MultiModalMetadata(
                modality=ModuleType.AUDIO,
                duration_seconds=3600 * 6,  # 6 hours (max 5)
            ),
        )
        assert processor.validate_content(content) is False

    def test_validate_content_valid(self):
        """Test validation passes for valid audio."""
        processor = AudioProcessor()
        content = MultiModalContent(
            id=uuid4(),
            session_id=uuid4(),
            modality=ModuleType.AUDIO,
            raw_content=b"x" * (1024 * 500),  # 500KB
            metadata=MultiModalMetadata(
                modality=ModuleType.AUDIO,
                file_name="test.mp3",
                duration_seconds=120.0,
            ),
        )
        assert processor.validate_content(content) is True

    def test_estimate_processing_time(self):
        """Test audio processing time estimate."""
        processor = AudioProcessor()
        content = MultiModalContent(
            id=uuid4(),
            session_id=uuid4(),
            modality=ModuleType.AUDIO,
            raw_content=b"data",
            metadata=MultiModalMetadata(
                modality=ModuleType.AUDIO,
                duration_seconds=60.0,  # 1 minute
            ),
        )
        time_ms = processor.estimate_processing_time_ms(content)
        assert time_ms == 60000 * 1.5  # 1.5x real-time

    def test_estimate_tokens(self):
        """Test token estimate for audio."""
        processor = AudioProcessor()
        content = MultiModalContent(
            id=uuid4(),
            session_id=uuid4(),
            modality=ModuleType.AUDIO,
            raw_content=b"data",
            metadata=MultiModalMetadata(
                modality=ModuleType.AUDIO,
                duration_seconds=120.0,  # 2 minutes
            ),
        )
        tokens = processor.estimate_tokens_used(content)
        assert tokens > 0  # Should estimate based on duration


# ============================================================================
# Tests for VideoProcessor
# ============================================================================


class TestVideoProcessor:
    """Test video processing."""

    def test_init(self):
        """Test video processor initialization."""
        processor = VideoProcessor(max_frames=30)
        assert processor.modality == ModuleType.VIDEO
        assert processor.max_frames == 30

    def test_get_supported_formats(self):
        """Test supported formats."""
        processor = VideoProcessor()
        formats = processor.get_supported_formats()
        assert "mp4" in formats
        assert "webm" in formats
        assert "mov" in formats

    def test_validate_content_invalid_format(self):
        """Test validation fails for unsupported format."""
        processor = VideoProcessor()
        content = MultiModalContent(
            id=uuid4(),
            session_id=uuid4(),
            modality=ModuleType.VIDEO,
            raw_content=b"data",
            metadata=MultiModalMetadata(
                modality=ModuleType.VIDEO,
                file_name="test.txt",  # Invalid!
            ),
        )
        assert processor.validate_content(content) is False

    def test_validate_content_too_long(self):
        """Test validation fails for excessively long video."""
        processor = VideoProcessor()
        content = MultiModalContent(
            id=uuid4(),
            session_id=uuid4(),
            modality=ModuleType.VIDEO,
            raw_content=b"data",
            metadata=MultiModalMetadata(
                modality=ModuleType.VIDEO,
                duration_seconds=3600 * 3,  # 3 hours (max 2)
            ),
        )
        assert processor.validate_content(content) is False

    def test_validate_content_valid(self):
        """Test validation passes for valid video."""
        processor = VideoProcessor()
        content = MultiModalContent(
            id=uuid4(),
            session_id=uuid4(),
            modality=ModuleType.VIDEO,
            raw_content=b"x" * (1024 * 1024 * 100),  # 100MB
            metadata=MultiModalMetadata(
                modality=ModuleType.VIDEO,
                file_name="test.mp4",
                duration_seconds=300.0,
            ),
        )
        assert processor.validate_content(content) is True

    def test_calculate_adaptive_frame_indices_short(self):
        """Test frame extraction for short video."""
        processor = VideoProcessor()
        indices = processor._calculate_adaptive_frame_indices(30.0)
        # Should get: 0-4 (5 frames per sec) + 5,8,11,14,17,20,23,26,29 (1 frame every 3 sec)
        assert len(indices) <= processor.max_frames
        assert 0.0 in indices  # First frame always included

    def test_calculate_adaptive_frame_indices_long(self):
        """Test frame extraction for long video."""
        processor = VideoProcessor(max_frames=30)
        indices = processor._calculate_adaptive_frame_indices(600.0)  # 10 minutes
        # Should get: frames from all three tiers, up to max_frames
        assert len(indices) <= 30
        assert 0.0 in indices

    def test_estimate_processing_time(self):
        """Test video processing time estimate."""
        processor = VideoProcessor()
        content = MultiModalContent(
            id=uuid4(),
            session_id=uuid4(),
            modality=ModuleType.VIDEO,
            raw_content=b"data",
            metadata=MultiModalMetadata(
                modality=ModuleType.VIDEO,
                duration_seconds=60.0,
            ),
        )
        time_ms = processor.estimate_processing_time_ms(content)
        assert time_ms == 60000 * 2  # 2x real-time


# ============================================================================
# Integration Tests
# ============================================================================


class TestMultiModalIntegration:
    """Integration tests for multi-modal processing."""

    @pytest.mark.asyncio
    async def test_image_processor_flow(self):
        """Test complete image processing flow."""
        processor = ImageProcessor()
        session_id = uuid4()

        content = MultiModalContent(
            id=uuid4(),
            session_id=session_id,
            modality=ModuleType.IMAGE,
            raw_content=b"fake_image_data",
            metadata=MultiModalMetadata(
                modality=ModuleType.IMAGE,
                file_name="test.jpg",
                file_size_bytes=1024 * 100,
            ),
        )

        # Validate first
        assert processor.validate_content(content) is True

        # Estimate before processing
        estimated_time = processor.estimate_processing_time_ms(content)
        assert estimated_time > 0

    @pytest.mark.asyncio
    async def test_audio_processor_flow(self):
        """Test complete audio processing flow."""
        processor = AudioProcessor()
        session_id = uuid4()

        content = MultiModalContent(
            id=uuid4(),
            session_id=session_id,
            modality=ModuleType.AUDIO,
            raw_content=b"fake_audio_data",
            metadata=MultiModalMetadata(
                modality=ModuleType.AUDIO,
                file_name="test.mp3",
                duration_seconds=120.0,
            ),
        )

        # Validate
        assert processor.validate_content(content) is True

        # Estimate tokens
        tokens = processor.estimate_tokens_used(content)
        assert tokens > 0

    @pytest.mark.asyncio
    async def test_video_processor_flow(self):
        """Test complete video processing flow."""
        processor = VideoProcessor(max_frames=20)
        session_id = uuid4()

        content = MultiModalContent(
            id=uuid4(),
            session_id=session_id,
            modality=ModuleType.VIDEO,
            raw_content=b"fake_video_data",
            metadata=MultiModalMetadata(
                modality=ModuleType.VIDEO,
                file_name="test.mp4",
                duration_seconds=300.0,
                width=1920,
                height=1080,
                fps=30.0,
            ),
        )

        # Validate
        assert processor.validate_content(content) is True

        # Calculate frames
        frames = processor._calculate_adaptive_frame_indices(300.0)
        assert len(frames) <= 20  # Max frames constraint
