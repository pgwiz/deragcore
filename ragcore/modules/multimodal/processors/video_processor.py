"""Video processor - Adaptive frame sampling with narration extraction."""

import logging
from typing import Optional, List
from uuid import UUID, uuid4
import time

from ragcore.modules.multimodal.models import (
    MultiModalContent,
    MultiModalChunk,
    ModuleType,
    ProcessingResult,
    VideoFormat,
)
from ragcore.modules.multimodal.processors.base import BaseModalityProcessor
from ragcore.modules.multimodal.chunking import VideoSceneChunker
from ragcore.config import settings

logger = logging.getLogger(__name__)


class VideoProcessor(BaseModalityProcessor):
    """Process video with adaptive frame sampling and narration extraction."""

    def __init__(
        self,
        vision_client=None,
        speech_client=None,
        embedding_model=None,
        max_frames: int = 30,
    ):
        """Initialize video processor.

        Args:
            vision_client: Claude Vision or Azure for frame analysis
            speech_client: Speech-to-Text for audio narration
            embedding_model: Embedding model for generating vectors
            max_frames: Maximum frames to extract (adaptive sampling)
        """
        super().__init__(ModuleType.VIDEO)
        self.vision_client = vision_client
        self.speech_client = speech_client
        self.embedding_model = embedding_model
        self.max_frames = max_frames

    def get_supported_formats(self) -> list:
        """Get supported video formats."""
        return [fmt.value for fmt in VideoFormat]

    def validate_content(self, content: MultiModalContent) -> bool:
        """Validate video content.

        Args:
            content: Content to validate

        Returns:
            True if valid video content
        """
        if content.modality != ModuleType.VIDEO:
            return False

        # Check file format if provided
        if content.metadata.file_name:
            ext = content.metadata.file_name.lower().split(".")[-1]
            if ext not in [fmt.value for fmt in VideoFormat]:
                self.logger.warning(f"Unsupported video format: {ext}")
                return False

        # Check size (max 5GB for video)
        if content.get_size_mb() > 5000:
            self.logger.warning("Video file too large (>5GB)")
            return False

        # Check duration if available (max 2 hours)
        if content.metadata.duration_seconds and content.metadata.duration_seconds > 7200:
            self.logger.warning("Video duration too long (>2 hours)")
            return False

        return True

    def estimate_processing_time_ms(self, content: MultiModalContent) -> float:
        """Estimate video processing time.

        Video processing includes:
        - Frame extraction: ~1 frame/sec avg
        - Frame analysis: ~2-3 seconds per frame with Claude
        - Audio extraction: ~1x real-time with speech-to-text
        - Total: duration_seconds * (1-2 seconds per second of video)

        Args:
            content: Video content

        Returns:
            Estimated time in ms
        """
        if content.metadata.duration_seconds:
            # Rough estimate: 1.5x real-time for extraction + analysis
            return content.metadata.duration_seconds * 2000
        return 30000  # Default 30 second estimate

    def estimate_tokens_used(self, content: MultiModalContent) -> int:
        """Estimate tokens for video processing.

        Token usage from video:
        - ~30 frames extracted (adaptive)
        - ~300-600 tokens per frame analysis with vision
        - ~1 token per 4 words of speech transcription
        - Total: 30 frames * 450 tokens + duration * 94 words/min

        Args:
            content: Video content

        Returns:
            Estimated token count
        """
        frame_tokens = self.max_frames * 450  # Frames analysis
        speech_tokens = 0
        if content.metadata.duration_seconds:
            speech_tokens = int(content.metadata.duration_seconds * 150 / 60 / 4)
        return frame_tokens + speech_tokens

    async def process(
        self,
        content: MultiModalContent,
        session_id: UUID,
        extract_frames: bool = True,
        extract_narration: bool = True,
        **kwargs,
    ) -> ProcessingResult:
        """Process video with adaptive sampling.

        Args:
            content: Video content to process
            session_id: Session ID for context
            extract_frames: If True, extract and analyze keyframes
            extract_narration: If True, extract and transcribe audio
            **kwargs: Additional arguments

        Returns:
            ProcessingResult with frame chunks and narration
        """
        start_time = time.time()
        await self._log_processing_start(content)

        # Validate content
        if not self.validate_content(content):
            result = ProcessingResult(
                success=False,
                modality=ModuleType.VIDEO,
                error_message="Invalid video content",
                processing_time_ms=0,
            )
            await self._log_processing_error(Exception("Invalid content"))
            return result

        try:
            chunks: List[MultiModalChunk] = []
            full_text = ""
            total_tokens = 0
            confidence_scores = []

            # Extract frames
            if extract_frames:
                frame_result = await self._extract_and_analyze_frames(
                    content,
                    session_id,
                )
                if frame_result.success:
                    chunks.extend(frame_result.chunks)
                    full_text += frame_result.extracted_text or ""
                    total_tokens += frame_result.tokens_used
                    confidence_scores.extend(frame_result.confidence_scores)
                else:
                    self.logger.warning("Frame extraction failed")

            # Extract narration/audio
            if extract_narration:
                narration_result = await self._extract_narration(
                    content,
                    session_id,
                )
                if narration_result.success:
                    chunks.extend(narration_result.chunks)
                    if full_text:
                        full_text += "\n\n[Audio narration]:\n"
                    full_text += narration_result.extracted_text or ""
                    total_tokens += narration_result.tokens_used
                    confidence_scores.extend(narration_result.confidence_scores)
                else:
                    self.logger.warning("Audio extraction failed")

            if not chunks:
                return ProcessingResult(
                    success=False,
                    modality=ModuleType.VIDEO,
                    error_message="No content extracted from video",
                    processing_time_ms=(time.time() - start_time) * 1000,
                )

            return ProcessingResult(
                success=True,
                modality=ModuleType.VIDEO,
                chunks=chunks,
                extracted_text=full_text,
                processing_time_ms=(time.time() - start_time) * 1000,
                tokens_used=total_tokens,
                confidence_scores=confidence_scores or [0.85],
            )

        except Exception as e:
            await self._log_processing_error(e)
            return ProcessingResult(
                success=False,
                modality=ModuleType.VIDEO,
                error_message=str(e),
                processing_time_ms=(time.time() - start_time) * 1000,
            )

    async def _extract_and_analyze_frames(
        self,
        content: MultiModalContent,
        session_id: UUID,
    ) -> ProcessingResult:
        """Extract keyframes from video and analyze with vision.

        Adaptive sampling strategy:
        - First 5 seconds: 1 frame/sec (5 frames)
        - Next 55 seconds: 1 frame every 3 sec (~18 frames)
        - Remainder: 1 frame every 30 sec (~7 frames)
        - Max 30 frames total

        Args:
            content: Video content
            session_id: Session ID

        Returns:
            ProcessingResult with frame analysis chunks
        """
        try:
            # Placeholder: would extract frames from video using ffmpeg or similar
            frame_indices = self._calculate_adaptive_frame_indices(
                content.metadata.duration_seconds or 60
            )

            self.logger.info(
                f"Extracting {len(frame_indices)} frames from video "
                f"({content.metadata.duration_seconds}s duration)"
            )

            chunks: List[MultiModalChunk] = []
            confidence_scores = []
            full_text = ""

            for idx, frame_timestamp in enumerate(frame_indices):
                # Extract frame at timestamp
                frame_data = await self._extract_frame_at_timestamp(
                    content.raw_content,
                    frame_timestamp,
                )

                if not frame_data:
                    continue

                # Analyze frame with vision
                analysis = await self._analyze_frame(frame_data)

                if analysis and analysis.get("text"):
                    chunk = MultiModalChunk(
                        id=uuid4(),
                        session_id=session_id,
                        modality=ModuleType.VIDEO,
                        content=analysis.get("text", ""),
                        embedding=[],
                        metadata=content.metadata,
                        source_index=idx,  # Frame index
                        confidence_score=analysis.get("confidence", 0.85),
                        is_critical=idx == 0,  # First frame is critical
                    )
                    chunks.append(chunk)
                    full_text += f"\n[Frame {idx} @ {frame_timestamp:.1f}s]: {analysis.get('text', '')}"
                    confidence_scores.append(chunk.confidence_score)

            return ProcessingResult(
                success=len(chunks) > 0,
                modality=ModuleType.VIDEO,
                chunks=chunks,
                extracted_text=full_text,
                processing_time_ms=0,  # Set by caller
                tokens_used=len(chunks) * 400,  # Estimate 400 tokens per frame
                confidence_scores=confidence_scores or [0.85],
            )

        except Exception as e:
            self.logger.error(f"Frame extraction failed: {e}")
            return ProcessingResult(
                success=False,
                modality=ModuleType.VIDEO,
                error_message=f"Frame extraction error: {str(e)}",
            )

    async def _extract_narration(
        self,
        content: MultiModalContent,
        session_id: UUID,
    ) -> ProcessingResult:
        """Extract audio narration from video with smart chunking.

        Args:
            content: Video content
            session_id: Session ID

        Returns:
            ProcessingResult with audio transcription chunks
        """
        try:
            # Placeholder: would extract audio from video
            audio_data = await self._extract_audio_from_video(content.raw_content)

            if not audio_data:
                return ProcessingResult(
                    success=False,
                    modality=ModuleType.VIDEO,
                    error_message="No audio found in video",
                )

            # Transcribe audio
            transcription = await self._transcribe_audio(audio_data)

            if not transcription or not transcription.get("text"):
                return ProcessingResult(
                    success=False,
                    modality=ModuleType.VIDEO,
                    error_message="Audio transcription failed",
                )

            narration_text = transcription.get("text", "")
            base_confidence = transcription.get("confidence", 0.80)

            # Apply smart audio chunking to narration
            try:
                from ragcore.modules.multimodal.chunking import AudioSilenceChunker

                silence_chunker = AudioSilenceChunker(
                    energy_percentile=settings.audio_silence_threshold,
                    min_chunk_duration_s=settings.audio_min_chunk_duration_s,
                )
                chunk_data = await silence_chunker.chunk_by_silence(narration_text)

                chunks: List[MultiModalChunk] = []
                confidence_scores = []
                full_text = ""

                for idx, chunk_info in enumerate(chunk_data):
                    chunk = MultiModalChunk(
                        id=uuid4(),
                        session_id=session_id,
                        modality=ModuleType.AUDIO,
                        content=chunk_info.get("content", ""),
                        embedding=[],
                        metadata=content.metadata,
                        source_index=idx,
                        confidence_score=chunk_info.get("confidence", base_confidence),
                        is_critical=idx == 0,  # First narration chunk is critical
                    )
                    chunks.append(chunk)
                    confidence_scores.append(chunk.confidence_score)
                    full_text += f"\n[Narration {idx}]: {chunk_info.get('content', '')}"

                if not chunks:
                    # Fallback to single chunk
                    chunk = MultiModalChunk(
                        id=uuid4(),
                        session_id=session_id,
                        modality=ModuleType.AUDIO,
                        content=narration_text,
                        embedding=[],
                        metadata=content.metadata,
                        source_index=0,
                        confidence_score=base_confidence,
                        is_critical=True,
                    )
                    chunks = [chunk]
                    confidence_scores = [base_confidence]
                    full_text = narration_text

                return ProcessingResult(
                    success=True,
                    modality=ModuleType.VIDEO,
                    chunks=chunks,
                    extracted_text=full_text,
                    processing_time_ms=0,  # Set by caller
                    tokens_used=max(100, int((content.metadata.duration_seconds or 60) * 150 / 60 / 4)),
                    confidence_scores=confidence_scores,
                )

            except Exception as e:
                self.logger.warning(f"Audio chunking failed: {e}, using single narration chunk")
                # Fallback to single chunk
                chunk = MultiModalChunk(
                    id=uuid4(),
                    session_id=session_id,
                    modality=ModuleType.AUDIO,
                    content=narration_text,
                    embedding=[],
                    metadata=content.metadata,
                    source_index=0,
                    confidence_score=base_confidence,
                    is_critical=True,
                )

                return ProcessingResult(
                    success=True,
                    modality=ModuleType.VIDEO,
                    chunks=[chunk],
                    extracted_text=narration_text,
                    processing_time_ms=0,  # Set by caller
                    tokens_used=max(100, int((content.metadata.duration_seconds or 60) * 150 / 60 / 4)),
                    confidence_scores=[base_confidence],
                )

        except Exception as e:
            self.logger.error(f"Narration extraction failed: {e}")
            return ProcessingResult(
                success=False,
                modality=ModuleType.VIDEO,
                error_message=f"Narration extraction error: {str(e)}",
            )

    def _calculate_adaptive_frame_indices(self, duration_seconds: float) -> List[float]:
        """Calculate adaptive frame sampling timestamps.

        Strategy:
        - 0-5s: 1 frame/sec (5 frames)
        - 5-60s: 1 frame every 3 sec (18 frames)
        - 60+s: 1 frame every 30 sec (remaining)
        - Max 30 frames total

        Args:
            duration_seconds: Video duration

        Returns:
            List of timestamps (in seconds) to extract frames
        """
        frames = []

        # First 5 seconds: 1 frame/sec
        for sec in range(0, min(5, int(duration_seconds))):
            frames.append(float(sec))

        if duration_seconds > 5:
            # 5-60 seconds: 1 frame every 3 sec
            for sec in range(5, min(60, int(duration_seconds)), 3):
                frames.append(float(sec))

        if duration_seconds > 60:
            # 60+ seconds: 1 frame every 30 sec
            for sec in range(60, int(duration_seconds), 30):
                frames.append(float(sec))

        # Return up to max_frames
        return frames[: self.max_frames]

    async def _extract_frame_at_timestamp(
        self,
        video_bytes: bytes,
        timestamp_seconds: float,
    ) -> Optional[bytes]:
        """Extract frame from video at specific timestamp.

        Placeholder for actual frame extraction.

        Args:
            video_bytes: Raw video bytes
            timestamp_seconds: Timestamp in seconds

        Returns:
            Frame as image bytes or None
        """
        # Placeholder - would use ffmpeg
        return b"[frame data]"

    async def _analyze_frame(self, frame_data: bytes) -> Optional[dict]:
        """Analyze extracted frame with vision API.

        Placeholder for actual analysis.

        Args:
            frame_data: Frame image bytes

        Returns:
            Dict with text and confidence
        """
        return {
            "text": "[Frame analysis]",
            "confidence": 0.85,
        }

    async def _extract_audio_from_video(self, video_bytes: bytes) -> Optional[bytes]:
        """Extract audio track from video.

        Placeholder for actual extraction.

        Args:
            video_bytes: Raw video bytes

        Returns:
            Audio bytes or None
        """
        # Placeholder - would use ffmpeg
        return b"[audio data]"

    async def _transcribe_audio(self, audio_bytes: bytes) -> Optional[dict]:
        """Transcribe audio to text.

        Placeholder for actual transcription.

        Args:
            audio_bytes: Raw audio bytes

        Returns:
            Dict with text and confidence
        """
        return {
            "text": "[Video narration transcription]",
            "confidence": 0.82,
        }
