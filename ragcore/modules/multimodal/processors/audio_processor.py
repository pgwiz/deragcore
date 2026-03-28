"""Audio processor - Azure Speech to Text with Whisper fallback."""

import logging
from typing import Optional
from uuid import UUID, uuid4
import time

from ragcore.modules.multimodal.models import (
    MultiModalContent,
    MultiModalChunk,
    ModuleType,
    ProcessingResult,
    AudioFormat,
)
from ragcore.modules.multimodal.processors.base import BaseModalityProcessor

logger = logging.getLogger(__name__)


class AudioProcessor(BaseModalityProcessor):
    """Process audio using Azure Speech-to-Text with Whisper fallback."""

    def __init__(
        self,
        azure_client=None,
        whisper_client=None,
        embedding_model=None,
    ):
        """Initialize audio processor.

        Args:
            azure_client: Azure Speech API client
            whisper_client: OpenAI Whisper API client for fallback
            embedding_model: Embedding model for generating vectors
        """
        super().__init__(ModuleType.AUDIO)
        self.azure_client = azure_client
        self.whisper_client = whisper_client
        self.embedding_model = embedding_model

    def get_supported_formats(self) -> list:
        """Get supported audio formats."""
        return [fmt.value for fmt in AudioFormat]

    def validate_content(self, content: MultiModalContent) -> bool:
        """Validate audio content.

        Args:
            content: Content to validate

        Returns:
            True if valid audio content
        """
        if content.modality != ModuleType.AUDIO:
            return False

        # Check file format if provided
        if content.metadata.file_name:
            ext = content.metadata.file_name.lower().split(".")[-1]
            if ext not in [fmt.value for fmt in AudioFormat]:
                self.logger.warning(f"Unsupported audio format: {ext}")
                return False

        # Check size (max 1GB for audio, but typically much smaller)
        if content.get_size_mb() > 1000:
            self.logger.warning("Audio file too large (>1GB)")
            return False

        # Check duration if available
        if content.metadata.duration_seconds and content.metadata.duration_seconds > 3600 * 5:
            self.logger.warning("Audio duration too long (>5 hours)")
            return False

        return True

    def estimate_processing_time_ms(self, content: MultiModalContent) -> float:
        """Estimate audio processing time.

        Speech-to-Text typically processes in real-time:
        - 60 second audio: ~60-120ms (real-time or faster)

        Args:
            content: Audio content

        Returns:
            Estimated time in ms
        """
        if content.metadata.duration_seconds:
            # Typically 1-2x real-time for speech recognition
            return content.metadata.duration_seconds * 1500  # 1.5x real-time estimate
        return 5000  # Default 5 second estimate

    def estimate_tokens_used(self, content: MultiModalContent) -> int:
        """Estimate tokens for audio processing.

        Whisper tokenization: ~1 token per 4 words in transcription
        Typical speech rate: 130-150 words per minute
        Estimate: duration_seconds * (150/60) * (1/4) = duration_seconds / 1.6

        Args:
            content: Audio content

        Returns:
            Estimated token count
        """
        if content.metadata.duration_seconds:
            # Rough estimate: speech generates ~94 words/min, 1 token per ~4 words
            return max(100, int(content.metadata.duration_seconds * 150 / 60 / 4))
        return 100

    async def process(
        self,
        content: MultiModalContent,
        session_id: UUID,
        language: str = "en-US",
        segment_audio: bool = True,
        **kwargs,
    ) -> ProcessingResult:
        """Process audio using Azure Speech-to-Text with fallback.

        Args:
            content: Audio content to process
            session_id: Session ID for context
            language: Language code (e.g., "en-US", "es-ES")
            segment_audio: If True, segment long audio into chunks
            **kwargs: Additional arguments (use_whisper_only=False)

        Returns:
            ProcessingResult with transcription chunks
        """
        start_time = time.time()
        await self._log_processing_start(content)

        # Validate content
        if not self.validate_content(content):
            result = ProcessingResult(
                success=False,
                modality=ModuleType.AUDIO,
                error_message="Invalid audio content",
                processing_time_ms=0,
            )
            await self._log_processing_error(Exception("Invalid content"))
            return result

        try:
            # Try Azure Speech-to-Text first (unless Whisper-only requested)
            use_whisper_only = kwargs.get("use_whisper_only", False)
            if not use_whisper_only and self.azure_client:
                result = await self._process_with_azure_speech(
                    content,
                    session_id,
                    language,
                    segment_audio,
                )
                if result.success:
                    result.processing_time_ms = (time.time() - start_time) * 1000
                    await self._log_processing_end(result)
                    return result
                self.logger.warning("Azure Speech failed, falling back to Whisper")

            # Fallback to Whisper
            if self.whisper_client:
                result = await self._process_with_whisper(
                    content,
                    session_id,
                    language,
                )
                result.processing_time_ms = (time.time() - start_time) * 1000
                await self._log_processing_end(result)
                return result

            # All backends failed
            return ProcessingResult(
                success=False,
                modality=ModuleType.AUDIO,
                error_message="No speech-to-text backend available (Azure and Whisper both failed)",
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            await self._log_processing_error(e)
            return ProcessingResult(
                success=False,
                modality=ModuleType.AUDIO,
                error_message=str(e),
                processing_time_ms=(time.time() - start_time) * 1000,
            )

    async def _process_with_azure_speech(
        self,
        content: MultiModalContent,
        session_id: UUID,
        language: str,
        segment_audio: bool,
    ) -> ProcessingResult:
        """Process audio with Azure Speech-to-Text.

        Args:
            content: Audio content
            session_id: Session ID
            language: Language code
            segment_audio: Whether to segment long audio

        Returns:
            ProcessingResult from Azure
        """
        try:
            # For long audio (>60 seconds), might want to segment
            if segment_audio and content.metadata.duration_seconds and content.metadata.duration_seconds > 60:
                return await self._process_with_azure_batch(
                    content,
                    session_id,
                    language,
                )

            # Call Azure Speech API
            response = await self._call_azure_speech(
                content.raw_content,
                language,
            )

            if not response or not response.get("text"):
                return ProcessingResult(
                    success=False,
                    modality=ModuleType.AUDIO,
                    error_message="Azure Speech returned empty transcription",
                )

            transcription = response.get("text", "")
            confidence = response.get("confidence", 0.90)

            # Create chunk for transcription
            chunk = MultiModalChunk(
                id=uuid4(),
                session_id=session_id,
                modality=ModuleType.AUDIO,
                content=transcription,
                embedding=[],
                metadata=content.metadata,
                source_index=0,
                confidence_score=confidence,
                is_critical=False,
            )

            return ProcessingResult(
                success=True,
                modality=ModuleType.AUDIO,
                chunks=[chunk],
                extracted_text=transcription,
                processing_time_ms=0,  # Set by caller
                tokens_used=self.estimate_tokens_used(content),
                confidence_scores=[confidence],
            )

        except Exception as e:
            self.logger.error(f"Azure Speech processing failed: {e}")
            return ProcessingResult(
                success=False,
                modality=ModuleType.AUDIO,
                error_message=f"Azure Speech error: {str(e)}",
            )

    async def _process_with_azure_batch(
        self,
        content: MultiModalContent,
        session_id: UUID,
        language: str,
    ) -> ProcessingResult:
        """Process long audio with Azure Batch transcription.

        Args:
            content: Audio content
            session_id: Session ID
            language: Language code

        Returns:
            ProcessingResult with segmented chunks
        """
        try:
            # Placeholder for batch processing
            response = await self._call_azure_batch_transcription(
                content.raw_content,
                language,
            )

            if not response or not response.get("segments"):
                return ProcessingResult(
                    success=False,
                    modality=ModuleType.AUDIO,
                    error_message="Azure Batch transcription failed",
                )

            # Create chunks for each segment
            chunks = []
            for i, segment in enumerate(response.get("segments", [])):
                chunk = MultiModalChunk(
                    id=uuid4(),
                    session_id=session_id,
                    modality=ModuleType.AUDIO,
                    content=segment.get("text", ""),
                    embedding=[],
                    metadata=content.metadata,
                    source_index=i,  # Segment index
                    confidence_score=segment.get("confidence", 0.90),
                    is_critical=False,
                )
                chunks.append(chunk)

            full_text = " ".join([seg.get("text", "") for seg in response.get("segments", [])])

            return ProcessingResult(
                success=True,
                modality=ModuleType.AUDIO,
                chunks=chunks,
                extracted_text=full_text,
                processing_time_ms=0,  # Set by caller
                tokens_used=self.estimate_tokens_used(content),
                confidence_scores=[ch.confidence_score for ch in chunks],
            )

        except Exception as e:
            self.logger.error(f"Azure Batch transcription failed: {e}")
            return ProcessingResult(
                success=False,
                modality=ModuleType.AUDIO,
                error_message=f"Azure Batch error: {str(e)}",
            )

    async def _process_with_whisper(
        self,
        content: MultiModalContent,
        session_id: UUID,
        language: str,
    ) -> ProcessingResult:
        """Process audio with Whisper API.

        Args:
            content: Audio content
            session_id: Session ID
            language: Language code

        Returns:
            ProcessingResult from Whisper
        """
        try:
            # Call Whisper API
            response = await self._call_whisper(
                content.raw_content,
                language,
            )

            if not response or not response.get("text"):
                return ProcessingResult(
                    success=False,
                    modality=ModuleType.AUDIO,
                    error_message="Whisper returned empty transcription",
                )

            transcription = response.get("text", "")

            # Create chunk for transcription
            chunk = MultiModalChunk(
                id=uuid4(),
                session_id=session_id,
                modality=ModuleType.AUDIO,
                content=transcription,
                embedding=[],
                metadata=content.metadata,
                source_index=0,
                confidence_score=0.85,  # Whisper doesn't provide confidence
                is_critical=False,
            )

            return ProcessingResult(
                success=True,
                modality=ModuleType.AUDIO,
                chunks=[chunk],
                extracted_text=transcription,
                processing_time_ms=0,  # Set by caller
                tokens_used=self.estimate_tokens_used(content),
                confidence_scores=[0.85],
            )

        except Exception as e:
            self.logger.error(f"Whisper processing failed: {e}")
            return ProcessingResult(
                success=False,
                modality=ModuleType.AUDIO,
                error_message=f"Whisper error: {str(e)}",
            )

    async def _call_azure_speech(
        self,
        audio_bytes: bytes,
        language: str,
    ) -> Optional[dict]:
        """Call Azure Speech-to-Text API.

        Placeholder for actual API call.

        Args:
            audio_bytes: Raw audio bytes
            language: Language code

        Returns:
            Dict with text and confidence
        """
        # Placeholder
        return {
            "text": "[Azure Speech transcription]",
            "confidence": 0.92,
        }

    async def _call_azure_batch_transcription(
        self,
        audio_bytes: bytes,
        language: str,
    ) -> Optional[dict]:
        """Call Azure Batch Transcription API.

        Placeholder for actual API call.

        Args:
            audio_bytes: Raw audio bytes
            language: Language code

        Returns:
            Dict with segments
        """
        # Placeholder
        return {
            "segments": [
                {"text": "[Segment 1]", "confidence": 0.90},
                {"text": "[Segment 2]", "confidence": 0.88},
            ]
        }

    async def _call_whisper(
        self,
        audio_bytes: bytes,
        language: str,
    ) -> Optional[dict]:
        """Call OpenAI Whisper API.

        Placeholder for actual API call.

        Args:
            audio_bytes: Raw audio bytes
            language: Language code

        Returns:
            Dict with text
        """
        # Placeholder
        return {
            "text": "[Whisper transcription]",
        }
