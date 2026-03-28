"""Base processor for multi-modal content extraction."""

import logging
from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID

from ragcore.modules.multimodal.models import (
    MultiModalContent,
    MultiModalMetadata,
    ModuleType,
    ProcessingResult,
)

logger = logging.getLogger(__name__)


class BaseModalityProcessor(ABC):
    """Abstract base class for modality-specific processors."""

    def __init__(self, modality: ModuleType):
        """Initialize processor for specific modality.

        Args:
            modality: Type of content this processor handles (image, audio, video)
        """
        self.modality = modality
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    async def process(
        self,
        content: MultiModalContent,
        session_id: UUID,
        **kwargs,
    ) -> ProcessingResult:
        """Process multi-modal content and extract text/embeddings.

        Args:
            content: Multi-modal content to process
            session_id: Session ID for context
            **kwargs: Additional processor-specific arguments

        Returns:
            ProcessingResult with chunks and metadata
        """
        pass

    @abstractmethod
    def validate_content(self, content: MultiModalContent) -> bool:
        """Validate that content is suitable for processing.

        Args:
            content: Content to validate

        Returns:
            True if valid, False otherwise
        """
        pass

    def get_supported_formats(self) -> list:
        """Get list of supported formats for this modality.

        Returns:
            List of supported format strings
        """
        return []

    def estimate_processing_time_ms(self, content: MultiModalContent) -> float:
        """Estimate processing time for content.

        Args:
            content: Content to estimate

        Returns:
            Estimated time in milliseconds
        """
        return 0.0

    def estimate_tokens_used(self, content: MultiModalContent) -> int:
        """Estimate token usage for LLM-based processing.

        Args:
            content: Content to estimate

        Returns:
            Estimated token count
        """
        return 0

    async def _log_processing_start(self, content: MultiModalContent):
        """Log start of processing."""
        size_mb = content.get_size_mb()
        self.logger.info(
            f"Starting {self.modality.value} processing: "
            f"size={size_mb:.2f}MB, file={content.metadata.file_name}"
        )

    async def _log_processing_end(self, result: ProcessingResult):
        """Log end of processing with results."""
        self.logger.info(
            f"Completed {self.modality.value} processing: "
            f"success={result.success}, chunks={len(result.chunks)}, "
            f"time={result.processing_time_ms:.0f}ms, tokens={result.tokens_used}"
        )

    async def _log_processing_error(self, error: Exception):
        """Log processing error."""
        self.logger.error(
            f"Error in {self.modality.value} processing: {str(error)}",
            exc_info=True,
        )
