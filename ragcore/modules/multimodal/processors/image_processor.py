"""Image processor - Claude Vision with Azure OCR fallback."""

import logging
from typing import Optional
from uuid import UUID, uuid4
import base64
import time

from ragcore.modules.multimodal.models import (
    MultiModalContent,
    MultiModalChunk,
    MultiModalMetadata,
    ModuleType,
    ProcessingResult,
    ImageFormat,
)
from ragcore.modules.multimodal.processors.base import BaseModalityProcessor

logger = logging.getLogger(__name__)


class ImageProcessor(BaseModalityProcessor):
    """Process images using Claude Vision with Azure OCR fallback."""

    def __init__(
        self,
        claude_client=None,
        azure_client=None,
        embedding_model=None,
    ):
        """Initialize image processor.

        Args:
            claude_client: Claude API client for vision
            azure_client: Azure Vision API client for fallback
            embedding_model: Embedding model for generating vectors
        """
        super().__init__(ModuleType.IMAGE)
        self.claude_client = claude_client
        self.azure_client = azure_client
        self.embedding_model = embedding_model

    def get_supported_formats(self) -> list:
        """Get supported image formats."""
        return [fmt.value for fmt in ImageFormat]

    def validate_content(self, content: MultiModalContent) -> bool:
        """Validate image content.

        Args:
            content: Content to validate

        Returns:
            True if valid image content
        """
        if content.modality != ModuleType.IMAGE:
            return False

        # Check file format if provided
        if content.metadata.file_name:
            ext = content.metadata.file_name.lower().split(".")[-1]
            if ext not in [fmt.value for fmt in ImageFormat]:
                self.logger.warning(f"Unsupported image format: {ext}")
                return False

        # Check size (max 20MB for image)
        if content.get_size_mb() > 20:
            self.logger.warning("Image too large (>20MB)")
            return False

        return True

    def estimate_processing_time_ms(self, content: MultiModalContent) -> float:
        """Estimate image processing time.

        Args:
            content: Image content

        Returns:
            Estimated time in ms (vision API: 2-5 seconds)
        """
        return 3000.0  # ~3 seconds average

    def estimate_tokens_used(self, content: MultiModalContent) -> int:
        """Estimate tokens for image processing.

        Claude Vision tokens depend on image size:
        - Small image (<1MB): ~300 tokens
        - Medium image (1-5MB): ~600 tokens
        - Large image (5-20MB): ~1200 tokens

        Args:
            content: Image content

        Returns:
            Estimated token count
        """
        size_mb = content.get_size_mb()
        if size_mb < 1:
            return 300
        elif size_mb < 5:
            return 600
        else:
            return 1200

    async def process(
        self,
        content: MultiModalContent,
        session_id: UUID,
        extraction_prompt: Optional[str] = None,
        **kwargs,
    ) -> ProcessingResult:
        """Process image using Claude Vision with fallback.

        Args:
            content: Image content to process
            session_id: Session ID for context
            extraction_prompt: Optional custom extraction prompt
            **kwargs: Additional arguments (use_azure_only=False)

        Returns:
            ProcessingResult with extracted text and metadata
        """
        start_time = time.time()
        await self._log_processing_start(content)

        # Validate content
        if not self.validate_content(content):
            result = ProcessingResult(
                success=False,
                modality=ModuleType.IMAGE,
                error_message="Invalid image content",
                processing_time_ms=0,
            )
            await self._log_processing_error(Exception("Invalid content"))
            return result

        try:
            # Try Claude Vision first (unless Azure-only requested)
            use_azure_only = kwargs.get("use_azure_only", False)
            if not use_azure_only and self.claude_client:
                result = await self._process_with_claude(content, session_id, extraction_prompt)
                if result.success:
                    result.processing_time_ms = (time.time() - start_time) * 1000
                    await self._log_processing_end(result)
                    return result
                self.logger.warning("Claude Vision failed, falling back to Azure")

            # Fallback to Azure OCR
            if self.azure_client:
                result = await self._process_with_azure(content, session_id)
                result.processing_time_ms = (time.time() - start_time) * 1000
                await self._log_processing_end(result)
                return result

            # All backends failed
            return ProcessingResult(
                success=False,
                modality=ModuleType.IMAGE,
                error_message="No vision backend available (Claude and Azure both failed)",
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        except Exception as e:
            await self._log_processing_error(e)
            return ProcessingResult(
                success=False,
                modality=ModuleType.IMAGE,
                error_message=str(e),
                processing_time_ms=(time.time() - start_time) * 1000,
            )

    async def _process_with_claude(
        self,
        content: MultiModalContent,
        session_id: UUID,
        extraction_prompt: Optional[str] = None,
    ) -> ProcessingResult:
        """Process image with Claude Vision.

        Args:
            content: Image content
            session_id: Session ID
            extraction_prompt: Custom extraction prompt

        Returns:
            ProcessingResult from Claude Vision
        """
        try:
            # Convert image to base64 if not already
            if isinstance(content.raw_content, bytes):
                image_data = base64.b64encode(content.raw_content).decode("utf-8")
            else:
                image_data = content.raw_content

            # Default extraction prompt
            if not extraction_prompt:
                extraction_prompt = (
                    "Analyze this image and extract all text, key concepts, visual elements, "
                    "and important information. Be comprehensive and detailed."
                )

            # Call Claude Vision
            # This is a placeholder - would use actual Claude API in production
            response = await self._call_claude_vision(
                image_data,
                extraction_prompt,
                content.metadata.file_name,
            )

            if not response or not response.get("text"):
                return ProcessingResult(
                    success=False,
                    modality=ModuleType.IMAGE,
                    error_message="Claude Vision returned empty response",
                )

            extracted_text = response.get("text", "")
            tokens_used = response.get("tokens_used", self.estimate_tokens_used(content))

            # Create single chunk for entire image
            chunk = MultiModalChunk(
                id=uuid4(),
                session_id=session_id,
                modality=ModuleType.IMAGE,
                content=extracted_text,
                embedding=[],  # Will be filled by embedding pipeline
                metadata=content.metadata,
                source_index=0,
                confidence_score=response.get("confidence", 0.95),
                is_critical=False,
            )

            return ProcessingResult(
                success=True,
                modality=ModuleType.IMAGE,
                chunks=[chunk],
                extracted_text=extracted_text,
                processing_time_ms=0,  # Set by caller
                tokens_used=tokens_used,
                confidence_scores=[chunk.confidence_score],
            )

        except Exception as e:
            self.logger.error(f"Claude Vision processing failed: {e}")
            return ProcessingResult(
                success=False,
                modality=ModuleType.IMAGE,
                error_message=f"Claude Vision error: {str(e)}",
            )

    async def _process_with_azure(
        self,
        content: MultiModalContent,
        session_id: UUID,
    ) -> ProcessingResult:
        """Process image with Azure Vision API (OCR).

        Args:
            content: Image content
            session_id: Session ID

        Returns:
            ProcessingResult from Azure
        """
        try:
            # This is a placeholder - would use actual Azure SDK in production
            response = await self._call_azure_ocr(content.raw_content)

            if not response or not response.get("text"):
                return ProcessingResult(
                    success=False,
                    modality=ModuleType.IMAGE,
                    error_message="Azure OCR returned empty response",
                )

            extracted_text = response.get("text", "")

            # Create chunk for extracted text
            chunk = MultiModalChunk(
                id=uuid4(),
                session_id=session_id,
                modality=ModuleType.IMAGE,
                content=extracted_text,
                embedding=[],
                metadata=content.metadata,
                source_index=0,
                confidence_score=response.get("confidence", 0.80),
                is_critical=False,
            )

            return ProcessingResult(
                success=True,
                modality=ModuleType.IMAGE,
                chunks=[chunk],
                extracted_text=extracted_text,
                processing_time_ms=0,  # Set by caller
                tokens_used=0,  # Azure OCR doesn't use generative tokens
                confidence_scores=[chunk.confidence_score],
            )

        except Exception as e:
            self.logger.error(f"Azure OCR processing failed: {e}")
            return ProcessingResult(
                success=False,
                modality=ModuleType.IMAGE,
                error_message=f"Azure OCR error: {str(e)}",
            )

    async def _call_claude_vision(
        self,
        image_data: str,
        prompt: str,
        filename: Optional[str] = None,
    ) -> Optional[dict]:
        """Call Claude Vision API.

        Placeholder for actual API call.

        Args:
            image_data: Base64 encoded image
            prompt: Extraction prompt
            filename: Original filename

        Returns:
            Dict with text, tokens_used, confidence
        """
        # Placeholder
        return {
            "text": f"[Claude Vision analysis of {filename}]",
            "tokens_used": 300,
            "confidence": 0.95,
        }

    async def _call_azure_ocr(self, image_bytes: bytes) -> Optional[dict]:
        """Call Azure Vision API for OCR.

        Placeholder for actual API call.

        Args:
            image_bytes: Raw image bytes

        Returns:
            Dict with text and confidence
        """
        # Placeholder
        return {
            "text": "[Azure OCR extracted text]",
            "confidence": 0.85,
        }
