"""Multi-modal content models for images, audio, and video."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from enum import Enum


class ModuleType(str, Enum):
    """Supported modality types."""
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TEXT = "text"


class ImageFormat(str, Enum):
    """Supported image formats."""
    JPEG = "jpeg"
    JPG = "jpg"
    PNG = "png"
    WEBP = "webp"
    GIF = "gif"


class AudioFormat(str, Enum):
    """Supported audio formats."""
    MP3 = "mp3"
    WAV = "wav"
    OGG = "ogg"
    M4A = "m4a"


class VideoFormat(str, Enum):
    """Supported video formats."""
    MP4 = "mp4"
    WEBM = "webm"
    MOV = "mov"
    AVI = "avi"


@dataclass
class MultiModalMetadata:
    """Metadata for multi-modal content."""

    modality: ModuleType
    source_url: Optional[str] = None
    file_name: Optional[str] = None
    file_size_bytes: int = 0
    duration_seconds: Optional[float] = None  # For audio/video
    width: Optional[int] = None  # For images/video
    height: Optional[int] = None  # For images/video
    fps: Optional[float] = None  # For video
    bit_rate_kbps: Optional[int] = None  # For audio/video
    sample_rate_hz: Optional[int] = None  # For audio
    channels: Optional[int] = None  # For audio
    processed_at: datetime = field(default_factory=datetime.utcnow)
    extraction_method: str = ""  # e.g., "claude_vision", "azure_speech", "whisper"
    processing_time_ms: float = 0.0
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiModalChunk:
    """Represents a chunk extracted from multi-modal content."""

    id: UUID
    session_id: UUID
    memory_id: Optional[UUID] = None  # Link to parent memory if exists
    modality: ModuleType = ModuleType.TEXT
    content: str = ""  # Extracted text
    embedding: List[float] = field(default_factory=list)  # 1536-dim vector
    metadata: MultiModalMetadata = field(default_factory=lambda: MultiModalMetadata(modality=ModuleType.TEXT))
    source_index: int = 0  # For multi-frame/multi-segment content (frame num, segment, etc)
    confidence_score: float = 1.0  # Extraction confidence (0.0-1.0)
    is_critical: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
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
        }


@dataclass
class MultiModalContent:
    """Represents raw multi-modal content before processing."""

    id: UUID
    session_id: UUID
    modality: ModuleType
    raw_content: bytes  # Binary content (image/audio/video) or text
    text_content: Optional[str] = None  # For text modality or extracted text
    metadata: MultiModalMetadata = field(default_factory=lambda: MultiModalMetadata(modality=ModuleType.TEXT))

    # Processing state
    is_processed: bool = False
    processed_chunks: List[MultiModalChunk] = field(default_factory=list)
    processing_error: Optional[str] = None

    # Storage references
    storage_path: Optional[str] = None  # S3/Blob/local path
    is_base64_inline: bool = False  # True if content < 100KB and stored inline

    created_at: datetime = field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None

    def get_size_mb(self) -> float:
        """Get content size in MB."""
        if isinstance(self.raw_content, bytes):
            return len(self.raw_content) / (1024 * 1024)
        return 0.0

    def should_inline(self, max_inline_kb: int = 100) -> bool:
        """Check if content should be stored inline vs S3/Blob."""
        size_kb = len(self.raw_content) / 1024 if isinstance(self.raw_content, bytes) else 0
        return size_kb <= max_inline_kb


@dataclass
class ProcessingResult:
    """Result from modality-specific processing."""

    success: bool
    modality: ModuleType
    chunks: List[MultiModalChunk] = field(default_factory=list)
    error_message: Optional[str] = None
    processing_time_ms: float = 0.0
    tokens_used: int = 0  # For LLM-based extraction
    extracted_text: Optional[str] = None  # Full extracted text
    confidence_scores: List[float] = field(default_factory=list)  # Per-chunk confidence
