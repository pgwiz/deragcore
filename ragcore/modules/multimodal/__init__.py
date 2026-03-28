"""Multi-modal content processing module."""

from ragcore.modules.multimodal.models import (
    ModuleType,
    MultiModalContent,
    MultiModalChunk,
    MultiModalMetadata,
    ProcessingResult,
)
from ragcore.modules.multimodal.processors.image_processor import ImageProcessor
from ragcore.modules.multimodal.processors.audio_processor import AudioProcessor
from ragcore.modules.multimodal.processors.video_processor import VideoProcessor
from ragcore.modules.multimodal.embedding_pipeline import MultiModalEmbeddingPipeline
from ragcore.modules.multimodal.context_manager import (
    ContextWindowManagerForMultiModal,
    ModalityWeights,
    MultiModalContextReport,
)

__all__ = [
    "ModuleType",
    "MultiModalContent",
    "MultiModalChunk",
    "MultiModalMetadata",
    "ProcessingResult",
    "ImageProcessor",
    "AudioProcessor",
    "VideoProcessor",
    "MultiModalEmbeddingPipeline",
    "ContextWindowManagerForMultiModal",
    "ModalityWeights",
    "MultiModalContextReport",
]
