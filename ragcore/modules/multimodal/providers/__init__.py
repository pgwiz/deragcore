"""Provider adapters for multi-modal processing.

Abstracts vendor-specific API calls (vision, audio, embeddings) behind
a unified interface that uses ModelProviderRegistry for runtime provider selection.
"""

from ragcore.modules.multimodal.providers.base_adapter import BaseProviderAdapter
from ragcore.modules.multimodal.providers.vision_adapter import VisionProviderAdapter
from ragcore.modules.multimodal.providers.audio_adapter import AudioProviderAdapter
from ragcore.modules.multimodal.providers.embedding_adapter import EmbeddingProviderAdapter

__all__ = [
    "BaseProviderAdapter",
    "VisionProviderAdapter",
    "AudioProviderAdapter",
    "EmbeddingProviderAdapter",
]
