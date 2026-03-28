"""Context window manager for multi-modal content - dynamic token allocation."""

import logging
from typing import Optional, Dict, List, Any, Tuple
from uuid import UUID
from dataclasses import dataclass
from datetime import datetime

from ragcore.modules.multimodal.models import (
    MultiModalChunk,
    ModuleType,
)
from ragcore.core.token_counter import TokenCounter

logger = logging.getLogger(__name__)


@dataclass
class ModalityWeights:
    """Token weight multipliers by modality for fair context allocation."""

    image: float = 1.5  # Images compressed to text, but analysis heavy
    audio: float = 2.0  # Audio transcribed to text, longer content
    video: float = 2.5  # Video = frames + narration, mostly heavy
    text: float = 1.0  # Pure text, baseline


@dataclass
class MultiModalContextReport:
    """Detailed report of multi-modal context allocation."""

    total_chunks: int = 0
    selected_chunks: int = 0
    total_tokens: int = 0
    available_tokens: int = 0
    used_tokens: int = 0
    allocation_by_modality: Dict[str, Dict[str, Any]] = None
    compression_triggered: bool = False
    warnings: List[str] = None

    def __post_init__(self):
        if self.allocation_by_modality is None:
            self.allocation_by_modality = {}
        if self.warnings is None:
            self.warnings = []


class ContextWindowManagerForMultiModal:
    """Orchestrate multi-modal context assembly within token budget.

    Allocates context window tokens fairly across modalities:
    - Text: 1.0x (baseline)
    - Images: 1.5x (visual analysis)
    - Audio: 2.0x (speech duration)
    - Video: 2.5x (frames + audio)

    Under budget pressure, drops low-confidence chunks, prioritizes
    by modality significance and relevance.
    """

    def __init__(
        self,
        context_window_size: int = 200000,
        output_buffer_percentage: float = 0.15,
        compression_threshold: float = 0.85,
        weights: Optional[ModalityWeights] = None,
    ):
        """Initialize multi-modal context manager.

        Args:
            context_window_size: Total context window tokens
            output_buffer_percentage: Reserve for output (e.g., 0.15 = 15%)
            compression_threshold: Trigger compression at this % (0.85 = 85%)
            weights: Modality weight multipliers (custom or default)
        """
        self.context_window_size = context_window_size
        self.output_buffer_percentage = output_buffer_percentage
        self.compression_threshold = compression_threshold
        self.weights = weights or ModalityWeights()
        self.token_counter = TokenCounter()

    def select_chunks_under_budget(
        self,
        chunks: List[MultiModalChunk],
        available_tokens: int,
        system_overhead_tokens: int = 1000,
    ) -> Tuple[List[MultiModalChunk], MultiModalContextReport]:
        """Intelligently select chunks to fit within token budget.

        Allocation strategy:
        1. Reserve system overhead (1000 tokens typical)
        2. Calculate available tokens for content
        3. Allocate by modality weight:
           - Text: 40% of available
           - Audio: 30% of available
           - Images: 20% of available
           - Video: 10% of available
        4. Within each modality, select by confidence desc + recency
        5. If space remains, take secondary modality chunks

        Args:
            chunks: All available chunks
            available_tokens: Total tokens available
            system_overhead_tokens: Reserve for system (typical 1000)

        Returns:
            (selected_chunks, report) tuple
        """
        report = MultiModalContextReport(
            total_chunks=len(chunks),
            available_tokens=available_tokens,
        )

        if not chunks:
            return [], report

        # Reserve system overhead
        content_tokens = available_tokens - system_overhead_tokens
        if content_tokens < 1000:
            logger.warning("Very limited token budget after overhead")
            report.warnings.append(f"Limited budget: only {content_tokens} tokens for content")
            return [], report

        # Group chunks by modality
        chunks_by_modality = self._group_by_modality(chunks)

        # Calculate allocation per modality
        allocations = self._calculate_modality_allocations(content_tokens, chunks_by_modality)

        logger.info(
            f"Multi-modal allocation: text={allocations.get('text', 0)} tokens, "
            f"audio={allocations.get('audio', 0)} tokens, "
            f"image={allocations.get('image', 0)} tokens, "
            f"video={allocations.get('video', 0)} tokens"
        )

        # Select chunks per modality
        selected = []
        tokens_used = 0

        for modality_str, max_tokens in allocations.items():
            if max_tokens < 100:
                continue

            modality = ModuleType(modality_str)
            modality_chunks = chunks_by_modality.get(modality, [])

            if not modality_chunks:
                continue

            # Sort by confidence (high first) then by source_index recency
            sorted_chunks = sorted(
                modality_chunks,
                key=lambda c: (c.confidence_score, c.source_index),
                reverse=True,
            )

            # Greedily select chunks
            for chunk in sorted_chunks:
                chunk_tokens = self._estimate_chunk_tokens(chunk)
                if tokens_used + chunk_tokens <= max_tokens:
                    selected.append(chunk)
                    tokens_used += chunk_tokens
                else:
                    break

            report.allocation_by_modality[modality_str] = {
                "count": len([c for c in selected if c.modality == modality]),
                "tokens": sum(self._estimate_chunk_tokens(c) for c in selected if c.modality == modality),
                "available": max_tokens,
            }

        report.selected_chunks = len(selected)
        report.used_tokens = tokens_used
        report.compression_triggered = (tokens_used / available_tokens) > self.compression_threshold

        logger.info(
            f"Selected {len(selected)}/{len(chunks)} chunks, "
            f"using {tokens_used}/{available_tokens} tokens "
            f"({100 * tokens_used / available_tokens:.1f}%)"
        )

        return selected, report

    def _group_by_modality(self, chunks: List[MultiModalChunk]) -> Dict[ModuleType, List[MultiModalChunk]]:
        """Group chunks by modality.

        Args:
            chunks: List of chunks

        Returns:
            Dict mapping modality → list of chunks
        """
        grouped = {}
        for chunk in chunks:
            if chunk.modality not in grouped:
                grouped[chunk.modality] = []
            grouped[chunk.modality].append(chunk)
        return grouped

    def _calculate_modality_allocations(
        self,
        total_tokens: int,
        chunks_by_modality: Dict[ModuleType, List[MultiModalChunk]],
    ) -> Dict[str, int]:
        """Calculate token allocation per modality.

        Strategy: weight by modality importance + chunk count.

        Args:
            total_tokens: Total tokens available
            chunks_by_modality: Chunks grouped by modality

        Returns:
            Dict mapping modality string → token allocation
        """
        # Base allocation: text 40%, audio 30%, image 20%, video 10%
        # Adjust if modalities absent
        modalities_present = list(chunks_by_modality.keys())

        if not modalities_present:
            return {}

        # Calculate weighted tokens per modality
        weights_map = {
            ModuleType.TEXT: self.weights.text * 0.40,
            ModuleType.AUDIO: self.weights.audio * 0.30,
            ModuleType.IMAGE: self.weights.image * 0.20,
            ModuleType.VIDEO: self.weights.video * 0.10,
        }

        # Normalize to only present modalities
        present_weight_map = {mod: weights_map[mod] for mod in modalities_present}
        total_weight = sum(present_weight_map.values())

        allocations = {}
        for modality in modalities_present:
            weight_fraction = present_weight_map[modality] / total_weight
            allocated = int(total_tokens * weight_fraction)
            allocations[modality.value] = allocated

        return allocations

    def _estimate_chunk_tokens(self, chunk: MultiModalChunk) -> int:
        """Estimate tokens for a chunk.

        Args:
            chunk: Chunk to estimate

        Returns:
            Estimated token count
        """
        # Use actual token counter for content
        if chunk.content:
            base_tokens = self.token_counter.count_tokens(chunk.content)
        else:
            base_tokens = 50

        # Apply modality weight multiplier for fair comparison
        multiplier = self._get_modality_weight(chunk.modality)
        return int(base_tokens * multiplier)

    def _get_modality_weight(self, modality: ModuleType) -> float:
        """Get weight multiplier for modality.

        Args:
            modality: Modality type

        Returns:
            Weight multiplier (1.0-2.5)
        """
        if modality == ModuleType.TEXT:
            return self.weights.text
        elif modality == ModuleType.IMAGE:
            return self.weights.image
        elif modality == ModuleType.AUDIO:
            return self.weights.audio
        elif modality == ModuleType.VIDEO:
            return self.weights.video
        return 1.0

    def get_available_tokens(self) -> int:
        """Get available tokens for content (after output buffer).

        Returns:
            Token count available for input
        """
        buffer_tokens = int(self.context_window_size * self.output_buffer_percentage)
        return self.context_window_size - buffer_tokens

    def is_under_pressure(self, current_tokens: int) -> bool:
        """Check if approaching token limit.

        Args:
            current_tokens: Current token count

        Returns:
            True if > compression threshold
        """
        available = self.get_available_tokens()
        return current_tokens > (available * self.compression_threshold)

    def estimate_allocation(
        self,
        chunks: List[MultiModalChunk],
    ) -> Dict[str, Any]:
        """Estimate how chunks would be allocated without actually selecting.

        Args:
            chunks: Chunks to analyze

        Returns:
            Allocation estimate report
        """
        available = self.get_available_tokens()
        chunks_by_modality = self._group_by_modality(chunks)
        allocations = self._calculate_modality_allocations(available, chunks_by_modality)

        estimate = {
            "total_available_tokens": available,
            "total_chunks": len(chunks),
            "chunks_by_modality": {str(k): len(v) for k, v in chunks_by_modality.items()},
            "allocation_by_modality": allocations,
            "weights": {
                "text": self.weights.text,
                "image": self.weights.image,
                "audio": self.weights.audio,
                "video": self.weights.video,
            },
        }
        return estimate
