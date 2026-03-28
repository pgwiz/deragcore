"""Memory compressor - Compress old conversation turns via extractive summarization."""

import logging
import re
from typing import List, Tuple, Optional
from datetime import datetime

from ragcore.modules.chat.history import ChatTurn
from ragcore.core.token_budget import TokenBudget

logger = logging.getLogger(__name__)


class CompressedTurn:
    """Represents a compressed turn group with extracted key points."""

    def __init__(
        self,
        turns: List[ChatTurn],
        extracted_sentences: List[str],
        compression_ratio: float,
    ):
        """Initialize compressed turn.

        Args:
            turns: Original turns that were compressed
            extracted_sentences: Key sentences extracted (first + last + important)
            compression_ratio: Ratio of original_tokens / compressed_tokens
        """
        self.turns = turns
        self.extracted_sentences = extracted_sentences
        self.compression_ratio = compression_ratio
        self.created_at = datetime.utcnow()

    def to_summary_text(self) -> str:
        """Get compressed summary as single text block.

        Returns:
            Summary text combining extracted sentences
        """
        return "\n".join(self.extracted_sentences)


class MemoryCompressor:
    """Compress conversation history via extractive summarization."""

    @staticmethod
    def should_compress(budget: TokenBudget) -> bool:
        """Check if compression should be triggered.

        Compress if under pressure (85%+ full).

        Args:
            budget: Token budget

        Returns:
            True if compression recommended
        """
        return budget.is_under_pressure()

    @staticmethod
    def _extract_sentences(text: str, num_sentences: int = 2) -> List[str]:
        """Extract first and last sentences from text (extractive summarization).

        Strategy: Extract first sentence + last sentence + any marked important.

        Args:
            text: Text to extract from
            num_sentences: Target number of sentences to extract

        Returns:
            List of extracted sentences
        """
        if not text:
            return []

        # Split by sentence-ending punctuation
        sentence_pattern = r"(?<=[.!?])\s+"
        sentences = re.split(sentence_pattern, text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return []

        if len(sentences) <= num_sentences:
            return sentences

        # Extraction strategy: first + last
        extracted = [sentences[0]]

        if len(sentences) > 1 and num_sentences > 1:
            extracted.append(sentences[-1])

        # Remove duplicates
        extracted = list(dict.fromkeys(extracted))
        logger.debug(f"Extracted {len(extracted)} sentences from {len(sentences)} total")

        return extracted

    @staticmethod
    def compress_turn(
        turn: ChatTurn,
        max_sentences: int = 2,
    ) -> str:
        """Compress a single turn to key sentences.

        Args:
            turn: Chat turn to compress
            max_sentences: Maximum sentences to extract

        Returns:
            Compressed text
        """
        if not turn.content:
            return ""

        extracted = MemoryCompressor._extract_sentences(turn.content, max_sentences)
        return " ".join(extracted)

    @staticmethod
    def compress_turn_group(
        turns: List[ChatTurn],
        num_summary_sentences: int = 2,
    ) -> CompressedTurn:
        """Compress a group of turns via extractive summarization.

        Strategy:
        1. Extract first sentence from first turn
        2. Extract last sentence from last turn
        3. Extract any marked-important sentences (if available)
        4. Join extracted sentences

        Args:
            turns: Turns to compress (typically older turns)
            num_summary_sentences: Sentences to extract per turn

        Returns:
            CompressedTurn with extracted key points
        """
        if not turns:
            return CompressedTurn([], [], 1.0)

        extracted_sentences = []

        for turn in turns:
            if turn.content:
                sentences = MemoryCompressor._extract_sentences(
                    turn.content, num_summary_sentences
                )
                extracted_sentences.extend(sentences)

        # Calculate compression ratio
        original_chars = sum(len(turn.content or "") for turn in turns)
        compressed_chars = sum(len(s) for s in extracted_sentences)
        compression_ratio = (
            original_chars / max(1, compressed_chars) if compressed_chars > 0 else 1.0
        )

        logger.debug(
            f"Compressed {len(turns)} turns: "
            f"{original_chars} chars → {compressed_chars} chars "
            f"(ratio={compression_ratio:.2f}x)"
        )

        return CompressedTurn(turns, extracted_sentences, compression_ratio)

    @staticmethod
    def rebuild_history_with_compression(
        turns: List[ChatTurn],
        budget: TokenBudget,
        keep_recent_turns: int = 3,
        compress_group_size: int = 3,
    ) -> Tuple[List[ChatTurn], Optional[CompressedTurn]]:
        """Rebuild history with compression applied to old turns.

        Strategy:
        1. Keep most recent N turns as-is
        2. Group older turns and compress via extraction
        3. Return: recent_turns + [compressed_summary turn]

        Args:
            turns: All history turns
            budget: Token budget (check if under pressure)
            keep_recent_turns: Number of recent turns to keep intact
            compress_group_size: Number of turns to group for compression

        Returns:
            Tuple of (filtered_turns, compressed_summary or None)
        """
        if not turns or not budget.is_under_pressure():
            return turns, None

        # Separate recent and older turns
        recent_turns = turns[-keep_recent_turns:] if len(turns) > keep_recent_turns else turns
        older_turns = turns[:-keep_recent_turns] if len(turns) > keep_recent_turns else []

        if not older_turns:
            return turns, None

        # Compress older turns
        compressed = MemoryCompressor.compress_turn_group(
            older_turns, num_summary_sentences=2
        )

        logger.info(
            f"Compressed {len(older_turns)} old turns to {len(compressed.extracted_sentences)} "
            f"sentences (ratio={compressed.compression_ratio:.2f}x)"
        )

        return recent_turns, compressed

    @staticmethod
    def get_compression_summary(compressed: CompressedTurn) -> str:
        """Get human-readable compression summary.

        Args:
            compressed: Compressed turn data

        Returns:
            Summary string for logging/display
        """
        num_original = len(compressed.turns)
        num_extracted = len(compressed.extracted_sentences)
        ratio = compressed.compression_ratio

        return (
            f"Compressed {num_original} turns into {num_extracted} key sentences "
            f"({ratio:.2f}x reduction)"
        )
