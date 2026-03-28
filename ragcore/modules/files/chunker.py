"""Text chunking - Split documents into overlapping chunks with token awareness."""

import logging
from typing import List, Dict, Any
import tiktoken

logger = logging.getLogger(__name__)


class TextChunker:
    """Split text into overlapping chunks with token awareness."""

    def __init__(self, chunk_size_tokens: int = 512, chunk_overlap_tokens: int = 50):
        """
        Initialize chunker with token-based sizing.

        Args:
            chunk_size_tokens: Target tokens per chunk (will be adjusted per actual text)
            chunk_overlap_tokens: Overlap between chunks in tokens
        """
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens

        # Use cl100k_base encoding (GPT-3.5/4 tokenizer)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        logger.info(f"TextChunker initialized: size={chunk_size_tokens}, overlap={chunk_overlap_tokens}")

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using GPT tokenizer."""
        return len(self.tokenizer.encode(text))

    def _split_by_delimiters(self, text: str, delimiters: List[str]) -> List[str]:
        """
        Recursively split text by delimiters, preserving structure.

        Strategy: Try primary delimiter, fall back to next if any chunk exceeds threshold.
        """
        if not delimiters or not text.strip():
            return [text] if text.strip() else []

        delimiter = delimiters[0]
        remaining_delimiters = delimiters[1:]

        splits = text.split(delimiter)

        # Check if any split is too large
        max_split_tokens = max(self._count_tokens(s) for s in splits) if splits else 0

        # If largest split fits or no more delimiters, return these splits
        if max_split_tokens <= self.chunk_size_tokens * 1.5 or not remaining_delimiters:
            return splits

        # Otherwise, recursively split using next delimiter
        result = []
        for split in splits:
            if self._count_tokens(split) > self.chunk_size_tokens * 1.5:
                result.extend(self._split_by_delimiters(split, remaining_delimiters))
            else:
                result.append(split)

        return result

    def chunk(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks with token-aware sizing and overlap.

        Strategy: Progressive delimiter-based splitting
        1. Split by double newlines (paragraphs)
        2. Split by single newlines (lines)
        3. Split by sentences (. or !)
        4. Split by spaces (words)
        5. Character-level as fallback

        Args:
            text: Full document text
            metadata: File metadata to include in chunk metadata

        Returns:
            List of chunks: [{
                'text': str,
                'tokens': int,
                'chunk_index': int,
                'metadata': {original_metadata + 'chunk_offset': char_offset}
            }]
        """
        if not text.strip():
            logger.warning("Empty text provided to chunker")
            return []

        metadata = metadata or {}

        # Progressive delimiters to try
        delimiters = [
            "\n\n",  # Paragraph boundary (strongest)
            "\n",    # Line boundary
            ". ",    # Sentence boundary
            "! ",
            "? ",
            " ",     # Word boundary
            "",      # Character fallback
        ]

        # Split recursively
        splits = self._split_by_delimiters(text, delimiters)
        splits = [s.strip() for s in splits if s.strip()]

        if not splits:
            logger.warning("No content after splitting")
            return []

        # Group splits into chunks with overlap
        chunks: List[Dict[str, Any]] = []
        current_chunk_texts = []
        current_tokens = 0
        char_offset = 0

        for split_text in splits:
            split_tokens = self._count_tokens(split_text)

            # If adding this split would exceed limit, finalize current chunk
            if current_tokens + split_tokens > self.chunk_size_tokens and current_chunk_texts:
                chunk_text = " ".join(current_chunk_texts)
                chunk_tokens = self._count_tokens(chunk_text)

                chunks.append({
                    "text": chunk_text,
                    "tokens": chunk_tokens,
                    "chunk_index": len(chunks),
                    "metadata": {
                        **metadata,
                        "char_offset": char_offset - len(chunk_text),  # Approximate
                    },
                })

                # Keep last portion for overlap
                if self.chunk_overlap_tokens > 0:
                    # Find how many splits to keep for overlap
                    overlap_text_parts = []
                    overlap_tokens = 0
                    for i in range(len(current_chunk_texts) - 1, -1, -1):
                        text_part = current_chunk_texts[i]
                        part_tokens = self._count_tokens(text_part)
                        if overlap_tokens + part_tokens <= self.chunk_overlap_tokens:
                            overlap_text_parts.insert(0, text_part)
                            overlap_tokens += part_tokens
                        else:
                            break
                    current_chunk_texts = overlap_text_parts
                    current_tokens = overlap_tokens
                else:
                    current_chunk_texts = []
                    current_tokens = 0

            current_chunk_texts.append(split_text)
            current_tokens += split_tokens
            char_offset += len(split_text) + 1  # +1 for delimiter

        # Finalize last chunk
        if current_chunk_texts:
            chunk_text = " ".join(current_chunk_texts)
            chunk_tokens = self._count_tokens(chunk_text)

            chunks.append({
                "text": chunk_text,
                "tokens": chunk_tokens,
                "chunk_index": len(chunks),
                "metadata": {
                    **metadata,
                    "char_offset": char_offset - len(chunk_text),
                },
            })

        logger.info(f"Chunked text into {len(chunks)} chunks (avg {sum(c['tokens'] for c in chunks) / len(chunks):.0f} tokens)")

        return chunks
