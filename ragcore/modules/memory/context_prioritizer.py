"""Context prioritizer - Rank and select chunks under budget constraints."""

import logging
from typing import List, Tuple, Optional
from datetime import datetime, timedelta

from ragcore.modules.chat.retriever import RetrievedChunk
from ragcore.modules.chat.history import ChatTurn

logger = logging.getLogger(__name__)


class ContextPrioritizer:
    """Rank chunks and history by importance for dynamic prioritization."""

    @staticmethod
    def rank_chunks(
        chunks: List[RetrievedChunk],
        query: Optional[str] = None,
        current_time: Optional[datetime] = None,
    ) -> List[Tuple[RetrievedChunk, float]]:
        """Rank chunks by relevance and criticality.

        Higher score = keep this chunk.

        Scoring:
        - Base score: similarity_score (0.0-1.0)
        - Criticality boost: +0.5 if is_critical flag set
        - Result: sorted descending, highest score first

        Args:
            chunks: List of retrieved chunks
            query: Query string (for future semantic reranking)
            current_time: Current time for age calculation

        Returns:
            List of (chunk, score) tuples sorted by score descending
        """
        if not chunks:
            return []

        scored_chunks: List[Tuple[RetrievedChunk, float]] = []

        for chunk in chunks:
            # Base score from similarity
            base_score = getattr(chunk, "similarity_score", 0.5)

            # Criticality boost (check for is_critical flag, default False)
            criticality_boost = 0.5 if getattr(chunk, "is_critical", False) else 0.0

            # Combine scores
            final_score = min(1.0, base_score + criticality_boost)

            scored_chunks.append((chunk, final_score))

        # Sort by score descending (highest first)
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        logger.debug(
            f"Ranked {len(scored_chunks)} chunks, top 3: "
            f"{[(x[0].chunk_id, round(x[1], 3)) for x in scored_chunks[:3]]}"
        )

        return scored_chunks

    @staticmethod
    def select_chunks_under_budget(
        chunks: List[RetrievedChunk],
        max_tokens: int,
        token_counter=None,
    ) -> List[RetrievedChunk]:
        """Greedily select highest-scored chunks until budget exhausted.

        Strategy: Rank chunks by importance, then greedily add until over budget.

        Args:
            chunks: List of all retrieved chunks
            max_tokens: Maximum tokens available for chunks
            token_counter: TokenCounter instance (for accurate counting)

        Returns:
            Selected chunks, sorted by original rank order
        """
        if not chunks or max_tokens <= 0:
            return []

        # Rank chunks
        ranked = ContextPrioritizer.rank_chunks(chunks)

        # Greedily select chunks
        selected = []
        total_tokens = 0

        for chunk, score in ranked:
            chunk_tokens = chunk.tokens or 0

            if total_tokens + chunk_tokens <= max_tokens:
                selected.append(chunk)
                total_tokens += chunk_tokens
            else:
                # Stop adding chunks once budget exhausted
                logger.debug(
                    f"Chunk budget exhausted: {total_tokens}/{max_tokens} tokens, "
                    f"skipping chunk with {chunk_tokens} tokens (score={score:.3f})"
                )
                break

        logger.debug(
            f"Selected {len(selected)} of {len(chunks)} chunks, "
            f"using {total_tokens}/{max_tokens} tokens"
        )

        return selected

    @staticmethod
    def rank_history(
        turns: List[ChatTurn],
        query: Optional[str] = None,
        current_time: Optional[datetime] = None,
    ) -> List[Tuple[ChatTurn, float]]:
        """Rank history turns by recency and relevance.

        Higher score = keep this turn.

        Scoring:
        - Recency: exponential decay (most recent = 1.0)
        - Age decay: score *= exp(-age_seconds / 3600) = reduce by 50% per hour
        - Result: sorted descending

        Args:
            turns: List of chat turns
            query: Query string (unused for now)
            current_time: Current time (default: now)

        Returns:
            List of (turn, score) tuples sorted by score descending
        """
        if not turns:
            return []

        if current_time is None:
            current_time = datetime.utcnow()

        from math import exp

        scored_turns: List[Tuple[ChatTurn, float]] = []

        for turn in turns:
            # Calculate age in seconds
            age_seconds = (current_time - turn.created_at).total_seconds()

            # Exponential decay: -0.5 per hour
            decay_factor = exp(-age_seconds / 3600 * 0.693)  # 0.693 = ln(2)

            # Base score from recency (older = lower)
            score = max(0.1, decay_factor)

            scored_turns.append((turn, score))

        # Sort by score descending
        scored_turns.sort(key=lambda x: x[1], reverse=True)

        logger.debug(
            f"Ranked {len(scored_turns)} history turns, "
            f"most recent score={scored_turns[0][1]:.3f}"
        )

        return scored_turns

    @staticmethod
    def select_history_under_budget(
        turns: List[ChatTurn],
        max_tokens: int,
        keep_recent_count: int = 3,
        token_counter=None,
    ) -> List[ChatTurn]:
        """Select history turns under budget, preferring recent turns.

        Strategy:
        1. Always keep most recent N turns intact
        2. Optionally compress older turns under budget pressure
        3. Return selected turns in chronological order

        Args:
            turns: List of all history turns
            max_tokens: Maximum tokens available for history
            keep_recent_count: Minimum number of recent turns to always keep (default: 3)
            token_counter: TokenCounter instance (for accurate counting)

        Returns:
            Selected turns in chronological order
        """
        if not turns or max_tokens <= 0:
            return []

        # Keep most recent turns intact
        recent_turns = turns[-keep_recent_count:] if len(turns) > keep_recent_count else turns
        older_turns = turns[:-keep_recent_count] if len(turns) > keep_recent_count else []

        # Count tokens in recent turns (always keep)
        recent_tokens = sum(len(turn.content or "") // 4 for turn in recent_turns)

        if recent_tokens >= max_tokens:
            # Budget is too small, just return recent turns
            logger.warning(
                f"Budget too small ({max_tokens} tokens) for recent turns ({recent_tokens} tokens)"
            )
            return recent_turns

        # Greedily add older turns
        remaining_budget = max_tokens - recent_tokens
        selected_older = []

        for turn in reversed(older_turns):
            turn_tokens = len(turn.content or "") // 4
            if turn_tokens <= remaining_budget:
                selected_older.append(turn)
                remaining_budget -= turn_tokens

        # Return in chronological order
        result = selected_older + recent_turns
        logger.debug(
            f"Selected {len(result)} of {len(turns)} history turns, "
            f"usage: {max_tokens - remaining_budget}/{max_tokens} tokens"
        )

        return result
