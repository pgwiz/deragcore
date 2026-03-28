"""Context window manager - Orchestrate token counting, budgeting, prioritization, and compression."""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from ragcore.core.token_counter import TokenCounter
from ragcore.core.token_budget import TokenBudget
from ragcore.modules.memory.context_prioritizer import ContextPrioritizer
from ragcore.modules.memory.memory_compressor import MemoryCompressor, CompressedTurn
from ragcore.modules.chat.retriever import RetrievedChunk
from ragcore.modules.chat.history import ChatTurn
from ragcore.config import settings

logger = logging.getLogger(__name__)


class ContextWindowManager:
    """Orchestrate all context window management operations.

    Responsibilities:
    - Track token usage via TokenCounter
    - Enforce budget limits via TokenBudget
    - Prioritize chunks and history via ContextPrioritizer
    - Compress old content via MemoryCompressor
    - Build optimized message list within budget
    """

    def __init__(
        self,
        context_window_size: int = 200000,
        output_buffer_percentage: float = 0.15,
        compression_threshold: float = 0.85,
    ):
        """Initialize context window manager.

        Args:
            context_window_size: Total context window in tokens (default: 200000 for Claude)
            output_buffer_percentage: Fraction reserved for output (default: 0.15 = 15%)
            compression_threshold: Trigger compression at this usage fraction (default: 0.85 = 85%)
        """
        self.token_counter = TokenCounter()
        self.budget = TokenBudget(
            context_window_size=context_window_size,
            output_buffer_percentage=output_buffer_percentage,
            compression_threshold=compression_threshold,
        )
        self.last_build_report = None

        logger.info(
            f"ContextWindowManager initialized: "
            f"window={context_window_size}, "
            f"buffer={self.budget.buffer_tokens}, "
            f"compression_at={self.budget.compression_trigger_tokens}"
        )

    def build_messages(
        self,
        system_prompt: Optional[str] = None,
        query: str = "",
        retrieved_chunks: Optional[List[RetrievedChunk]] = None,
        history: Optional[List[ChatTurn]] = None,
        enable_compression: bool = True,
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
        """Build final message list within context budget.

        Strategy:
        1. Count system prompt tokens (always include)
        2. Count query tokens (always include)
        3. Prioritize and fit chunks
        4. Prioritize and fit history, applying compression if needed
        5. Return messages + detailed report

        Args:
            system_prompt: System instruction (always included)
            query: Current user query (always included)
            retrieved_chunks: Retrieved context chunks (prioritized)
            history: Conversation history (prioritized, may be compressed)
            enable_compression: Allow compressing old history

        Returns:
            Tuple of (messages, report)
            - messages: Ready for AIController
            - report: Dict with token breakdown and decisions
        """
        self.budget.reset()
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "steps": [],
            "components": {},
            "warnings": [],
            "over_budget": False,
        }

        messages = []

        # =====================================================================
        # 1. System Prompt (always include)
        # =====================================================================
        system_tokens = 0
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            system_tokens = self.token_counter.count_tokens(system_prompt)
            self.budget.add_tokens(system_tokens)
            report["steps"].append(
                f"Added system prompt: {system_tokens} tokens"
            )

        report["components"]["system_prompt"] = system_tokens

        # =====================================================================
        # 2. Query (always include)
        # =====================================================================
        query_tokens = self.token_counter.count_tokens(query)
        self.budget.add_tokens(query_tokens)
        report["components"]["query"] = query_tokens
        report["steps"].append(f"Added query: {query_tokens} tokens")

        # =====================================================================
        # 3. Chunks (prioritize under budget)
        # =====================================================================
        chunks_budget = self.budget.remaining_tokens() * 0.4  # Allocate 40% of remaining
        chunks_message = None
        chunks_token_count = 0

        if retrieved_chunks and chunks_budget > 0:
            selected_chunks = ContextPrioritizer.select_chunks_under_budget(
                retrieved_chunks,
                int(chunks_budget),
                self.token_counter,
            )

            if selected_chunks:
                context_sections = []
                for i, chunk in enumerate(selected_chunks, 1):
                    context_sections.append(
                        f"[Source {i}: {chunk.filename} - chunk {chunk.chunk_index}]\n"
                        f"{chunk.text}"
                    )

                chunks_content = "\n\n".join(context_sections)
                chunks_message = f"Based on the following retrieved documents:\n\n{chunks_content}"
                chunks_token_count = self.token_counter.count_tokens(chunks_message)

                self.budget.add_tokens(chunks_token_count)
                report["steps"].append(
                    f"Added {len(selected_chunks)} of {len(retrieved_chunks)} chunks: "
                    f"{chunks_token_count} tokens"
                )
                report["components"]["chunks"] = {
                    "selected": len(selected_chunks),
                    "total": len(retrieved_chunks),
                    "tokens": chunks_token_count,
                }

        else:
            report["components"]["chunks"] = {"selected": 0, "total": 0, "tokens": 0}

        # =====================================================================
        # 4. History (prioritize with optional compression)
        # =====================================================================
        history_budget = self.budget.remaining_tokens() - 500  # Reserve 500 tokens
        history_tokens = 0
        compressed_summary = None

        if history and history_budget > 0:
            # Check if compression should be applied
            if enable_compression and self.budget.is_under_pressure():
                history_turns, compressed = MemoryCompressor.rebuild_history_with_compression(
                    history,
                    self.budget,
                    keep_recent_turns=3,
                )

                if compressed:
                    compressed_summary = compressed
                    # Add recent turns as normal
                    for turn in history_turns:
                        messages.append(turn.to_message())
                        history_tokens += self.token_counter.count_tokens(turn.content or "")

                    # Add compressed summary as-is context
                    summary_text = compressed.to_summary_text()
                    history_tokens += self.token_counter.count_tokens(summary_text)

                    report["steps"].append(
                        f"Applied compression: {len(compressed.turns)} old turns → "
                        f"{len(compressed.extracted_sentences)} key points "
                        f"({compressed.compression_ratio:.2f}x reduction)"
                    )

                else:
                    # No compression applied, use all history
                    history_turns = history

                    for turn in history_turns:
                        messages.append(turn.to_message())
                        history_tokens += self.token_counter.count_tokens(turn.content or "")

            else:
                # No compression, prioritize recent history
                selected_history = ContextPrioritizer.select_history_under_budget(
                    history,
                    int(history_budget),
                    keep_recent_count=3,
                )

                for turn in selected_history:
                    messages.append(turn.to_message())
                    history_tokens += self.token_counter.count_tokens(turn.content or "")

                if len(selected_history) < len(history):
                    report["steps"].append(
                        f"Dropped {len(history) - len(selected_history)} old history turns"
                    )

            self.budget.add_tokens(history_tokens)
            report["components"]["history"] = {
                "included": len(history) if not compressed_summary else len(history_turns),
                "total": len(history),
                "tokens": history_tokens,
                "compressed": compressed_summary is not None,
            }

        else:
            report["components"]["history"] = {
                "included": 0,
                "total": 0,
                "tokens": 0,
                "compressed": False,
            }

        # =====================================================================
        # 5. Add chunks context message (if any chunks were selected)
        # =====================================================================
        if chunks_message:
            messages.insert(len([m for m in messages if m.get("role") == "system"]), {
                "role": "user",
                "content": chunks_message,
            })

        # =====================================================================
        # 6. Add final query
        # =====================================================================
        messages.append({"role": "user", "content": query})

        # =====================================================================
        # 7. Generate final report
        # =====================================================================
        report["total_tokens"] = self.budget.current_usage
        report["available_tokens"] = self.budget.available_tokens
        report["usage_percentage"] = round(self.budget.get_usage_percentage() * 100, 1)
        report["over_budget"] = self.budget.is_over_budget()
        report["total_messages"] = len(messages)

        if self.budget.is_over_budget():
            report["warnings"].append(
                f"Over budget: {self.budget.current_usage} / {self.budget.available_tokens} tokens"
            )
            logger.warning(f"Context window over budget: {report['warnings'][0]}")

        if self.budget.is_under_pressure():
            report["warnings"].append(
                f"Under compression threshold: {report['usage_percentage']}% full"
            )

        logger.info(
            f"Built context: {len(messages)} messages, "
            f"{self.budget.current_usage}/{self.budget.available_tokens} tokens "
            f"({report['usage_percentage']}%)"
        )

        self.last_build_report = report
        return messages, report

    def get_last_report(self) -> Optional[Dict[str, Any]]:
        """Get last build report.

        Returns:
            Last build report dict, or None if no builds yet
        """
        return self.last_build_report

    def estimate_query_tokens(
        self,
        system_prompt: Optional[str] = None,
        query: str = "",
        chunks: Optional[List[RetrievedChunk]] = None,
    ) -> Dict[str, int]:
        """Estimate tokens for a query without actually building messages.

        Useful for planning/analysis.

        Args:
            system_prompt: System instruction
            query: User query
            chunks: Retrieved chunks

        Returns:
            Dict with token estimates
        """
        estimate = {
            "system_prompt": 0,
            "query": 0,
            "chunks": 0,
            "total": 0,
        }

        if system_prompt:
            estimate["system_prompt"] = self.token_counter.count_tokens(system_prompt)

        estimate["query"] = self.token_counter.count_tokens(query)

        if chunks:
            for chunk in chunks:
                estimate["chunks"] += chunk.tokens or 0

        estimate["total"] = sum(estimate.values())
        return estimate
