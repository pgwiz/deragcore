"""Context building - Assemble final messages for AI completion."""

import logging
from typing import List, Dict, Any, Optional

from ragcore.modules.chat.history import ChatTurn
from ragcore.modules.chat.retriever import RetrievedChunk
from ragcore.config import settings

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Build context windows for RAG completions."""

    @staticmethod
    def build(
        system_prompt: str,
        query: str,
        retrieved_chunks: List[RetrievedChunk],
        history: List[ChatTurn],
    ) -> List[Dict[str, str]]:
        """
        Assemble final message list for AI provider.

        Message structure:
        1. System prompt (if provided)
        2. Full conversation history (all prior turns - no truncation)
        3. Context block with retrieved chunks
        4. Current user query

        Args:
            system_prompt: System instruction/persona
            query: Current user query
            retrieved_chunks: Retrieved context chunks
            history: Conversation history turns

        Returns:
            List of messages ready for AIController
        """
        messages: List[Dict[str, str]] = []

        # =====================================================================
        # 1. System Prompt (as first message if provided)
        # =====================================================================
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })
            logger.debug("Added system prompt")

        # =====================================================================
        # 2. Full Conversation History (no truncation - user decision from Phase 2)
        # =====================================================================
        for turn in history:
            messages.append(turn.to_message())

        # =====================================================================
        # 3. Context Block with Retrieved Chunks
        # =====================================================================
        if retrieved_chunks:
            context_sections = []

            for i, chunk in enumerate(retrieved_chunks, 1):
                context_sections.append(
                    f"[Source {i}: {chunk.filename} - chunk {chunk.chunk_index}]\n"
                    f"{chunk.text}"
                )

            context_block = "\n\n".join(context_sections)

            context_message = (
                f"Based on the following retrieved documents, answer the query:\n\n"
                f"{context_block}"
            )

            messages.append({
                "role": "user",
                "content": context_message,
            })

            logger.debug(
                f"Added context block with {len(retrieved_chunks)} chunks "
                f"({sum(c.tokens for c in retrieved_chunks)} tokens)"
            )

        # =====================================================================
        # 4. Current Query
        # =====================================================================
        messages.append({
            "role": "user",
            "content": query,
        })

        logger.debug(
            f"Built context: system_prompt={bool(system_prompt)}, "
            f"history_turns={len(history)}, "
            f"retrieved={len(retrieved_chunks)}, "
            f"total_messages={len(messages)}"
        )

        return messages

    @staticmethod
    def estimate_tokens(messages: List[Dict[str, str]]) -> int:
        """
        Rough estimate of total tokens in messages.

        Uses simple heuristic: ~4 chars per token average.

        Args:
            messages: Message list

        Returns:
            Rough token count
        """
        total_chars = sum(
            len(msg.get("content", "")) for msg in messages
        )
        return max(1, total_chars // 4)

    @staticmethod
    def format_sources(
        chunks: List[RetrievedChunk],
    ) -> List[Dict[str, Any]]:
        """
        Format source attribution for response.

        Args:
            chunks: Retrieved chunks

        Returns:
            List of source dicts with metadata
        """
        return [
            {
                "chunk_id": str(chunk.chunk_id),
                "file_id": str(chunk.file_id),
                "filename": chunk.filename,
                "similarity_score": chunk.similarity_score,
                "excerpt": chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text,
            }
            for chunk in chunks
        ]

    @staticmethod
    def build_compound(
        system_prompt: str,
        query: str,
        retrieved_chunks: Optional[List[RetrievedChunk]] = None,
        research_findings: Optional[str] = None,
        research_sources: Optional[List[Dict[str, str]]] = None,
        history: Optional[List[ChatTurn]] = None,
    ) -> List[Dict[str, str]]:
        """
        Assemble context for compound mode (documents + web search).

        Distinguishes sources: [DOC: name] vs [WEB: source]

        Args:
            system_prompt: Compound mode system instruction
            query: User question
            retrieved_chunks: File chunks (optional)
            research_findings: Web research synthesis (optional)
            research_sources: Web source references (optional)
            history: Conversation history (optional)

        Returns:
            List of messages ready for AIController
        """
        messages: List[Dict[str, str]] = []
        history = history or []
        retrieved_chunks = retrieved_chunks or []
        research_sources = research_sources or []

        # System prompt
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })
            logger.debug("Added compound mode system prompt")

        # History
        for turn in history:
            messages.append(turn.to_message())

        # =====================================================================
        # Context Block: Mixed Sources [DOC:] + [WEB:]
        # =====================================================================

        context_sections = []

        # Document chunks first
        if retrieved_chunks:
            context_sections.append("=== DOCUMENT SOURCES ===\n")
            for i, chunk in enumerate(retrieved_chunks, 1):
                context_sections.append(
                    f"[DOC {i}: {chunk.filename}]\n{chunk.text}"
                )

        # Web research findings
        if research_findings:
            context_sections.append("\n=== WEB RESEARCH SOURCES ===\n")
            context_sections.append(research_findings)

            # Add source references
            if research_sources:
                context_sections.append("\n=== RESEARCH SOURCES ===")
                for i, src in enumerate(research_sources[:5], 1):
                    context_sections.append(
                        f"[WEB {i}: {src.get('title', 'Source')}]\n"
                        f"URL: {src.get('url', 'N/A')}"
                    )

        if context_sections:
            context_block = "\n\n".join(context_sections)

            context_message = (
                "Based on the following documents and web research, answer the query:\n\n"
                f"{context_block}"
            )

            messages.append({
                "role": "user",
                "content": context_message,
            })

            logger.debug(
                f"Added compound context: "
                f"{len(retrieved_chunks)} doc chunks, "
                f"{len(research_sources)} web sources"
            )

        # Current query
        messages.append({
            "role": "user",
            "content": query,
        })

        logger.debug(
            f"Built compound context: "
            f"docs={len(retrieved_chunks)}, "
            f"web_sources={len(research_sources)}, "
            f"total_messages={len(messages)}"
        )

        return messages
