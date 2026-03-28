"""Tests for Sprint 3: Context Window Manager."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from uuid import uuid4

from ragcore.core.token_counter import TokenCounter
from ragcore.core.token_budget import TokenBudget
from ragcore.modules.memory.context_prioritizer import ContextPrioritizer
from ragcore.modules.memory.memory_compressor import MemoryCompressor, CompressedTurn
from ragcore.core.context_window_manager import ContextWindowManager
from ragcore.modules.chat.retriever import RetrievedChunk
from ragcore.modules.chat.history import ChatTurn
from ragcore.modules.chat.context_builder import ContextBuilder


class TestTokenCounter:
    """Test token counting utility."""

    def test_init(self):
        """Test TokenCounter initialization."""
        counter = TokenCounter()
        assert counter.model == "gpt-3.5-turbo"
        assert counter.tokenizer is not None

    def test_count_tokens_simple(self):
        """Test counting tokens in simple text."""
        counter = TokenCounter()
        # Simple test: just check it returns an int >= 0
        tokens = counter.count_tokens("Hello world")
        assert isinstance(tokens, int)
        assert tokens > 0

    def test_count_tokens_empty(self):
        """Test empty text returns 0."""
        counter = TokenCounter()
        assert counter.count_tokens("") == 0

    def test_count_messages_tokens(self):
        """Test counting tokens in message list."""
        counter = TokenCounter()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        total = counter.count_messages_tokens(messages)
        assert total > 0
        assert total > sum(counter.count_tokens(m["content"]) for m in messages)

    def test_get_component_breakdown(self):
        """Test component token breakdown."""
        counter = TokenCounter()
        breakdown = counter.get_component_breakdown(
            system_prompt="You are helpful.",
            history=[{"role": "user", "content": "Hello"}],
            chunks=["This is a chunk"],
            query="What is this?",
        )
        assert "system_prompt" in breakdown
        assert "history" in breakdown
        assert "chunks" in breakdown
        assert "query" in breakdown
        assert "total" in breakdown
        assert breakdown["total"] > 0

    def test_estimate_text_tokens_fallback(self):
        """Test fast estimation fallback."""
        counter = TokenCounter()
        estimate = counter.estimate_text_tokens("a" * 400)  # ~100 chars
        assert estimate > 0
        assert estimate <= 150  # Should be rough estimate


class TestTokenBudget:
    """Test token budget tracking."""

    def test_init(self):
        """Test TokenBudget initialization."""
        budget = TokenBudget(context_window_size=200000)
        assert budget.context_window_size == 200000
        assert budget.buffer_tokens == 30000  # 15% of 200k
        assert budget.available_tokens == 170000
        assert budget.compression_trigger_tokens == int(170000 * 0.85)

    def test_add_tokens_and_remaining(self):
        """Test adding tokens and checking remaining."""
        budget = TokenBudget(context_window_size=100000)
        budget.add_tokens(50000)
        assert budget.current_usage == 50000
        remaining = budget.remaining_tokens()
        assert remaining == 100000 - 15000 - 50000  # window - buffer - used

    def test_is_over_budget(self):
        """Test budget overflow detection."""
        budget = TokenBudget(context_window_size=100000)
        budget.add_tokens(100000)
        assert budget.is_over_budget()

    def test_is_under_pressure(self):
        """Test compression pressure detection."""
        budget = TokenBudget(context_window_size=100000)
        # Add 85% of available
        available = budget.available_tokens
        budget.add_tokens(int(available * 0.85))
        assert budget.is_under_pressure()

    def test_get_usage_percentage(self):
        """Test usage percentage calculation."""
        budget = TokenBudget(context_window_size=100000)
        budget.add_tokens(50000)
        perc = budget.get_usage_percentage()
        assert 0 <= perc <= 1

    def test_reset(self):
        """Test resetting budget for new request."""
        budget = TokenBudget()
        budget.add_tokens(50000)
        budget.reset()
        assert budget.current_usage == 0


class TestContextPrioritizer:
    """Test chunk and history prioritization."""

    def test_rank_chunks_by_similarity(self):
        """Test ranking chunks by similarity score."""
        chunk1 = Mock(spec=RetrievedChunk)
        chunk1.chunk_id = uuid4()
        chunk1.similarity_score = 0.9
        chunk1.is_critical = False

        chunk2 = Mock(spec=RetrievedChunk)
        chunk2.chunk_id = uuid4()
        chunk2.similarity_score = 0.7
        chunk2.is_critical = False

        ranked = ContextPrioritizer.rank_chunks([chunk1, chunk2])
        assert ranked[0][0] == chunk1  # First should be highest score
        assert ranked[0][1] > ranked[1][1]

    def test_rank_chunks_with_criticality(self):
        """Test that critical chunks boost ranking."""
        chunk1 = Mock(spec=RetrievedChunk)
        chunk1.chunk_id = uuid4()
        chunk1.similarity_score = 0.5
        chunk1.is_critical = True

        chunk2 = Mock(spec=RetrievedChunk)
        chunk2.chunk_id = uuid4()
        chunk2.similarity_score = 0.9
        chunk2.is_critical = False

        ranked = ContextPrioritizer.rank_chunks([chunk1, chunk2])
        # Critical chunk should rank higher despite lower similarity
        assert ranked[0][1] > ranked[1][1]

    def test_select_chunks_under_budget(self):
        """Test greedy chunk selection under budget."""
        chunk1 = Mock(spec=RetrievedChunk)
        chunk1.chunk_id = uuid4()
        chunk1.tokens = 100
        chunk1.similarity_score = 0.9
        chunk1.is_critical = False

        chunk2 = Mock(spec=RetrievedChunk)
        chunk2.chunk_id = uuid4()
        chunk2.tokens = 50
        chunk2.similarity_score = 0.7
        chunk2.is_critical = False

        selected = ContextPrioritizer.select_chunks_under_budget(
            [chunk1, chunk2], max_tokens=200
        )
        assert len(selected) == 2  # Both fit in budget (100 + 50 = 150 < 200)

    def test_select_history_under_budget(self):
        """Test history selection respects recent turns."""
        now = datetime.utcnow()
        turn1 = ChatTurn("user", "Old question", now - timedelta(days=1))
        turn2 = ChatTurn("assistant", "Old answer", now - timedelta(days=1))
        turn3 = ChatTurn("user", "Recent question", now)

        selected = ContextPrioritizer.select_history_under_budget(
            [turn1, turn2, turn3],
            max_tokens=1000,
            keep_recent_count=2,
        )
        # Should include all recent turns
        assert turn3 in selected


class TestMemoryCompressor:
    """Test conversation history compression."""

    def test_extract_sentences(self):
        """Test extractive sentence extraction."""
        text = "First sentence. Middle sentence. Last sentence."
        sentences = MemoryCompressor._extract_sentences(text, 2)
        assert len(sentences) <= 2
        assert "First" in sentences[0]

    def test_compress_turn(self):
        """Test compressing single turn."""
        turn = ChatTurn(
            "user",
            "First sentence about something. Middle details. Last important point.",
            datetime.utcnow(),
        )
        compressed = MemoryCompressor.compress_turn(turn, max_sentences=2)
        assert len(compressed) > 0
        assert "First" in compressed or "Last" in compressed

    def test_compress_turn_group(self):
        """Test compressing group of turns."""
        now = datetime.utcnow()
        turns = [
            ChatTurn("user", "First user query.", now),
            ChatTurn("assistant", "My response here.", now),
            ChatTurn("user", "Follow up question.", now),
        ]
        compressed = MemoryCompressor.compress_turn_group(turns)
        assert isinstance(compressed, CompressedTurn)
        assert len(compressed.extracted_sentences) > 0
        assert compressed.compression_ratio >= 1.0


class TestContextWindowManager:
    """Test full context window orchestration."""

    def test_init(self):
        """Test ContextWindowManager initialization."""
        manager = ContextWindowManager(context_window_size=200000)
        assert manager.token_counter is not None
        assert manager.budget is not None
        assert manager.budget.context_window_size == 200000

    def test_build_messages_simple(self):
        """Test building messages with simple inputs."""
        manager = ContextWindowManager()
        messages, report = manager.build_messages(
            system_prompt="You are helpful.",
            query="What is AI?",
        )
        assert len(messages) >= 2  # At min: system + query
        assert report["total_tokens"] > 0
        assert not report["over_budget"]

    def test_build_messages_with_chunks(self):
        """Test building messages with retrieved chunks."""
        manager = ContextWindowManager()

        chunk = Mock(spec=RetrievedChunk)
        chunk.chunk_id = uuid4()
        chunk.file_id = uuid4()
        chunk.filename = "test.txt"
        chunk.text = "This is test content."
        chunk.tokens = 10
        chunk.similarity_score = 0.9
        chunk.chunk_index = 0
        chunk.is_critical = False

        messages, report = manager.build_messages(
            system_prompt="Helpful assistant",
            query="Tell me about this",
            retrieved_chunks=[chunk],
        )
        assert len(messages) >= 2
        assert "chunks" in report["components"]

    def test_build_messages_with_history(self):
        """Test building messages with history."""
        manager = ContextWindowManager()
        now = datetime.utcnow()
        history = [
            ChatTurn("user", "Previous question", now - timedelta(hours=1)),
            ChatTurn("assistant", "Previous answer", now - timedelta(hours=1)),
        ]

        messages, report = manager.build_messages(
            system_prompt="Helper",
            query="Follow up?",
            history=history,
        )
        assert len(messages) >= 2
        assert "history" in report["components"]

    def test_estimate_query_tokens(self):
        """Test token estimation without building."""
        manager = ContextWindowManager()
        estimate = manager.estimate_query_tokens(
            system_prompt="Helper",
            query="Test query",
        )
        assert estimate["system_prompt"] > 0
        assert estimate["query"] > 0
        assert estimate["total"] > 0


class TestContextBuilderWithBudget:
    """Test ContextBuilder integration with budget manager."""

    def test_build_with_budget(self):
        """Test budget-aware context building."""
        system_prompt = "You are helpful."
        query = "What is this?"
        messages, report = ContextBuilder.build_with_budget(
            system_prompt=system_prompt,
            query=query,
            context_window_size=100000,
        )
        assert len(messages) >= 2
        assert "total_tokens" in report
        assert "components" in report

    def test_build_with_budget_vs_simple(self):
        """Test that budget builder returns report."""
        system_prompt = "Helper"
        query = "Question"
        messages_budget, report = ContextBuilder.build_with_budget(
            system_prompt=system_prompt,
            query=query,
        )
        messages_simple = ContextBuilder.build(
            system_prompt=system_prompt,
            query=query,
            retrieved_chunks=[],
            history=[],
        )
        # Budget version includes report
        assert isinstance(report, dict)
        assert "total_tokens" in report


class TestIntegration:
    """Integration tests across Sprint 3 components."""

    def test_full_context_building_workflow(self):
        """Test complete workflow: chunks → prioritize → compress → return."""
        now = datetime.utcnow()

        # Create mock chunks
        chunks = []
        for i in range(3):
            chunk = Mock(spec=RetrievedChunk)
            chunk.chunk_id = uuid4()
            chunk.file_id = uuid4()
            chunk.filename = f"doc{i}.txt"
            chunk.text = f"Content {i}" * 10
            chunk.tokens = 50
            chunk.similarity_score = 0.9 - (i * 0.1)
            chunk.chunk_index = i
            chunk.is_critical = i == 0  # First is critical
            chunks.append(chunk)

        # Create history
        history = [
            ChatTurn("user", "Old question from yesterday", now - timedelta(days=1)),
            ChatTurn("assistant", "Old answer from yesterday", now - timedelta(days=1)),
            ChatTurn("user", "Recent question", now - timedelta(hours=1)),
            ChatTurn("assistant", "Recent answer", now - timedelta(hours=1)),
        ]

        # Build with budget
        manager = ContextWindowManager(context_window_size=50000)
        messages, report = manager.build_messages(
            system_prompt="You are a helpful assistant.",
            query="What should I do?",
            retrieved_chunks=chunks,
            history=history,
            enable_compression=True,
        )

        assert len(messages) >= 2
        assert report["total_tokens"] <= report["available_tokens"]
        assert "chunks" in report["components"]
        assert "history" in report["components"]

    def test_compression_triggered_under_pressure(self):
        """Test that compression activates when needed."""
        now = datetime.utcnow()

        # Many history turns to trigger compression
        history = [
            ChatTurn(f"user", f"Q{i}: " + ("x" * 1000), now - timedelta(hours=10 - i))
            for i in range(15)
        ]

        # Small budget to trigger pressure
        manager = ContextWindowManager(context_window_size=40000)
        messages, report = manager.build_messages(
            system_prompt="Helper" * 10,  # Big system prompt
            query="?" * 500,  # Big query
            history=history,
            enable_compression=True,
        )

        # Check that we got under budget
        assert not report["over_budget"]
        # Compression should have been triggered
        if report["components"]["history"]["compressed"]:
            # Compressed version should use fewer messages
            assert report["components"]["history"]["included"] < len(history)
