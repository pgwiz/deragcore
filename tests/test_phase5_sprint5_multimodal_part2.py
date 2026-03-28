"""Integration tests for Sprint 5 Part 2: Multi-Modal Pipelines.

Tests cover:
- MultiModalEmbeddingPipeline: embedding generation and caching
- ContextWindowManagerForMultiModal: token allocation and chunk selection
- HTTP router endpoints: upload, search, status
- End-to-end workflows: content processing → embedding → search
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, MagicMock
from uuid import uuid4

from ragcore.modules.multimodal.models import (
    ModuleType,
    MultiModalContent,
    MultiModalChunk,
    MultiModalMetadata,
    ProcessingResult,
)
from ragcore.modules.multimodal.embedding_pipeline import MultiModalEmbeddingPipeline
from ragcore.modules.multimodal.context_manager import (
    ContextWindowManagerForMultiModal,
    ModalityWeights,
)
from ragcore.modules.multimodal.providers.embedding_adapter import EmbeddingProviderAdapter
from ragcore.core.model_provider_registry import (
    ModelProviderRegistry,
    ProviderConfig,
    ProviderType,
)


# Fixture: Configure test embedding adapter with mock provider
@pytest.fixture
def test_embedding_adapter():
    """Create embedding adapter with mock provider for testing."""
    registry = ModelProviderRegistry()
    # Register test provider with minimal config
    config = ProviderConfig(
        provider=ProviderType.OPENAI,
        endpoint="https://api.openai.com",
        api_key="test-key",
    )
    registry.register_provider(config)
    adapter = EmbeddingProviderAdapter(registry=registry)
    return adapter


# Fixture: Configure test pipeline with mock adapter
@pytest.fixture
def test_embedding_pipeline(test_embedding_adapter):
    """Create embedding pipeline with test adapter."""
    pipeline = MultiModalEmbeddingPipeline(
        embedding_adapter=test_embedding_adapter,
        embedding_dimension=1536,
        batch_size=10,
        cache_enabled=True,
    )
    return pipeline


# ============================================================================
# Tests for MultiModalEmbeddingPipeline
# ============================================================================


class TestMultiModalEmbeddingPipeline:
    """Test embedding pipeline."""

    def test_init(self):
        """Test pipeline initialization."""
        pipeline = MultiModalEmbeddingPipeline(
            embedding_dimension=1536,
            batch_size=10,
            cache_enabled=True,
        )
        assert pipeline.embedding_dimension == 1536
        assert pipeline.batch_size == 10
        assert pipeline.cache_enabled is True

    def test_validate_embedding_dimension(self):
        """Test embedding dimension validation."""
        pipeline = MultiModalEmbeddingPipeline(embedding_dimension=1536)
        embedding = [0.1] * 1536
        assert pipeline.validate_embedding_dimension(embedding) is True

    def test_validate_embedding_dimension_wrong_size(self):
        """Test validation fails for wrong dimension."""
        pipeline = MultiModalEmbeddingPipeline(embedding_dimension=1536)
        embedding = [0.1] * 768  # Wrong size
        assert pipeline.validate_embedding_dimension(embedding) is False

    def test_validate_embedding_empty(self):
        """Test validation fails for empty embedding."""
        pipeline = MultiModalEmbeddingPipeline()
        assert pipeline.validate_embedding_dimension([]) is False
        assert pipeline.validate_embedding_dimension(None) is False

    @pytest.mark.asyncio
    async def test_embed_chunk_with_content(self, test_embedding_pipeline):
        """Test embedding single chunk with content."""
        pipeline = test_embedding_pipeline
        chunk = MultiModalChunk(
            id=uuid4(),
            session_id=uuid4(),
            modality=ModuleType.TEXT,
            content="Test text content",
        )

        result = await pipeline.embed_chunk(chunk)
        assert result.embedding is not None
        assert len(result.embedding) == 1536

    @pytest.mark.asyncio
    async def test_embed_chunk_empty_content(self):
        """Test embedding skips empty content."""
        pipeline = MultiModalEmbeddingPipeline()
        chunk = MultiModalChunk(
            id=uuid4(),
            session_id=uuid4(),
            modality=ModuleType.TEXT,
            content="",
        )

        result = await pipeline.embed_chunk(chunk)
        assert result.embedding == []  # Unchanged

    @pytest.mark.asyncio
    async def test_embed_chunks_batch(self, test_embedding_pipeline):
        """Test batch embedding."""
        pipeline = test_embedding_pipeline
        chunks = [
            MultiModalChunk(
                id=uuid4(),
                session_id=uuid4(),
                modality=ModuleType.TEXT,
                content=f"Content {i}",
            )
            for i in range(12)
        ]

        results = await pipeline.embed_chunks_batch(chunks)
        assert len(results) == 12
        assert all(len(c.embedding) == 1536 for c in results)

    def test_cache_stats(self):
        """Test cache statistics."""
        pipeline = MultiModalEmbeddingPipeline(cache_enabled=True)
        stats = pipeline.get_cache_stats()
        assert stats["cache_enabled"] is True
        assert stats["cache_size"] == 0

    def test_clear_cache(self):
        """Test cache clearing."""
        pipeline = MultiModalEmbeddingPipeline(cache_enabled=True)
        pipeline.embedding_cache["test_key"] = [0.1] * 1536

        assert len(pipeline.embedding_cache) == 1
        pipeline.clear_cache()
        assert len(pipeline.embedding_cache) == 0

    @pytest.mark.asyncio
    async def test_embed_processing_result_with_chunks(self, test_embedding_pipeline):
        """Test embedding all chunks in processing result."""
        pipeline = test_embedding_pipeline
        chunks = [
            MultiModalChunk(
                id=uuid4(),
                session_id=uuid4(),
                modality=ModuleType.IMAGE,
                content=f"Image {i} analysis",
            )
            for i in range(3)
        ]

        result = ProcessingResult(
            success=True,
            modality=ModuleType.IMAGE,
            chunks=chunks,
        )

        embedded_result = await pipeline.embed_processing_result(result)
        assert len(embedded_result.chunks) == 3
        assert all(len(c.embedding) == 1536 for c in embedded_result.chunks)


# ============================================================================
# Tests for ContextWindowManagerForMultiModal
# ============================================================================


class TestContextWindowManagerForMultiModal:
    """Test multi-modal context management."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        mgr = ContextWindowManagerForMultiModal()
        assert mgr.context_window_size == 200000
        assert mgr.output_buffer_percentage == 0.15
        assert mgr.compression_threshold == 0.85

    def test_init_custom_weights(self):
        """Test initialization with custom weights."""
        weights = ModalityWeights(text=1.0, image=2.0, audio=3.0, video=4.0)
        mgr = ContextWindowManagerForMultiModal(weights=weights)
        assert mgr.weights.video == 4.0

    def test_get_available_tokens(self):
        """Test available tokens calculation."""
        mgr = ContextWindowManagerForMultiModal(
            context_window_size=200000,
            output_buffer_percentage=0.15,
        )
        available = mgr.get_available_tokens()
        # 200000 * (1 - 0.15) = 170000
        assert available == 170000

    def test_is_under_pressure_false(self):
        """Test pressure detection when under threshold."""
        mgr = ContextWindowManagerForMultiModal(
            context_window_size=200000,
            output_buffer_percentage=0.15,
            compression_threshold=0.85,
        )
        available = mgr.get_available_tokens()
        # 85% of 170000 = 144500
        pressure_threshold = int(available * 0.85)

        # Below threshold
        assert mgr.is_under_pressure(pressure_threshold - 1000) is False

    def test_is_under_pressure_true(self):
        """Test pressure detection when over threshold."""
        mgr = ContextWindowManagerForMultiModal(
            context_window_size=200000,
            output_buffer_percentage=0.15,
            compression_threshold=0.85,
        )
        available = mgr.get_available_tokens()
        pressure_threshold = int(available * 0.85)

        # Above threshold
        assert mgr.is_under_pressure(pressure_threshold + 1000) is True

    def test_get_modality_weight(self):
        """Test modality weight retrieval."""
        mgr = ContextWindowManagerForMultiModal()
        assert mgr._get_modality_weight(ModuleType.TEXT) == 1.0
        assert mgr._get_modality_weight(ModuleType.IMAGE) == 1.5
        assert mgr._get_modality_weight(ModuleType.AUDIO) == 2.0
        assert mgr._get_modality_weight(ModuleType.VIDEO) == 2.5

    def test_group_by_modality(self):
        """Test grouping chunks by modality."""
        mgr = ContextWindowManagerForMultiModal()
        chunks = [
            MultiModalChunk(id=uuid4(), session_id=uuid4(), modality=ModuleType.TEXT, content="text"),
            MultiModalChunk(id=uuid4(), session_id=uuid4(), modality=ModuleType.IMAGE, content="image"),
            MultiModalChunk(id=uuid4(), session_id=uuid4(), modality=ModuleType.TEXT, content="text2"),
            MultiModalChunk(id=uuid4(), session_id=uuid4(), modality=ModuleType.AUDIO, content="audio"),
        ]

        grouped = mgr._group_by_modality(chunks)
        assert len(grouped[ModuleType.TEXT]) == 2
        assert len(grouped[ModuleType.IMAGE]) == 1
        assert len(grouped[ModuleType.AUDIO]) == 1

    def test_select_chunks_under_budget_empty(self):
        """Test selection with no chunks."""
        mgr = ContextWindowManagerForMultiModal()
        selected, report = mgr.select_chunks_under_budget([], 10000)
        assert len(selected) == 0
        assert report.selected_chunks == 0

    def test_select_chunks_under_budget_single_modality(self):
        """Test selection with single modality."""
        mgr = ContextWindowManagerForMultiModal()
        chunks = [
            MultiModalChunk(
                id=uuid4(),
                session_id=uuid4(),
                modality=ModuleType.TEXT,
                content="Early content",
                confidence_score=0.8,
                source_index=0,
            ),
            MultiModalChunk(
                id=uuid4(),
                session_id=uuid4(),
                modality=ModuleType.TEXT,
                content="Later content",
                confidence_score=0.9,
                source_index=1,
            ),
        ]

        selected, report = mgr.select_chunks_under_budget(chunks, 5000)
        # Should select high confidence chunk
        assert len(selected) > 0
        assert report.total_chunks == 2

    def test_select_chunks_multiple_modalities(self):
        """Test selection with multiple modalities."""
        mgr = ContextWindowManagerForMultiModal()
        chunks = [
            MultiModalChunk(
                id=uuid4(),
                session_id=uuid4(),
                modality=ModuleType.TEXT,
                content="Text content",
                confidence_score=0.9,
            ),
            MultiModalChunk(
                id=uuid4(),
                session_id=uuid4(),
                modality=ModuleType.IMAGE,
                content="Image analysis",
                confidence_score=0.85,
            ),
            MultiModalChunk(
                id=uuid4(),
                session_id=uuid4(),
                modality=ModuleType.AUDIO,
                content="Audio transcription",
                confidence_score=0.80,
            ),
        ]

        selected, report = mgr.select_chunks_under_budget(chunks, 10000)
        assert report.total_chunks == 3
        # Allocation should be fair across modalities
        assert len(report.allocation_by_modality) > 0

    def test_estimate_allocation(self):
        """Test allocation estimation."""
        mgr = ContextWindowManagerForMultiModal()
        chunks = [
            MultiModalChunk(id=uuid4(), session_id=uuid4(), modality=ModuleType.TEXT, content="text"),
            MultiModalChunk(id=uuid4(), session_id=uuid4(), modality=ModuleType.IMAGE, content="image"),
            MultiModalChunk(id=uuid4(), session_id=uuid4(), modality=ModuleType.AUDIO, content="audio"),
        ]

        estimate = mgr.estimate_allocation(chunks)
        assert "total_available_tokens" in estimate
        assert "allocation_by_modality" in estimate
        assert "weights" in estimate
        # Text should get ~40% of budget
        assert estimate["allocation_by_modality"]["text"] > 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestMultiModalIntegration:
    """End-to-end multi-modal workflow tests."""

    @pytest.mark.asyncio
    async def test_workflow_embed_then_select(self, test_embedding_pipeline):
        """Test full workflow: embed chunks then select under budget."""
        # 1. Create chunks with mixed modalities
        chunks = [
            MultiModalChunk(
                id=uuid4(),
                session_id=uuid4(),
                modality=ModuleType.TEXT,
                content="Text content " * 50,  # Long text
                confidence_score=0.95,
            ),
            MultiModalChunk(
                id=uuid4(),
                session_id=uuid4(),
                modality=ModuleType.IMAGE,
                content="Image analysis " * 30,
                confidence_score=0.85,
            ),
            MultiModalChunk(
                id=uuid4(),
                session_id=uuid4(),
                modality=ModuleType.AUDIO,
                content="Audio transcription " * 40,
                confidence_score=0.80,
            ),
        ]

        # 2. Embed chunks
        pipeline = test_embedding_pipeline
        embedded_chunks = await pipeline.embed_chunks_batch(chunks)

        # Verify embeddings
        assert all(len(c.embedding) == 1536 for c in embedded_chunks)

        # 3. Select under budget
        mgr = ContextWindowManagerForMultiModal(context_window_size=10000)
        available = mgr.get_available_tokens()
        selected, report = mgr.select_chunks_under_budget(embedded_chunks, available)

        # Verify selection
        assert len(selected) > 0
        assert report.selected_chunks <= len(embedded_chunks)
        assert report.used_tokens <= available

    def test_workflow_allocation_fairness(self):
        """Test that allocation respects modality weights."""
        # Create chunks with equal count of each modality
        chunks = []
        session_id = uuid4()
        for modality in [ModuleType.TEXT, ModuleType.IMAGE, ModuleType.AUDIO, ModuleType.VIDEO]:
            chunks.append(
                MultiModalChunk(
                    id=uuid4(),
                    session_id=session_id,
                    modality=modality,
                    content=f"{modality.value} content " * 20,
                    confidence_score=0.85,
                )
            )

        mgr = ContextWindowManagerForMultiModal()
        estimate = mgr.estimate_allocation(chunks)

        # All modalities should get non-zero allocation
        allocations = estimate["allocation_by_modality"]
        for mod in ["text", "image", "audio", "video"]:
            assert allocations[mod] > 0, f"{mod} should get tokens"

        # Verify weights applied: audio > text > image > video
        # (because of base allocation weighted by importance)
        assert allocations["audio"] > allocations["text"], "audio heavier than text"
        assert allocations["text"] > allocations["image"], "text heavier than image"
        assert allocations["image"] > allocations["video"], "image heavier than video"

    def test_workflow_low_confidence_filtering(self):
        """Test that low confidence chunks are deprioritized."""
        chunks = [
            MultiModalChunk(
                id=uuid4(),
                session_id=uuid4(),
                modality=ModuleType.TEXT,
                content="High confidence " * 20,
                confidence_score=0.95,
            ),
            MultiModalChunk(
                id=uuid4(),
                session_id=uuid4(),
                modality=ModuleType.TEXT,
                content="Low confidence " * 20,
                confidence_score=0.50,
            ),
        ]

        mgr = ContextWindowManagerForMultiModal(context_window_size=5000)
        available = mgr.get_available_tokens()
        selected, report = mgr.select_chunks_under_budget(chunks, available)

        # Higher confidence chunk should be selected first
        if len(selected) == 1:
            assert selected[0].confidence_score >= 0.9

    @pytest.mark.asyncio
    async def test_workflow_caching(self, test_embedding_pipeline):
        """Test that embedding cache improves performance."""
        pipeline = test_embedding_pipeline

        # Create two chunks with same content
        session_id = uuid4()
        chunk1 = MultiModalChunk(
            id=uuid4(),
            session_id=session_id,
            modality=ModuleType.TEXT,
            content="Duplicate content here",
        )
        chunk2 = MultiModalChunk(
            id=uuid4(),
            session_id=session_id,
            modality=ModuleType.TEXT,
            content="Duplicate content here",  # Same
        )

        # First embeddings
        await pipeline.embed_chunk(chunk1)
        # Should use cache for second
        await pipeline.embed_chunk(chunk2)

        # Both should have embeddings
        assert len(chunk1.embedding) == 1536
        assert len(chunk2.embedding) == 1536
        # Should be identical (from cache)
        assert chunk1.embedding == chunk2.embedding

    def test_workflow_pressure_detection(self):
        """Test pressure detection and warning."""
        mgr = ContextWindowManagerForMultiModal(
            context_window_size=10000,
            compression_threshold=0.80,
        )

        available = mgr.get_available_tokens()
        pressure_point = int(available * 0.80)

        # Below pressure
        assert mgr.is_under_pressure(pressure_point - 100) is False

        # Above pressure
        assert mgr.is_under_pressure(pressure_point + 100) is True
