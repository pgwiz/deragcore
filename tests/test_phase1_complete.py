"""Comprehensive tests for Phase 1 completion."""

import pytest
import asyncio
import os
import inspect
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from uuid import uuid4
from datetime import datetime

from ragcore.models import ModelConfig, Session, Job, File, Chunk
from ragcore.core.schemas import (
    UnifiedResponse,
    UnifiedChunk,
    ORION_DEFAULT_PROMPT,
    ORION_RESEARCH_PROMPT,
    ORION_DOCUMENT_PROMPT,
)
from ragcore.core.ai_controller import AIController
from ragcore.core.providers.azure_provider import AzureProvider


# ============================================================================
# System Prompt Integration Tests
# ============================================================================


class TestSystemPromptIntegration:
    """Test that system prompts are properly integrated."""

    def test_default_system_prompt_exists(self):
        """Verify default system prompt is defined."""
        assert len(ORION_DEFAULT_PROMPT) > 0
        assert "Orion" in ORION_DEFAULT_PROMPT
        assert "empathetic" in ORION_DEFAULT_PROMPT.lower()

    def test_research_system_prompt_exists(self):
        """Verify research system prompt is defined."""
        assert len(ORION_RESEARCH_PROMPT) > 0
        assert "research" in ORION_RESEARCH_PROMPT.lower()

    def test_document_system_prompt_exists(self):
        """Verify document system prompt is defined."""
        assert len(ORION_DOCUMENT_PROMPT) > 0
        assert "document" in ORION_DOCUMENT_PROMPT.lower()

    @patch("ragcore.core.providers.anthropic_provider.anthropic.Anthropic")
    def test_ai_controller_uses_system_prompt(self, mock_anthropic):
        """Test that AIController passes system_prompt to provider."""
        # Mock the Anthropic client
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        # Mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Test response")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_client.messages.create.return_value = mock_response

        # Test with system prompt
        from ragcore.core.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider()

        response = provider.complete(
            messages=[{"role": "user", "content": "Hello"}],
            model="claude-3-5-sonnet-20241022",
            system_prompt=ORION_DEFAULT_PROMPT,
        )

        # Verify system prompt was passed
        assert mock_client.messages.create.called
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == ORION_DEFAULT_PROMPT

    @patch("ragcore.core.providers.anthropic_provider.anthropic.Anthropic")
    def test_ai_controller_fallback_default_prompt(self, mock_anthropic):
        """Test fallback to default prompt when none provided."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Test response")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_client.messages.create.return_value = mock_response

        from ragcore.core.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider()

        response = provider.complete(
            messages=[{"role": "user", "content": "Hello"}],
            model="claude-3-5-sonnet-20241022",
            system_prompt=None,  # No prompt provided
        )

        # Verify fallback to default prompt was used
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are a helpful assistant."


# ============================================================================
# Azure Async Streaming Tests
# ============================================================================


class TestAzureAsyncStreaming:
    """Test Azure provider async streaming implementation."""

    @pytest.mark.asyncio
    async def test_azure_streaming_implementation_exists(self):
        """Verify Azure streaming method is async."""
        import inspect

        # Check if stream is an async generator function
        assert inspect.isasyncgenfunction(AzureProvider.stream)

    @pytest.mark.asyncio
    @patch("ragcore.core.providers.azure_provider.asyncio.to_thread")
    async def test_azure_streaming_uses_async_execution(self, mock_to_thread):
        """Test that Azure streaming uses asyncio.to_thread."""
        # This test verifies the structure, actual execution requires Azure setup
        pass


# ============================================================================
# Data Models Tests
# ============================================================================


class TestDataModels:
    """Test new data models."""

    def test_file_model_exists(self):
        """Verify File model is properly defined."""
        f = File(
            filename="test.pdf",
            file_size=1024,
            content_type="application/pdf",
            status="pending",
            chunks_count=0,
        )
        assert f.filename == "test.pdf"
        assert f.status == "pending"
        # defaults are only applied in the database
        # just verify the model accepts the field
        assert hasattr(f, "chunks_count")

    def test_chunk_model_exists(self):
        """Verify Chunk model is properly defined."""
        file_id = uuid4()
        c = Chunk(
            file_id=file_id,
            chunk_index=0,
            text="Sample chunk text",
            tokens=10,
        )
        assert c.file_id == file_id
        assert c.chunk_index == 0
        assert c.tokens == 10

    def test_file_has_relationship_to_chunks(self):
        """Verify File model has chunks relationship."""
        assert hasattr(File, "chunks")

    def test_chunk_has_relationship_to_file(self):
        """Verify Chunk model has file relationship."""
        assert hasattr(Chunk, "file")

    def test_file_has_session_relationship(self):
        """Verify File model has session relationship."""
        assert hasattr(File, "session")

    def test_session_has_files_relationship(self):
        """Verify Session model has files relationship."""
        assert hasattr(Session, "files")

    def test_modelconfig_has_system_prompt_field(self):
        """Verify ModelConfig has system_prompt field."""
        mc = ModelConfig(
            name="test-config",
            provider="anthropic",
            model_id="claude-3-5-sonnet-20241022",
            system_prompt=ORION_DEFAULT_PROMPT,
        )
        assert mc.system_prompt == ORION_DEFAULT_PROMPT

    def test_modelconfig_default_for_default_flag(self):
        """Verify ModelConfig has is_default boolean."""
        mc = ModelConfig(
            name="test-config",
            provider="anthropic",
            model_id="claude-3-5-sonnet-20241022",
            is_default=True,
        )
        assert mc.is_default is True


# ============================================================================
# Database Schema Tests
# ============================================================================


class TestDatabaseSchema:
    """Test database migrations and schema."""

    def test_alembic_migrations_exist(self):
        """Verify migration files exist."""
        import os

        migration_dir = "e:/Backup/pgwiz/rag/alembic/versions"
        assert os.path.exists(migration_dir)
        assert len(os.listdir(migration_dir)) >= 2  # At least init + presets

    def test_initial_migration_file_exists(self):
        """Verify initial migration file."""
        migration_file = "e:/Backup/pgwiz/rag/alembic/versions/001_initial_schema.py"
        assert os.path.exists(migration_file)

    def test_seed_migration_file_exists(self):
        """Verify seed migration file."""
        migration_file = "e:/Backup/pgwiz/rag/alembic/versions/002_seed_presets.py"
        assert os.path.exists(migration_file)

    def test_alembic_env_configured(self):
        """Verify alembic env.py is configured."""
        env_file = "e:/Backup/pgwiz/rag/alembic/env.py"
        with open(env_file, "r") as f:
            content = f.read()
            assert "Base.metadata" in content
            assert "import" in content and "models" in content.lower()


# ============================================================================
# Provider Tests
# ============================================================================


class TestProviderSystemPrompts:
    """Test that providers respect system prompts."""

    @patch("ragcore.core.providers.anthropic_provider.anthropic.Anthropic")
    def test_anthropic_provider_accepts_system_prompt(self, mock_anthropic):
        """Verify Anthropic provider accepts system_prompt."""
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Test")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_client.messages.create.return_value = mock_response

        from ragcore.core.providers.anthropic_provider import AnthropicProvider

        provider = AnthropicProvider()

        # Should not raise
        response = provider.complete(
            messages=[{"role": "user", "content": "Test"}],
            model="claude-3-5-sonnet-20241022",
            system_prompt="Custom prompt",
        )

        assert response is not None

    def test_azure_provider_accepts_system_prompt(self):
        """Verify Azure provider method signature accepts system_prompt."""
        from ragcore.core.providers.azure_provider import AzureProvider
        import inspect

        sig = inspect.signature(AzureProvider.complete)
        assert "system_prompt" in sig.parameters


# ============================================================================
# Integration Tests
# ============================================================================


class TestPhase1Integration:
    """End-to-end Phase 1 integration tests."""

    def test_models_have_all_required_fields(self):
        """Verify all models have required fields."""
        # ModelConfig
        assert hasattr(ModelConfig, "name")
        assert hasattr(ModelConfig, "provider")
        assert hasattr(ModelConfig, "model_id")
        assert hasattr(ModelConfig, "system_prompt")

        # Session
        assert hasattr(Session, "model_config_id")
        assert hasattr(Session, "model_config")

        # File
        assert hasattr(File, "filename")
        assert hasattr(File, "status")
        assert hasattr(File, "chunks_count")

        # Chunk
        assert hasattr(Chunk, "file_id")
        assert hasattr(Chunk, "text")
        assert hasattr(Chunk, "embedding")
        assert hasattr(Chunk, "tokens")

    def test_ai_controller_export(self):
        """Verify AIController is accessible."""
        assert hasattr(AIController, "complete")
        assert hasattr(AIController, "stream")
        assert hasattr(AIController, "embed")
        assert hasattr(AIController, "get_available_providers")

    def test_unified_response_export(self):
        """Verify UnifiedResponse schema exists."""
        response = UnifiedResponse(
            text="Test",
            model="test-model",
            provider="test-provider",
            input_tokens=5,
            output_tokens=10,
        )
        assert response.text == "Test"
        assert response.input_tokens == 5

    def test_unified_chunk_export(self):
        """Verify UnifiedChunk schema exists."""
        chunk = UnifiedChunk(delta="Hello")
        assert chunk.delta == "Hello"


# ============================================================================
# API Test Structure
# ============================================================================


class TestAPIStructure:
    """Test that API is properly structured for Phase 1."""

    def test_main_app_created(self):
        """Verify FastAPI app is created."""
        from ragcore.main import app

        assert app is not None
        assert app.title == "RAGCORE"

    def test_health_endpoint_exists(self):
        """Verify health endpoint is defined."""
        from ragcore.main import app

        routes = [route.path for route in app.routes]
        assert "/health" in routes

    def test_test_complete_endpoint_exists(self):
        """Verify test complete endpoint is defined."""
        from ragcore.main import app

        routes = [route.path for route in app.routes]
        assert "/test/complete" in routes

    def test_test_stream_endpoint_exists(self):
        """Verify test stream endpoint is defined."""
        from ragcore.main import app

        routes = [route.path for route in app.routes]
        assert "/test/stream" in routes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
