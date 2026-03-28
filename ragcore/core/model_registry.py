"""Model context window registry - Map providers and models to context window sizes."""

import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

# Mapping: (provider, model_id) -> context_window_tokens
MODEL_CONTEXT_WINDOWS: Dict[Tuple[str, str], int] = {
    # Anthropic Claude models
    ("anthropic", "claude-3-5-sonnet-20241022"): 200000,
    ("anthropic", "claude-3-5-sonnet"): 200000,
    ("anthropic", "claude-3-opus-20250219"): 200000,
    ("anthropic", "claude-3-opus"): 200000,
    ("anthropic", "claude-3-sonnet-20240229"): 200000,
    ("anthropic", "claude-3-sonnet"): 200000,
    ("anthropic", "claude-3-haiku-20240307"): 200000,
    ("anthropic", "claude-3-haiku"): 200000,

    # OpenAI GPT-4 models
    ("openai", "gpt-4-turbo"): 128000,
    ("openai", "gpt-4-turbo-preview"): 128000,
    ("openai", "gpt-4-0125-preview"): 128000,
    ("openai", "gpt-4-1106-preview"): 128000,
    ("openai", "gpt-4"): 8000,
    ("openai", "gpt-4-32k"): 32000,
    ("openai", "gpt-4o"): 128000,
    ("openai", "gpt-4o-2024-11-20"): 128000,
    ("openai", "gpt-3.5-turbo"): 16000,

    # Azure (same as OpenAI models)
    ("azure", "gpt-4-turbo"): 128000,
    ("azure", "gpt-4-turbo-preview"): 128000,
    ("azure", "gpt-4"): 8000,
    ("azure", "gpt-4-32k"): 32000,
    ("azure", "gpt-4o"): 128000,
    ("azure", "gpt-3.5-turbo"): 16000,

    # Meta Llama models
    ("ollama", "llama2"): 4096,
    ("ollama", "llama2:13b"): 4096,
    ("ollama", "llama2:70b"): 4096,
    ("ollama", "llama3"): 8000,
    ("ollama", "llama3:8b"): 8000,
    ("ollama", "llama3:70b"): 8000,
    ("ollama", "llama3.1"): 128000,
    ("ollama", "llama3.1:8b"): 128000,
    ("ollama", "llama3.1:70b"): 128000,

    # Mistral models (via Ollama)
    ("ollama", "mistral"): 32000,
    ("ollama", "mistral:7b"): 32000,
    ("ollama", "mistral:large"): 32000,

    # Google/Other models
    ("openai", "phi-4"): 16000,  # Common embedding model
}


class ModelRegistry:
    """Registry for model-specific configuration like context windows."""

    @staticmethod
    def get_context_window(
        provider: str,
        model_id: str,
        default: int = 8000,
    ) -> int:
        """
        Get context window size for a model.

        Args:
            provider: Provider name ('anthropic', 'openai', 'azure', 'ollama')
            model_id: Model ID/name
            default: Fallback window size if not found (default: 8000)

        Returns:
            Context window size in tokens
        """
        key = (provider.lower(), model_id.lower())

        # Exact match
        if key in MODEL_CONTEXT_WINDOWS:
            window = MODEL_CONTEXT_WINDOWS[key]
            logger.debug(f"Context window for {key}: {window} tokens")
            return window

        # Fallback: try matching provider + partial model name
        for (prov, mid), window in MODEL_CONTEXT_WINDOWS.items():
            if prov == provider.lower() and model_id.lower() in mid:
                logger.debug(
                    f"Context window for {provider}/{model_id} (partial match {mid}): {window} tokens"
                )
                return window

        # Not found, use default
        logger.warning(
            f"Unknown model {provider}/{model_id}, using default context window: {default} tokens"
        )
        return default

    @staticmethod
    def list_models() -> Dict[str, list]:
        """List all known models by provider.

        Returns:
            Dict mapping provider name to list of (model_id, context_window) tuples
        """
        models_by_provider = {}
        for (provider, model), window in sorted(MODEL_CONTEXT_WINDOWS.items()):
            if provider not in models_by_provider:
                models_by_provider[provider] = []
            models_by_provider[provider].append((model, window))
        return models_by_provider
