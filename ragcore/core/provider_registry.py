"""Provider registry - lazy loading and caching of providers."""

import logging
from typing import Dict, Optional
from ragcore.core.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """
    Singleton registry for AI provider instances.

    Providers are lazy-loaded on first request and cached.
    This ensures we don't initialize providers we don't need.
    """

    def __init__(self):
        """Initialize empty provider cache."""
        self._providers: Dict[str, Optional[BaseProvider]] = {}
        self._initialized: Dict[str, bool] = {}

    def get_provider(self, name: str) -> Optional[BaseProvider]:
        """
        Get or lazy-load a provider by name.

        Args:
            name: Provider name ('anthropic', 'azure', 'openai', 'ollama')

        Returns:
            Provider instance or None if not available
        """
        name = name.lower().strip()

        # Return cached instance if available
        if name in self._providers and self._initialized.get(name):
            return self._providers[name]

        # Try to instantiate provider
        try:
            provider = self._load_provider(name)
            self._providers[name] = provider
            self._initialized[name] = True
            logger.info(f"Provider '{name}' initialized successfully")
            return provider
        except Exception as e:
            logger.warning(f"Failed to initialize provider '{name}': {e}")
            self._providers[name] = None
            self._initialized[name] = False
            return None

    def _load_provider(self, name: str) -> BaseProvider:
        """
        Factory method to instantiate a provider.

        Args:
            name: Provider name

        Returns:
            Provider instance

        Raises:
            ValueError: If provider name unknown
        """
        if name == "anthropic":
            from ragcore.core.providers.anthropic_provider import AnthropicProvider

            return AnthropicProvider()
        elif name == "azure":
            from ragcore.core.providers.azure_provider import AzureProvider

            return AzureProvider()
        elif name == "openai":
            from ragcore.core.providers.openai_provider import OpenAIProvider

            return OpenAIProvider()
        elif name == "ollama":
            from ragcore.core.providers.ollama_provider import OllamaProvider

            return OllamaProvider()
        else:
            raise ValueError(f"Unknown provider: {name}")

    def list_available_providers(self) -> Dict[str, bool]:
        """
        Scan for available providers without initializing them.

        Returns:
            Dict mapping provider names to availability status
        """
        providers = {
            "anthropic": self.get_provider("anthropic") is not None,
            "azure": self.get_provider("azure") is not None,
            "openai": self.get_provider("openai") is not None,
            "ollama": self.get_provider("ollama") is not None,
        }
        return {k: v for k, v in providers.items() if v}

    def reset(self):
        """Clear all cached providers (for testing)."""
        self._providers.clear()
        self._initialized.clear()
        logger.info("Provider registry reset")


# Global registry instance
registry = ProviderRegistry()
