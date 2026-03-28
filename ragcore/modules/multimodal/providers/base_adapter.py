"""Base provider adapter - abstract interface for multi-modal API providers."""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import logging

from ragcore.core.model_provider_registry import (
    ModelProviderRegistry,
    ProviderType,
    ModelCapability,
)

logger = logging.getLogger(__name__)


class BaseProviderAdapter(ABC):
    """Abstract base for provider adapters.

    Defines interface that all provider-specific adapters must implement.
    Handles provider selection via registry and fallback logic.
    """

    def __init__(
        self,
        registry: Optional[ModelProviderRegistry] = None,
        primary_provider: Optional[ProviderType] = None,
        fallback_provider: Optional[ProviderType] = None,
    ):
        """Initialize provider adapter.

        Args:
            registry: ModelProviderRegistry for provider resolution
            primary_provider: Preferred provider (e.g., ProviderType.ANTHROPIC)
            fallback_provider: Fallback if primary unavailable
        """
        from ragcore.core.model_provider_registry import get_registry

        self.registry = registry or get_registry()
        self.primary_provider = primary_provider
        self.fallback_provider = fallback_provider
        self.last_used_provider: Optional[ProviderType] = None
        self.provider_health: Dict[ProviderType, bool] = {}

    def get_available_provider(self) -> Optional[ProviderType]:
        """Resolve which provider to use based on configuration and availability.

        Returns:
            Provider type to use, or None if none available
        """
        # Try primary first
        if self.primary_provider:
            config = self.registry.get_provider(self.primary_provider)
            if config and self.registry.validate_configuration(self.primary_provider):
                return self.primary_provider

        # Try fallback
        if self.fallback_provider:
            config = self.registry.get_provider(self.fallback_provider)
            if config and self.registry.validate_configuration(self.fallback_provider):
                return self.fallback_provider

        # Scan all providers for one that validates
        for provider_type in self.registry.list_providers():
            if self.registry.validate_configuration(provider_type):
                return provider_type

        logger.warning(f"No valid provider found for {self.__class__.__name__}")
        return None

    def get_provider_config(self, provider: ProviderType):
        """Get configuration for a provider.

        Args:
            provider: Provider type

        Returns:
            ProviderConfig or None
        """
        return self.registry.get_provider(provider)

    def record_provider_health(self, provider: ProviderType, healthy: bool) -> None:
        """Track provider health for fallback decisions.

        Args:
            provider: Provider type
            healthy: Whether provider is responsive
        """
        self.provider_health[provider] = healthy
        if not healthy:
            logger.warning(f"Provider {provider.value} marked unhealthy")

    def is_provider_healthy(self, provider: ProviderType) -> bool:
        """Check if provider is marked healthy.

        Args:
            provider: Provider type

        Returns:
            True if healthy (default True if no record)
        """
        return self.provider_health.get(provider, True)

    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """Execute provider-specific operation.

        Must be implemented by subclasses for specific modality/capability.

        Args:
            *args: Provider-specific positional arguments
            **kwargs: Provider-specific keyword arguments

        Returns:
            Result from provider
        """
        pass

    def get_estimated_cost(
        self, provider: ProviderType, input_size: int
    ) -> Optional[float]:
        """Estimate cost for an operation on given provider.

        Args:
            provider: Provider type
            input_size: Size of input (tokens, bytes, etc)

        Returns:
            Estimated cost or None if not available
        """
        # TODO: Implement cost calculation based on provider pricing
        return None
