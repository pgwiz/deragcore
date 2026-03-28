"""Model provider registry - support for multiple LLM/embedding providers.

Supports: Claude (Anthropic), GPT-4/3.5 (OpenAI), Gemini/Codey (Vertex AI),
Azure Foundry (serverless), Azure OpenAI, Ollama (local).

Each provider has different endpoint structures, auth methods, and model names.
Registry abstracts these differences for unified access.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ProviderType(str, Enum):
    """Supported model providers."""
    ANTHROPIC = "anthropic"  # Claude
    OPENAI = "openai"  # GPT-4, GPT-3.5
    AZURE_FOUNDRY = "azure_foundry"  # Azure Foundry serverless endpoints
    AZURE_OPENAI = "azure_openai"  # Azure OpenAI dedicated instances
    VERTEX_AI = "vertex_ai"  # Google Vertex AI
    OLLAMA = "ollama"  # Local Ollama


class ModelCapability(str, Enum):
    """Capabilities a model supports."""
    CHAT = "chat"  # Chat completions
    VISION = "vision"  # Image understanding
    EMBEDDINGS = "embeddings"  # Text embeddings
    AUDIO = "audio"  # Audio/speech


@dataclass
class ProviderConfig:
    """Configuration for a model provider."""

    provider: ProviderType
    endpoint: Optional[str] = None  # URL base (required for cloud providers)
    api_key: Optional[str] = None  # Authentication key
    azure_key: Optional[str] = None  # For Azure-specific auth
    region: Optional[str] = None  # For regional services (Azure, Vertex)
    project_id: Optional[str] = None  # For Vertex AI
    api_version: Optional[str] = None  # For versioned APIs
    deployment_name: Optional[str] = None  # For Azure Foundry/OpenAI deployments
    deployment_id: Optional[str] = None  # Alternate to deployment_name
    local_port: Optional[int] = 11434  # For Ollama local (default 11434)
    model_name: Optional[str] = None  # Provider-specific model identifier

    def get_endpoint_url(self, task: str = "chat") -> str:
        """Get full endpoint URL for this provider.

        Args:
            task: Task type (chat, vision, embeddings, audio)

        Returns:
            Full endpoint URL
        """
        if self.provider == ProviderType.AZURE_FOUNDRY:
            # https://<project>.<region>.models.ai.azure.com/serverless/v1/chat/completions
            if not self.endpoint or not self.region:
                raise ValueError("Azure Foundry requires endpoint and region")
            return f"{self.endpoint}/serverless/v1/{task}/completions"

        elif self.provider == ProviderType.AZURE_OPENAI:
            # https://<resource>.openai.azure.com/openai/deployments/<deployment>/chat/completions
            if not self.endpoint or not self.deployment_name:
                raise ValueError("Azure OpenAI requires endpoint and deployment_name")
            return f"{self.endpoint}/openai/deployments/{self.deployment_name}/{task}/completions"

        elif self.provider == ProviderType.OPENAI:
            # https://api.openai.com/v1/chat/completions
            return f"https://api.openai.com/v1/{task}/completions"

        elif self.provider == ProviderType.ANTHROPIC:
            # https://api.anthropic.com/v1/messages (fixed)
            return "https://api.anthropic.com/v1/messages"

        elif self.provider == ProviderType.VERTEX_AI:
            # https://<region>-aiplatform.googleapis.com/v1/projects/<project>/locations/<region>/publishers/google/models/<model>:predict
            if not self.region or not self.project_id or not self.model_name:
                raise ValueError("Vertex AI requires region, project_id, and model_name")
            return f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/publishers/google/models/{self.model_name}:predict"

        elif self.provider == ProviderType.OLLAMA:
            # http://localhost:11434/api/<task>
            port = self.local_port or 11434
            return f"http://localhost:{port}/api/{task}"

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def get_auth_headers(self) -> Dict[str, str]:
        """Get HTTP headers for authentication.

        Returns:
            Dict of auth headers
        """
        headers = {}

        if self.provider == ProviderType.AZURE_FOUNDRY:
            if self.api_key:
                headers["api-key"] = self.api_key

        elif self.provider == ProviderType.AZURE_OPENAI:
            if self.api_key:
                headers["api-key"] = self.api_key

        elif self.provider == ProviderType.OPENAI:
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

        elif self.provider == ProviderType.ANTHROPIC:
            if self.api_key:
                headers["x-api-key"] = self.api_key
                headers["anthropic-version"] = "2023-06-01"

        elif self.provider == ProviderType.VERTEX_AI:
            # Vertex uses OAuth2, would need token generation
            pass

        # Ollama has no auth

        headers["Content-Type"] = "application/json"
        return headers


@dataclass
class ProviderModel:
    """Model information for a provider."""

    provider: ProviderType
    provider_model_id: str  # Provider-specific identifier (e.g., "gpt-4-turbo", "claude-opus")
    display_name: str  # User-friendly name
    context_window: int  # Max tokens
    supports_vision: bool = False
    supports_embeddings: bool = False
    supports_audio: bool = False
    capabilities: List[ModelCapability] = None

    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []


class ModelProviderRegistry:
    """Registry mapping providers → configurations → models.

    Enables switching providers without changing application code.
    Supports: Claude, GPT-4/3.5, Vertex, Azure Foundry, Azure OpenAI, Ollama.
    """

    def __init__(self):
        """Initialize registry with default models."""
        self.providers: Dict[ProviderType, ProviderConfig] = {}
        self.models: Dict[str, ProviderModel] = {}

        # Register default models
        self._register_default_models()

    def _register_default_models(self):
        """Register well-known models from all providers."""
        # Anthropic Claude
        self.register_model(
            ProviderModel(
                provider=ProviderType.ANTHROPIC,
                provider_model_id="claude-opus-4-6",
                display_name="Claude Opus 4.6",
                context_window=200000,
                supports_vision=True,
                supports_embeddings=False,
                supports_audio=False,
                capabilities=[ModelCapability.CHAT, ModelCapability.VISION],
            )
        )

        self.register_model(
            ProviderModel(
                provider=ProviderType.ANTHROPIC,
                provider_model_id="claude-sonnet-4-6",
                display_name="Claude Sonnet 4.6",
                context_window=200000,
                supports_vision=True,
                capabilities=[ModelCapability.CHAT, ModelCapability.VISION],
            )
        )

        # OpenAI
        self.register_model(
            ProviderModel(
                provider=ProviderType.OPENAI,
                provider_model_id="gpt-4-turbo",
                display_name="GPT-4 Turbo",
                context_window=128000,
                supports_vision=True,
                capabilities=[ModelCapability.CHAT, ModelCapability.VISION],
            )
        )

        self.register_model(
            ProviderModel(
                provider=ProviderType.OPENAI,
                provider_model_id="gpt-3.5-turbo",
                display_name="GPT-3.5 Turbo",
                context_window=16384,
                supports_vision=False,
                capabilities=[ModelCapability.CHAT],
            )
        )

        self.register_model(
            ProviderModel(
                provider=ProviderType.OPENAI,
                provider_model_id="text-embedding-3-large",
                display_name="Text Embedding 3 Large",
                context_window=8191,
                supports_embeddings=True,
                capabilities=[ModelCapability.EMBEDDINGS],
            )
        )

        # Google Vertex AI
        self.register_model(
            ProviderModel(
                provider=ProviderType.VERTEX_AI,
                provider_model_id="gemini-2.0-flash",
                display_name="Gemini 2.0 Flash",
                context_window=1000000,
                supports_vision=True,
                capabilities=[ModelCapability.CHAT, ModelCapability.VISION],
            )
        )

        # Azure Foundry (generic - actual models deployed per project)
        self.register_model(
            ProviderModel(
                provider=ProviderType.AZURE_FOUNDRY,
                provider_model_id="custom-vision-model",
                display_name="Custom Vision Model (Foundry)",
                context_window=8192,
                supports_vision=True,
                capabilities=[ModelCapability.VISION],
            )
        )

    def register_provider(self, config: ProviderConfig) -> None:
        """Register a provider configuration.

        Args:
            config: Provider configuration with auth details
        """
        self.providers[config.provider] = config
        logger.info(f"Registered provider: {config.provider.value}")

    def register_model(self, model: ProviderModel) -> None:
        """Register a model.

        Args:
            model: Model information
        """
        key = self._model_key(model.provider, model.provider_model_id)
        self.models[key] = model
        logger.debug(f"Registered model: {model.display_name} ({key})")

    def get_provider(self, provider_type: ProviderType) -> Optional[ProviderConfig]:
        """Get provider configuration.

        Args:
            provider_type: Type of provider

        Returns:
            Provider config or None if not registered
        """
        return self.providers.get(provider_type)

    def get_model(self, provider: ProviderType, model_id: str) -> Optional[ProviderModel]:
        """Get model information.

        Args:
            provider: Provider type
            model_id: Model identifier

        Returns:
            Model info or None if not found
        """
        key = self._model_key(provider, model_id)
        return self.models.get(key)

    def list_providers(self) -> List[ProviderType]:
        """List registered providers.

        Returns:
            List of provider types
        """
        return list(self.providers.keys())

    def list_models(self, provider: Optional[ProviderType] = None) -> List[ProviderModel]:
        """List available models, optionally filtered by provider.

        Args:
            provider: Optional provider type to filter

        Returns:
            List of models
        """
        models = list(self.models.values())
        if provider:
            models = [m for m in models if m.provider == provider]
        return models

    def list_models_by_capability(self, capability: ModelCapability) -> List[ProviderModel]:
        """List models supporting a specific capability.

        Args:
            capability: Required capability (vision, embeddings, audio, etc)

        Returns:
            List of models with that capability
        """
        return [m for m in self.models.values() if capability in m.capabilities]

    def _model_key(self, provider: ProviderType, model_id: str) -> str:
        """Generate a unique key for a model.

        Args:
            provider: Provider type
            model_id: Model identifier

        Returns:
            Unique key
        """
        return f"{provider.value}:{model_id}"

    def validate_configuration(self, provider: ProviderType) -> bool:
        """Validate that a provider is properly configured.

        Args:
            provider: Provider type to validate

        Returns:
            True if configured and ready to use
        """
        config = self.get_provider(provider)
        if not config:
            logger.warning(f"Provider not registered: {provider.value}")
            return False

        # Provider-specific validation
        if provider == ProviderType.AZURE_FOUNDRY:
            if not config.endpoint or not config.api_key:
                logger.warning("Azure Foundry requires endpoint and api_key")
                return False

        elif provider == ProviderType.AZURE_OPENAI:
            if not config.endpoint or not config.api_key or not config.deployment_name:
                logger.warning("Azure OpenAI requires endpoint, api_key, and deployment_name")
                return False

        elif provider == ProviderType.OPENAI:
            if not config.api_key:
                logger.warning("OpenAI requires api_key")
                return False

        elif provider == ProviderType.ANTHROPIC:
            if not config.api_key:
                logger.warning("Anthropic requires api_key")
                return False

        elif provider == ProviderType.VERTEX_AI:
            if not config.project_id or not config.region:
                logger.warning("Vertex AI requires project_id and region")
                return False

        # Ollama is always available locally

        return True


# Global registry instance
_registry = ModelProviderRegistry()


def get_registry() -> ModelProviderRegistry:
    """Get the global model provider registry.

    Returns:
        ModelProviderRegistry instance
    """
    return _registry
