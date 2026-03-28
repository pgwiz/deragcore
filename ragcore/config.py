"""Configuration management for RAGCORE."""

import logging
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    anthropic_api_key: Optional[str] = None
    azure_api_key: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_deployment_name: Optional[str] = None
    openai_api_key: Optional[str] = None

    # Ollama (optional local model)
    enable_ollama: bool = False
    ollama_base_url: str = "http://localhost:11434"

    # Database
    database_url: str = "postgresql+asyncpg://ragcore:ragcore@localhost:5432/ragcore"

    # Cache
    redis_url: str = "redis://localhost:6379/0"

    # App Settings
    log_level: str = "INFO"
    workers: int = 1
    port: int = 8000
    env: str = "development"

    # ========== PHASE 2: File Processing & Chat ==========
    # File Limits
    max_file_size_mb: int = 50
    max_files_per_session: int = 100

    # Chunking Strategy
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 50

    # Embedding Configuration (Phase 2)
    embedding_provider: str = "azure"  # azure, openai, ollama
    embedding_model: str = "phi-4"  # Azure: phi-4, llama-3.3, qwen-2.5
    embedding_batch_size: int = 10

    # Chat History
    preserve_all_history: bool = True  # No truncation - use all turns
    chat_history_limit: int = 1000  # Max turns to preserve per session

    # Background Job Processing
    job_timeout_seconds: int = 300
    job_max_retries: int = 3
    job_check_interval_ms: int = 500  # Client polling interval

    # ========== PHASE 3: Research Module ==========
    # Web Search API Keys
    tavily_api_key: Optional[str] = None
    serpapi_api_key: Optional[str] = None
    gpt_researcher_enabled: bool = False

    # Research Strategy
    research_default_mode: str = "standard"  # "standard" |"deep"
    research_max_turns: int = 3  # Multi-turn loop limit
    research_max_results_per_search: int = 5
    research_tool_priority: list = ["tavily", "serpapi", "duckduckgo"]

    # WebSocket
    research_stream_batch_size: int = 50  # Tokens per batch
    research_tool_execution_timeout_seconds: int = 30

    # ========== PHASE 4: Production Hardening ==========
    # Authentication
    auth_enabled: bool = True
    jwt_secret_key: Optional[str] = None  # Set in production
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    require_api_key: bool = True  # If false, public access

    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 10  # Per IP
    rate_limit_exempt_paths: list = ["/health", "/docs", "/openapi.json"]

    # API Key Quotas
    quota_enabled: bool = True
    quota_daily_limit: int = 1000  # Requests per day
    quota_monthly_limit: int = 30000  # Requests per month

    # Research Query Costs
    research_query_cost: int = 10  # Cost units per research query
    research_cost_token_multiplier: float = 0.01  # Cost per token

    # Logging & Audit
    audit_logging_enabled: bool = True
    request_logging_enabled: bool = True

    # Monitoring
    prometheus_enabled: bool = True
    metrics_port: int = 9090

    # Webhooks
    webhook_enabled: bool = True
    webhook_max_retries: int = 3
    webhook_retry_backoff_seconds: int = 5
    webhook_timeout_seconds: int = 10

    # ========== PHASE 5: ChromaDB Integration ==========
    # ChromaDB Deployment
    chroma_enabled: bool = True
    chroma_deployment_mode: str = "hybrid"  # hybrid, chroma_primary, postgres_only

    # ChromaDB Connection
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_api_key: Optional[str] = None

    # ChromaDB Persistence
    chroma_persistence_mode: str = "persistent"  # persistent, ephemeral
    chroma_persistence_path: str = "/data/chroma"
    chroma_db_impl: str = "duckdb"  # duckdb, duckdb+parquet

    # ChromaDB Performance Tuning
    chroma_collection_retention_days: int = 365
    chroma_auto_compact_threshold: int = 10000
    chroma_batch_sync_window_ms: int = 100
    chroma_sync_retry_max_attempts: int = 3

    # ChromaDB High Availability
    chroma_connection_pool_size: int = 10
    chroma_connection_timeout_seconds: int = 5
    chroma_circuit_breaker_threshold: int = 5
    chroma_circuit_breaker_reset_minutes: int = 10

    # ChromaDB Memory Management
    chroma_embedding_cache_size: int = 1000
    chroma_max_batch_size: int = 100

    # ========== PHASE 5: Multi-Modal Support ==========
    # Multi-Modal Processing
    multimodal_enabled: bool = True
    multimodal_max_image_size_mb: int = 20
    multimodal_max_audio_size_mb: int = 1000
    multimodal_max_video_size_mb: int = 5000
    multimodal_max_inline_file_size_kb: int = 100

    # Vision Processing
    vision_provider: str = "claude"  # claude, azure, multimodal
    vision_model: str = "claude-3-5-sonnet-20241022"
    vision_enable_azure_fallback: bool = True
    azure_vision_api_key: Optional[str] = None
    azure_vision_endpoint: Optional[str] = None

    # Audio Processing
    audio_provider: str = "azure"  # azure, whisper, multimodal
    audio_transcription_language: str = "en-US"
    audio_enable_whisper_fallback: bool = True
    openai_whisper_api_key: Optional[str] = None

    # Video Processing
    video_max_frame_extraction: int = 30  # Adaptive sampling limit
    video_frame_extraction_strategy: str = "adaptive"  # adaptive, uniform, keyframe
    video_extract_narration: bool = True
    video_extract_frames: bool = True

    # Multi-Modal Storage
    multimodal_storage_backend: str = "local"  # local, s3, azure_blob
    multimodal_storage_path: str = "/data/multimodal"
    multimodal_s3_bucket: Optional[str] = None
    multimodal_s3_region: Optional[str] = None
    multimodal_azure_container: Optional[str] = None
    multimodal_azure_account_name: Optional[str] = None

    # Multi-Modal Embedding
    multimodal_embedding_dimension: int = 1536  # Unified embedding space
    multimodal_embedding_model: str = "text-embedding-3-large"
    multimodal_enable_cross_modal_search: bool = True

    # Performance & Limits
    multimodal_processing_timeout_seconds: int = 300
    multimodal_batch_size: int = 5
    multimodal_cache_extracted_content: bool = True
    multimodal_token_budget_per_content: int = 2000  # Max tokens per item

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


# Global settings instance
settings = Settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)
logger.info(f"RAGCORE initialized in {settings.env} mode")
