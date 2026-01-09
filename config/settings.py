"""Application settings using pydantic-settings."""

from __future__ import annotations

from functools import lru_cache

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Settings are organized into logical groups:
    - Environment: Runtime environment configuration
    - API Keys: External service authentication
    - Database: Neo4j, Weaviate, Redis connection settings
    - Retrieval: Search and retrieval parameters
    - Generation: LLM generation parameters
    - Embedding: Vector embedding configuration
    - Rate Limits: API rate limiting
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ==========================================================================
    # Environment
    # ==========================================================================
    env: str = "development"
    """Runtime environment: development, staging, or production."""

    debug: bool = True
    """Enable debug mode with verbose logging."""

    log_level: str = "INFO"
    """Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL."""

    # ==========================================================================
    # API Keys (Required)
    # ==========================================================================
    voyage_api_key: SecretStr
    """Voyage AI API key for legal embeddings (voyage-law-2 model)."""

    anthropic_api_key: SecretStr
    """Anthropic API key for Claude LLM generation."""

    neo4j_password: SecretStr
    """Neo4j database password."""

    # ==========================================================================
    # API Keys (Optional)
    # ==========================================================================
    openai_api_key: SecretStr | None = None
    """OpenAI API key for GPT-4 evaluation judge (optional)."""

    weaviate_api_key: SecretStr | None = None
    """Weaviate API key (optional, for cloud deployments)."""

    # ==========================================================================
    # Neo4j (Knowledge Graph)
    # ==========================================================================
    neo4j_uri: str = "bolt://localhost:7687"
    """Neo4j connection URI."""

    neo4j_user: str = "neo4j"
    """Neo4j username."""

    neo4j_connection_timeout: int = 30
    """Neo4j connection timeout in seconds."""

    neo4j_max_connection_pool_size: int = 50
    """Maximum Neo4j connection pool size."""

    # ==========================================================================
    # Weaviate (Vector Store)
    # ==========================================================================
    weaviate_url: str = "http://localhost:8080"
    """Weaviate server URL."""

    weaviate_timeout: int = 30
    """Weaviate request timeout in seconds."""

    # ==========================================================================
    # Redis (Embedding Cache)
    # ==========================================================================
    redis_url: str = "redis://localhost:6379/0"
    """Redis connection URL (preferred over individual host/port/db)."""

    redis_host: str = "localhost"
    """Redis host (legacy, use redis_url instead)."""

    redis_port: int = 6379
    """Redis port (legacy, use redis_url instead)."""

    redis_db: int = 0
    """Redis database number (legacy, use redis_url instead)."""

    # ==========================================================================
    # Retrieval Settings
    # ==========================================================================
    retrieval_top_k: int = 20
    """Number of documents to retrieve."""

    hybrid_alpha: float = 0.25
    """Hybrid search alpha: 0.0=keyword, 1.0=vector. Optimal: 0.25 (keyword-heavy)."""

    expand_graph: bool = True
    """Enable graph expansion for related documents."""

    max_graph_hops: int = 2
    """Maximum graph traversal depth for expansion."""

    # ==========================================================================
    # Generation Settings
    # ==========================================================================
    llm_model: str = "claude-sonnet-4-20250514"
    """Claude model for generation."""

    llm_temperature: float = 0.1
    """LLM temperature (lower = more deterministic)."""

    max_tokens: int = 4096
    """Maximum tokens for LLM response."""

    llm_timeout: int = 60
    """LLM request timeout in seconds."""

    # ==========================================================================
    # Embedding Settings
    # ==========================================================================
    embedding_model: str = "voyage-law-2"
    """Voyage embedding model for legal documents."""

    embedding_batch_size: int = 128
    """Batch size for embedding requests."""

    embedding_cache_ttl: int = 86400
    """Embedding cache TTL in seconds (default: 24 hours)."""

    # ==========================================================================
    # Rate Limits
    # ==========================================================================
    rate_limit_per_minute: int = 60
    """API rate limit per minute per client."""

    voyage_requests_per_minute: int = 300
    """Voyage AI rate limit."""

    anthropic_requests_per_minute: int = 60
    """Anthropic API rate limit."""

    # ==========================================================================
    # Validators
    # ==========================================================================
    @field_validator("env")
    @classmethod
    def validate_env(cls, v: str) -> str:
        """Validate environment is one of allowed values."""
        allowed = {"development", "staging", "production"}
        if v.lower() not in allowed:
            raise ValueError(f"env must be one of {allowed}, got '{v}'")
        return v.lower()

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is valid."""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got '{v}'")
        return v.upper()

    @field_validator("hybrid_alpha")
    @classmethod
    def validate_alpha(cls, v: float) -> float:
        """Validate alpha is between 0 and 1."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"hybrid_alpha must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("llm_temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is between 0 and 2."""
        if not 0.0 <= v <= 2.0:
            raise ValueError(f"llm_temperature must be between 0.0 and 2.0, got {v}")
        return v

    # ==========================================================================
    # Computed Properties
    # ==========================================================================
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.env == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.env == "development"

    @property
    def is_staging(self) -> bool:
        """Check if running in staging environment."""
        return self.env == "staging"

    @property
    def redis_connection_url(self) -> str:
        """Build Redis URL from components (for backward compatibility)."""
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Settings are loaded once and cached for the lifetime of the application.
    To reload settings, call `get_settings.cache_clear()` first.
    """
    return Settings()
