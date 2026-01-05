"""Application settings using pydantic-settings."""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Voyage AI (Legal Embeddings)
    voyage_api_key: SecretStr

    # Anthropic (Claude LLM)
    anthropic_api_key: SecretStr

    # Neo4j (Knowledge Graph)
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr

    # Weaviate (Vector Store)
    weaviate_url: str = "http://localhost:8080"
    weaviate_api_key: SecretStr | None = None


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
