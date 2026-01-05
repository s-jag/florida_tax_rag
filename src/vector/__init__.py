"""Weaviate vector store for semantic search."""

from .client import SearchResult, WeaviateClient, WeaviateConfig
from .schema import (
    LEGAL_CHUNK_PROPERTIES,
    VOYAGE_LAW_2_DIMENSION,
    CollectionName,
    get_legal_chunk_collection_config,
)

__all__ = [
    # Client
    "WeaviateClient",
    "WeaviateConfig",
    "SearchResult",
    # Schema
    "CollectionName",
    "LEGAL_CHUNK_PROPERTIES",
    "VOYAGE_LAW_2_DIMENSION",
    "get_legal_chunk_collection_config",
]
