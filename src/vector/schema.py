"""Weaviate schema definitions for the Florida Tax vector store."""

from __future__ import annotations

from enum import Enum
from typing import Any

from weaviate.classes.config import (
    Configure,
    DataType,
    Property,
    Tokenization,
    VectorDistances,
)


class CollectionName(str, Enum):
    """Weaviate collection names."""

    LEGAL_CHUNK = "LegalChunk"


# LegalChunk collection schema
LEGAL_CHUNK_PROPERTIES: list[Property] = [
    Property(
        name="chunk_id",
        data_type=DataType.TEXT,
        description="Unique chunk identifier (e.g., chunk:statute:212.05:0)",
        tokenization=Tokenization.FIELD,  # No tokenization - exact match only
    ),
    Property(
        name="doc_id",
        data_type=DataType.TEXT,
        description="Parent document identifier (e.g., statute:212.05)",
        tokenization=Tokenization.FIELD,
    ),
    Property(
        name="doc_type",
        data_type=DataType.TEXT,
        description="Document type (statute, rule, case, taa)",
        tokenization=Tokenization.WORD,  # Enable filtering by type
    ),
    Property(
        name="level",
        data_type=DataType.TEXT,
        description="Chunk hierarchy level (parent or child)",
        tokenization=Tokenization.FIELD,
    ),
    Property(
        name="ancestry",
        data_type=DataType.TEXT,
        description="Hierarchical path (e.g., Florida Statutes > Chapter 212 > § 212.05)",
        tokenization=Tokenization.WORD,  # BM25 searchable
    ),
    Property(
        name="citation",
        data_type=DataType.TEXT,
        description="Legal citation (e.g., Fla. Stat. § 212.05)",
        tokenization=Tokenization.WORD,
    ),
    Property(
        name="text",
        data_type=DataType.TEXT,
        description="Raw chunk text for BM25 keyword search",
        tokenization=Tokenization.WORD,  # Full BM25 tokenization
    ),
    Property(
        name="text_with_ancestry",
        data_type=DataType.TEXT,
        description="Chunk text with ancestry prefix (used for embedding)",
        tokenization=Tokenization.WORD,
    ),
    Property(
        name="effective_date",
        data_type=DataType.DATE,
        description="Effective date of the legal document",
    ),
    Property(
        name="token_count",
        data_type=DataType.INT,
        description="Number of tokens in the chunk",
    ),
]


def get_legal_chunk_collection_config() -> dict[str, Any]:
    """Get the configuration for creating the LegalChunk collection.

    Returns:
        Dictionary of collection configuration arguments for client.collections.create()
    """
    return {
        "name": CollectionName.LEGAL_CHUNK.value,
        "description": "Florida tax law document chunks for hybrid RAG retrieval",
        "properties": LEGAL_CHUNK_PROPERTIES,
        # Use self_provided for external vectors from Voyage AI
        "vector_config": Configure.Vectors.self_provided(
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE,
            ),
        ),
        "inverted_index_config": Configure.inverted_index(
            bm25_b=0.75,  # Document length normalization
            bm25_k1=1.2,  # Term frequency saturation
        ),
    }


# Vector dimension for Voyage AI voyage-law-2 model
VOYAGE_LAW_2_DIMENSION = 1024
