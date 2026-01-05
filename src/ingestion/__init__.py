"""Ingestion module for data consolidation and processing."""

from .chunking import (
    ChunkLevel,
    LegalChunk,
    chunk_case,
    chunk_corpus,
    chunk_document,
    chunk_rule,
    chunk_statute,
    chunk_taa,
)
from .consolidate import (
    consolidate_all,
    consolidate_cases,
    consolidate_rules,
    consolidate_statutes,
    consolidate_taas,
)
from .models import Corpus, CorpusMetadata, DocumentType, LegalDocument
from .tokenizer import count_tokens, get_encoder, truncate_to_tokens

__all__ = [
    # Models
    "Corpus",
    "CorpusMetadata",
    "DocumentType",
    "LegalDocument",
    "ChunkLevel",
    "LegalChunk",
    # Consolidation
    "consolidate_all",
    "consolidate_cases",
    "consolidate_rules",
    "consolidate_statutes",
    "consolidate_taas",
    # Chunking
    "chunk_case",
    "chunk_corpus",
    "chunk_document",
    "chunk_rule",
    "chunk_statute",
    "chunk_taa",
    # Tokenizer
    "count_tokens",
    "get_encoder",
    "truncate_to_tokens",
]
