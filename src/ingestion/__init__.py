"""Ingestion module for data consolidation and processing."""

from .build_citation_graph import (
    CitationIndex,
    ResolvedEdge,
    build_citation_graph,
    build_citation_index,
    deduplicate_edges,
    resolve_citation,
)
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
from .citation_extractor import (
    Citation,
    CitationRelation,
    CitationType,
    RelationType,
    detect_relation_type,
    extract_all_citations,
    extract_case_citations,
    extract_chapter_citations,
    extract_rule_citations,
    extract_statute_citations,
    get_context,
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
    # Citation extraction
    "Citation",
    "CitationRelation",
    "CitationType",
    "RelationType",
    "detect_relation_type",
    "extract_all_citations",
    "extract_case_citations",
    "extract_chapter_citations",
    "extract_rule_citations",
    "extract_statute_citations",
    "get_context",
    # Citation graph building
    "CitationIndex",
    "ResolvedEdge",
    "build_citation_graph",
    "build_citation_index",
    "deduplicate_edges",
    "resolve_citation",
]
