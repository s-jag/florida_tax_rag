"""Neo4j knowledge graph for legal citation networks."""

from .client import Neo4jClient, Neo4jConfig
from .loader import (
    init_schema,
    load_all,
    load_chunks,
    load_citations,
    load_documents,
    load_hierarchy,
)
from .queries import (
    ChunkNode,
    CitationEdge,
    DocumentNode,
    InterpretationChainResult,
    find_path_between,
    get_all_citations_for_chunk,
    get_cited_documents,
    get_citing_documents,
    get_document_with_chunks,
    get_interpretation_chain,
    get_rules_with_authority,
    get_statute_with_implementing_rules,
    get_statutes_by_chapter,
)
from .schema import (
    CONSTRAINTS,
    INDEXES,
    EdgeType,
    NodeLabel,
    get_drop_constraints_queries,
    get_drop_indexes_queries,
    get_schema_queries,
)

__all__ = [
    # Client
    "Neo4jClient",
    "Neo4jConfig",
    # Schema
    "NodeLabel",
    "EdgeType",
    "CONSTRAINTS",
    "INDEXES",
    "get_schema_queries",
    "get_drop_constraints_queries",
    "get_drop_indexes_queries",
    # Loader
    "init_schema",
    "load_documents",
    "load_chunks",
    "load_hierarchy",
    "load_citations",
    "load_all",
    # Query models
    "DocumentNode",
    "ChunkNode",
    "CitationEdge",
    "InterpretationChainResult",
    # Query functions
    "get_statute_with_implementing_rules",
    "get_all_citations_for_chunk",
    "get_interpretation_chain",
    "find_path_between",
    "get_document_with_chunks",
    "get_citing_documents",
    "get_cited_documents",
    "get_statutes_by_chapter",
    "get_rules_with_authority",
]
