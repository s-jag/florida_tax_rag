"""Neo4j schema definitions for the Florida Tax knowledge graph."""

from __future__ import annotations

from enum import Enum


class NodeLabel(str, Enum):
    """Neo4j node labels."""

    DOCUMENT = "Document"
    STATUTE = "Statute"
    RULE = "Rule"
    CASE = "Case"
    TAA = "TAA"
    CHUNK = "Chunk"


class EdgeType(str, Enum):
    """Neo4j relationship types."""

    # Citation relationships (match RelationType from citation_extractor.py)
    CITES = "CITES"
    IMPLEMENTS = "IMPLEMENTS"
    AUTHORITY = "AUTHORITY"
    INTERPRETS = "INTERPRETS"
    AMENDS = "AMENDS"
    SUPERSEDES = "SUPERSEDES"

    # Structural relationships
    HAS_CHUNK = "HAS_CHUNK"  # Document -> Chunk
    CHILD_OF = "CHILD_OF"  # Child Chunk -> Parent Chunk


# Constraint queries for unique IDs
CONSTRAINTS: list[str] = [
    "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
]

# Index queries for common query patterns
INDEXES: list[str] = [
    # Document indexes
    "CREATE INDEX doc_type_idx IF NOT EXISTS FOR (d:Document) ON (d.doc_type)",
    "CREATE INDEX doc_section_idx IF NOT EXISTS FOR (d:Document) ON (d.section)",
    "CREATE INDEX doc_chapter_idx IF NOT EXISTS FOR (d:Document) ON (d.chapter)",
    "CREATE INDEX doc_citation_idx IF NOT EXISTS FOR (d:Document) ON (d.full_citation)",
    # Chunk indexes
    "CREATE INDEX chunk_doc_id_idx IF NOT EXISTS FOR (c:Chunk) ON (c.doc_id)",
    "CREATE INDEX chunk_level_idx IF NOT EXISTS FOR (c:Chunk) ON (c.level)",
    "CREATE INDEX chunk_doc_type_idx IF NOT EXISTS FOR (c:Chunk) ON (c.doc_type)",
    "CREATE INDEX chunk_citation_idx IF NOT EXISTS FOR (c:Chunk) ON (c.citation)",
]


def get_schema_queries() -> list[str]:
    """Return all schema creation queries in order.

    Returns:
        List of Cypher queries for schema creation
    """
    return CONSTRAINTS + INDEXES


def get_drop_constraints_queries() -> list[str]:
    """Return queries to drop all constraints.

    Returns:
        List of Cypher queries to drop constraints
    """
    return [
        "DROP CONSTRAINT doc_id IF EXISTS",
        "DROP CONSTRAINT chunk_id IF EXISTS",
    ]


def get_drop_indexes_queries() -> list[str]:
    """Return queries to drop all indexes.

    Returns:
        List of Cypher queries to drop indexes
    """
    return [
        "DROP INDEX doc_type_idx IF EXISTS",
        "DROP INDEX doc_section_idx IF EXISTS",
        "DROP INDEX doc_chapter_idx IF EXISTS",
        "DROP INDEX chunk_doc_id_idx IF EXISTS",
        "DROP INDEX chunk_level_idx IF EXISTS",
        "DROP INDEX chunk_doc_type_idx IF EXISTS",
    ]
