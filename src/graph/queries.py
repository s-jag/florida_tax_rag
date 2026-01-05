"""Common graph queries for the Florida Tax RAG system."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from .client import Neo4jClient


class DocumentNode(BaseModel):
    """A document node from the graph."""

    id: str = Field(..., description="Document ID")
    doc_type: str = Field(..., description="Document type")
    title: str = Field(..., description="Document title")
    full_citation: str = Field(..., description="Full legal citation")
    section: Optional[str] = Field(default=None, description="Section number")
    chapter: Optional[str] = Field(default=None, description="Chapter number")


class ChunkNode(BaseModel):
    """A chunk node from the graph."""

    id: str = Field(..., description="Chunk ID")
    doc_id: str = Field(..., description="Parent document ID")
    level: str = Field(..., description="Chunk level (parent/child)")
    ancestry: str = Field(..., description="Hierarchy path")
    subsection_path: str = Field(default="", description="Subsection path")
    citation: str = Field(..., description="Legal citation")
    token_count: int = Field(..., description="Token count")


class CitationEdge(BaseModel):
    """A citation relationship."""

    source_id: str = Field(..., description="Source document ID")
    target_id: str = Field(..., description="Target document ID")
    relation_type: str = Field(..., description="Relationship type")
    citation_text: str = Field(..., description="Citation text")
    confidence: float = Field(..., description="Confidence score")


class InterpretationChainResult(BaseModel):
    """Result of an interpretation chain query."""

    statute: DocumentNode = Field(..., description="The statute")
    implementing_rules: list[DocumentNode] = Field(
        default_factory=list, description="Rules implementing the statute"
    )
    interpreting_cases: list[DocumentNode] = Field(
        default_factory=list, description="Cases interpreting the statute"
    )
    interpreting_taas: list[DocumentNode] = Field(
        default_factory=list, description="TAAs interpreting the statute"
    )


def get_statute_with_implementing_rules(
    client: Neo4jClient,
    section: str,
) -> Optional[InterpretationChainResult]:
    """Get a statute with all rules that implement it.

    Args:
        client: Neo4j client
        section: Statute section number (e.g., '212.05')

    Returns:
        InterpretationChainResult or None if statute not found
    """
    query = """
    MATCH (s:Statute {section: $section})
    OPTIONAL MATCH (r:Rule)-[:IMPLEMENTS]->(s)
    RETURN s, collect(DISTINCT r) AS rules
    """

    results = client.run_query(query, {"section": section})

    if not results or results[0]["s"] is None:
        return None

    row = results[0]
    statute_data = dict(row["s"])
    rules_data = [dict(r) for r in row["rules"] if r is not None]

    return InterpretationChainResult(
        statute=DocumentNode(**statute_data),
        implementing_rules=[DocumentNode(**r) for r in rules_data],
    )


def get_all_citations_for_chunk(
    client: Neo4jClient,
    chunk_id: str,
) -> list[CitationEdge]:
    """Get all citations originating from a specific chunk.

    Args:
        client: Neo4j client
        chunk_id: The chunk ID

    Returns:
        List of citation edges
    """
    query = """
    MATCH (c:Chunk {id: $chunk_id})<-[:HAS_CHUNK]-(source:Document)
    MATCH (source)-[r]->(target:Document)
    WHERE type(r) IN ['CITES', 'IMPLEMENTS', 'AUTHORITY', 'INTERPRETS', 'AMENDS', 'SUPERSEDES']
      AND r.source_chunk_id = $chunk_id
    RETURN source.id AS source_id, target.id AS target_id,
           type(r) AS relation_type, r.citation_text AS citation_text,
           r.confidence AS confidence
    """

    results = client.run_query(query, {"chunk_id": chunk_id})

    return [
        CitationEdge(
            source_id=r["source_id"],
            target_id=r["target_id"],
            relation_type=r["relation_type"],
            citation_text=r["citation_text"],
            confidence=r["confidence"],
        )
        for r in results
    ]


def get_interpretation_chain(
    client: Neo4jClient,
    statute_section: str,
) -> Optional[InterpretationChainResult]:
    """Get full interpretation chain: statute -> rules -> cases/TAAs.

    This traverses the legal authority hierarchy to find all documents
    that implement or interpret a given statute.

    Args:
        client: Neo4j client
        statute_section: Statute section number (e.g., '212.05')

    Returns:
        InterpretationChainResult with full chain
    """
    query = """
    MATCH (s:Statute {section: $section})

    // Get implementing rules
    OPTIONAL MATCH (r:Rule)-[:IMPLEMENTS|AUTHORITY]->(s)

    // Get interpreting cases (directly or via rules)
    OPTIONAL MATCH (c:Case)-[:INTERPRETS|CITES]->(s)
    OPTIONAL MATCH (c2:Case)-[:INTERPRETS|CITES]->(r)

    // Get TAAs
    OPTIONAL MATCH (t:TAA)-[:INTERPRETS|CITES]->(s)
    OPTIONAL MATCH (t2:TAA)-[:INTERPRETS|CITES]->(r)

    WITH s,
         collect(DISTINCT r) AS rules,
         collect(DISTINCT c) + collect(DISTINCT c2) AS cases,
         collect(DISTINCT t) + collect(DISTINCT t2) AS taas

    RETURN s,
           rules,
           [x IN cases WHERE x IS NOT NULL] AS cases,
           [x IN taas WHERE x IS NOT NULL] AS taas
    """

    results = client.run_query(query, {"section": statute_section})

    if not results or results[0]["s"] is None:
        return None

    row = results[0]

    return InterpretationChainResult(
        statute=DocumentNode(**dict(row["s"])),
        implementing_rules=[DocumentNode(**dict(r)) for r in row["rules"] if r],
        interpreting_cases=[DocumentNode(**dict(c)) for c in row["cases"] if c],
        interpreting_taas=[DocumentNode(**dict(t)) for t in row["taas"] if t],
    )


def find_path_between(
    client: Neo4jClient,
    source_id: str,
    target_id: str,
    max_hops: int = 4,
) -> list[dict[str, Any]]:
    """Find the shortest path between two documents.

    Args:
        client: Neo4j client
        source_id: Source document ID
        target_id: Target document ID
        max_hops: Maximum path length

    Returns:
        List of path segments with nodes and relationships
    """
    query = f"""
    MATCH path = shortestPath(
        (source:Document {{id: $source_id}})-[*1..{max_hops}]-(target:Document {{id: $target_id}})
    )
    UNWIND relationships(path) AS rel
    WITH path, rel, startNode(rel) AS from_node, endNode(rel) AS to_node
    RETURN from_node.id AS from_id, from_node.title AS from_title,
           type(rel) AS relation,
           to_node.id AS to_id, to_node.title AS to_title
    """

    return client.run_query(
        query, {"source_id": source_id, "target_id": target_id}
    )


def get_document_with_chunks(
    client: Neo4jClient,
    doc_id: str,
) -> Optional[dict[str, Any]]:
    """Get a document with all its chunks.

    Args:
        client: Neo4j client
        doc_id: Document ID

    Returns:
        Dict with document and chunks data
    """
    query = """
    MATCH (d:Document {id: $doc_id})
    OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
    OPTIONAL MATCH (c)-[:CHILD_OF]->(parent:Chunk)
    RETURN d, collect({chunk: c, parent_id: parent.id}) AS chunks
    """

    results = client.run_query(query, {"doc_id": doc_id})

    if not results or results[0]["d"] is None:
        return None

    row = results[0]
    doc_data = dict(row["d"])
    chunks_data = [
        {**dict(c["chunk"]), "parent_chunk_id": c["parent_id"]}
        for c in row["chunks"]
        if c["chunk"] is not None
    ]

    return {
        "document": doc_data,
        "chunks": sorted(chunks_data, key=lambda x: x["id"]),
    }


def get_citing_documents(
    client: Neo4jClient,
    doc_id: str,
) -> list[DocumentNode]:
    """Get all documents that cite a given document.

    Args:
        client: Neo4j client
        doc_id: Document ID

    Returns:
        List of citing documents
    """
    query = """
    MATCH (source:Document)-[r]->(target:Document {id: $doc_id})
    WHERE type(r) IN ['CITES', 'IMPLEMENTS', 'AUTHORITY', 'INTERPRETS']
    RETURN DISTINCT source
    """

    results = client.run_query(query, {"doc_id": doc_id})

    return [DocumentNode(**dict(r["source"])) for r in results]


def get_cited_documents(
    client: Neo4jClient,
    doc_id: str,
) -> list[DocumentNode]:
    """Get all documents cited by a given document.

    Args:
        client: Neo4j client
        doc_id: Document ID

    Returns:
        List of cited documents
    """
    query = """
    MATCH (source:Document {id: $doc_id})-[r]->(target:Document)
    WHERE type(r) IN ['CITES', 'IMPLEMENTS', 'AUTHORITY', 'INTERPRETS', 'AMENDS', 'SUPERSEDES']
    RETURN DISTINCT target
    """

    results = client.run_query(query, {"doc_id": doc_id})

    return [DocumentNode(**dict(r["target"])) for r in results]


def get_statutes_by_chapter(
    client: Neo4jClient,
    chapter: str,
) -> list[DocumentNode]:
    """Get all statutes in a chapter.

    Args:
        client: Neo4j client
        chapter: Chapter number (e.g., '212')

    Returns:
        List of statutes in the chapter
    """
    query = """
    MATCH (s:Statute {chapter: $chapter})
    RETURN s
    ORDER BY s.section
    """

    results = client.run_query(query, {"chapter": chapter})

    return [DocumentNode(**dict(r["s"])) for r in results]


def get_rules_with_authority(
    client: Neo4jClient,
) -> list[dict[str, Any]]:
    """Get all rules with their rulemaking authority statutes.

    Returns:
        List of dicts with rule and authority statute info
    """
    query = """
    MATCH (r:Rule)-[:AUTHORITY]->(s:Statute)
    RETURN r.id AS rule_id, r.title AS rule_title,
           collect({id: s.id, section: s.section, title: s.title}) AS authority_statutes
    ORDER BY r.id
    """

    return client.run_query(query)
