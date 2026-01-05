"""Data loading functions for Neo4j knowledge graph."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .client import Neo4jClient
from .schema import get_schema_queries

logger = logging.getLogger(__name__)

# Batch size for UNWIND operations
BATCH_SIZE = 500


def init_schema(client: Neo4jClient) -> None:
    """Initialize Neo4j schema with constraints and indexes.

    Args:
        client: Neo4j client instance
    """
    logger.info("Initializing Neo4j schema")

    for query in get_schema_queries():
        logger.debug(f"Running schema query: {query[:60]}...")
        client.run_write(query)

    logger.info("Schema initialization complete")


def load_documents(
    client: Neo4jClient, documents: list[dict[str, Any]]
) -> dict[str, int]:
    """Load Document nodes from corpus.

    Creates nodes with appropriate labels based on doc_type:
    - statute -> :Document:Statute
    - rule -> :Document:Rule
    - case -> :Document:Case
    - taa -> :Document:TAA

    Args:
        client: Neo4j client
        documents: List of document dicts from corpus.json

    Returns:
        Statistics dict with counts
    """
    logger.info(f"Loading {len(documents)} documents")

    # Group by doc_type for labeled creation
    by_type: dict[str, list[dict[str, Any]]] = {
        "statute": [],
        "rule": [],
        "case": [],
        "taa": [],
    }

    for doc in documents:
        doc_type = doc["doc_type"]

        # Extract searchable fields from metadata
        metadata = doc.get("metadata", {})

        # Extract section from doc_id if not in metadata
        section = doc["id"].split(":")[-1] if ":" in doc["id"] else None

        # Extract chapter from metadata or section
        chapter = metadata.get("chapter")
        if chapter is None and section and "." in str(section):
            chapter = str(section).split(".")[0]

        node_data = {
            "id": doc["id"],
            "doc_type": doc_type,
            "title": doc["title"],
            "full_citation": doc["full_citation"],
            "source_url": doc.get("source_url", ""),
            "effective_date": doc.get("effective_date"),
            "section": str(section) if section else None,
            "chapter": str(chapter) if chapter else None,
        }

        if doc_type in by_type:
            by_type[doc_type].append(node_data)

    totals: dict[str, int] = {"nodes_created": 0}

    # Create nodes with type-specific labels
    type_label_map = {
        "statute": "Statute",
        "rule": "Rule",
        "case": "Case",
        "taa": "TAA",
    }

    for doc_type, nodes in by_type.items():
        if not nodes:
            continue

        label = type_label_map[doc_type]
        query = f"""
        UNWIND $docs AS doc
        CREATE (d:Document:{label} {{
            id: doc.id,
            doc_type: doc.doc_type,
            title: doc.title,
            full_citation: doc.full_citation,
            source_url: doc.source_url,
            effective_date: doc.effective_date,
            section: doc.section,
            chapter: doc.chapter
        }})
        """

        result = client.batch_write(query, "docs", nodes, BATCH_SIZE)
        totals["nodes_created"] += result["nodes_created"]
        logger.info(f"Loaded {len(nodes)} {label} documents")

    return totals


def load_chunks(
    client: Neo4jClient, chunks: list[dict[str, Any]]
) -> dict[str, int]:
    """Load Chunk nodes and link to Documents.

    Args:
        client: Neo4j client
        chunks: List of chunk dicts from chunks.json

    Returns:
        Statistics dict with counts
    """
    logger.info(f"Loading {len(chunks)} chunks")

    # Prepare chunk data (exclude full text)
    chunk_nodes = []
    for chunk in chunks:
        chunk_nodes.append(
            {
                "id": chunk["id"],
                "doc_id": chunk["doc_id"],
                "level": chunk["level"],
                "ancestry": chunk["ancestry"],
                "subsection_path": chunk.get("subsection_path", ""),
                "citation": chunk["citation"],
                "doc_type": chunk["doc_type"],
                "token_count": chunk["token_count"],
            }
        )

    # Create Chunk nodes
    create_query = """
    UNWIND $chunks AS chunk
    CREATE (c:Chunk {
        id: chunk.id,
        doc_id: chunk.doc_id,
        level: chunk.level,
        ancestry: chunk.ancestry,
        subsection_path: chunk.subsection_path,
        citation: chunk.citation,
        doc_type: chunk.doc_type,
        token_count: chunk.token_count
    })
    """

    result = client.batch_write(create_query, "chunks", chunk_nodes, BATCH_SIZE)
    logger.info(f"Created {result['nodes_created']} chunk nodes")

    # Create HAS_CHUNK relationships (Document -> Chunk)
    link_query = """
    UNWIND $chunks AS chunk
    MATCH (d:Document {id: chunk.doc_id})
    MATCH (c:Chunk {id: chunk.id})
    CREATE (d)-[:HAS_CHUNK]->(c)
    """

    link_result = client.batch_write(link_query, "chunks", chunk_nodes, BATCH_SIZE)
    logger.info(f"Created {link_result['relationships_created']} HAS_CHUNK edges")

    return {
        "nodes_created": result["nodes_created"],
        "relationships_created": link_result["relationships_created"],
    }


def load_hierarchy(
    client: Neo4jClient, chunks: list[dict[str, Any]]
) -> dict[str, int]:
    """Create CHILD_OF relationships between chunks.

    Args:
        client: Neo4j client
        chunks: List of chunk dicts from chunks.json

    Returns:
        Statistics dict with counts
    """
    # Filter to only child chunks with parent references
    child_chunks = [
        {"id": c["id"], "parent_id": c["parent_chunk_id"]}
        for c in chunks
        if c.get("parent_chunk_id")
    ]

    if not child_chunks:
        logger.info("No chunk hierarchy to load")
        return {"relationships_created": 0}

    logger.info(f"Loading {len(child_chunks)} chunk hierarchy edges")

    query = """
    UNWIND $children AS child
    MATCH (c:Chunk {id: child.id})
    MATCH (p:Chunk {id: child.parent_id})
    CREATE (c)-[:CHILD_OF]->(p)
    """

    result = client.batch_write(query, "children", child_chunks, BATCH_SIZE)
    logger.info(f"Created {result['relationships_created']} CHILD_OF edges")

    return result


def load_citations(
    client: Neo4jClient, edges: list[dict[str, Any]]
) -> dict[str, int]:
    """Create citation relationships between documents.

    Args:
        client: Neo4j client
        edges: List of edge dicts from citation_graph.json

    Returns:
        Statistics dict with counts
    """
    logger.info(f"Loading {len(edges)} citation edges")

    # Group edges by relation type for typed relationship creation
    by_type: dict[str, list[dict[str, Any]]] = {}

    for edge in edges:
        rel_type = edge["relation_type"].upper()
        if rel_type not in by_type:
            by_type[rel_type] = []

        by_type[rel_type].append(
            {
                "source_doc_id": edge["source_doc_id"],
                "source_chunk_id": edge["source_chunk_id"],
                "target_doc_id": edge["target_doc_id"],
                "target_chunk_id": edge.get("target_chunk_id"),
                "citation_text": edge["citation_text"],
                "confidence": edge["confidence"],
            }
        )

    totals: dict[str, int] = {"relationships_created": 0}

    for rel_type, rel_edges in by_type.items():
        # Create relationships at document level
        query = f"""
        UNWIND $edges AS edge
        MATCH (s:Document {{id: edge.source_doc_id}})
        MATCH (t:Document {{id: edge.target_doc_id}})
        CREATE (s)-[r:{rel_type} {{
            citation_text: edge.citation_text,
            confidence: edge.confidence,
            source_chunk_id: edge.source_chunk_id,
            target_chunk_id: edge.target_chunk_id
        }}]->(t)
        """

        result = client.batch_write(query, "edges", rel_edges, BATCH_SIZE)
        totals["relationships_created"] += result["relationships_created"]
        logger.info(f"Created {result['relationships_created']} {rel_type} edges")

    return totals


def load_all(
    client: Neo4jClient,
    corpus_path: Path,
    chunks_path: Path,
    citations_path: Path,
) -> dict[str, Any]:
    """Load all data into Neo4j.

    Args:
        client: Neo4j client
        corpus_path: Path to corpus.json
        chunks_path: Path to chunks.json
        citations_path: Path to citation_graph.json

    Returns:
        Complete statistics dict
    """
    stats: dict[str, Any] = {
        "documents": {"count": 0, "nodes_created": 0},
        "chunks": {"count": 0, "nodes_created": 0, "relationships_created": 0},
        "hierarchy": {"relationships_created": 0},
        "citations": {"count": 0, "relationships_created": 0},
    }

    # Load corpus
    logger.info(f"Loading corpus from {corpus_path}")
    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)

    documents = corpus["documents"]
    stats["documents"]["count"] = len(documents)
    doc_result = load_documents(client, documents)
    stats["documents"]["nodes_created"] = doc_result["nodes_created"]

    # Load chunks
    logger.info(f"Loading chunks from {chunks_path}")
    with open(chunks_path, encoding="utf-8") as f:
        chunks_data = json.load(f)

    chunks = chunks_data["chunks"]
    stats["chunks"]["count"] = len(chunks)
    chunk_result = load_chunks(client, chunks)
    stats["chunks"]["nodes_created"] = chunk_result["nodes_created"]
    stats["chunks"]["relationships_created"] = chunk_result["relationships_created"]

    # Load hierarchy
    hierarchy_result = load_hierarchy(client, chunks)
    stats["hierarchy"]["relationships_created"] = hierarchy_result[
        "relationships_created"
    ]

    # Load citations
    logger.info(f"Loading citations from {citations_path}")
    with open(citations_path, encoding="utf-8") as f:
        citation_data = json.load(f)

    edges = citation_data["edges"]
    stats["citations"]["count"] = len(edges)
    citation_result = load_citations(client, edges)
    stats["citations"]["relationships_created"] = citation_result[
        "relationships_created"
    ]

    return stats
