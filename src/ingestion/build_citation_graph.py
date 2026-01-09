"""Build citation graph from extracted citations.

This module resolves citations to document IDs and builds edges
for the knowledge graph.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from .citation_extractor import (
    Citation,
    CitationRelation,
    CitationType,
    RelationType,
    detect_relation_type,
    extract_all_citations,
    get_context,
)


class ResolvedEdge(BaseModel):
    """A fully resolved citation edge for the graph."""

    source_chunk_id: str = Field(..., description="ID of source chunk")
    source_doc_id: str = Field(..., description="ID of source document")
    target_doc_id: str = Field(..., description="ID of target document")
    target_chunk_id: str | None = Field(default=None, description="ID of target parent chunk")
    relation_type: RelationType = Field(..., description="Type of relationship")
    citation_text: str = Field(..., description="Original citation text")
    confidence: float = Field(..., description="Resolution confidence (0.0 to 1.0)")


@dataclass
class CitationIndex:
    """Index for fast citation resolution."""

    # doc_id -> list of chunk IDs
    statute_to_chunks: dict[str, list[str]] = field(default_factory=dict)
    rule_to_chunks: dict[str, list[str]] = field(default_factory=dict)
    case_to_chunks: dict[str, list[str]] = field(default_factory=dict)
    taa_to_chunks: dict[str, list[str]] = field(default_factory=dict)

    # Normalized case citation -> case doc_id
    # "366 so 2d 1173" -> "case:1234567"
    case_citation_to_id: dict[str, str] = field(default_factory=dict)

    # Section number -> doc_id (for resolution)
    # "212.05" -> "statute:212.05"
    section_to_doc: dict[str, str] = field(default_factory=dict)

    # Rule number -> doc_id
    # "12A-1.005" -> "rule:12A-1.005"
    rule_to_doc: dict[str, str] = field(default_factory=dict)


def normalize_case_citation_for_index(citation: str) -> str:
    """Normalize case citation for matching.

    Args:
        citation: The citation string to normalize

    Returns:
        Normalized string for case-insensitive matching
    """
    # Remove extra whitespace
    citation = re.sub(r"\s+", " ", citation).strip()
    # Remove periods from reporter abbreviations
    citation = citation.replace(". ", " ").replace(".", "")
    # Lowercase for case-insensitive matching
    return citation.lower()


def build_citation_index(chunks: list[dict]) -> CitationIndex:
    """Build an index from chunks for fast citation resolution.

    Args:
        chunks: List of chunk dictionaries from chunks.json

    Returns:
        CitationIndex for resolution
    """
    index = CitationIndex()

    for chunk in chunks:
        chunk_id = chunk["id"]
        doc_id = chunk["doc_id"]
        doc_type = chunk["doc_type"]
        citation_text = chunk.get("citation", "")

        # Index by doc_id
        if doc_type == "statute":
            if doc_id not in index.statute_to_chunks:
                index.statute_to_chunks[doc_id] = []
            index.statute_to_chunks[doc_id].append(chunk_id)

            # Extract section number for resolution
            section = doc_id.replace("statute:", "")
            index.section_to_doc[section] = doc_id

        elif doc_type == "rule":
            if doc_id not in index.rule_to_chunks:
                index.rule_to_chunks[doc_id] = []
            index.rule_to_chunks[doc_id].append(chunk_id)

            # Extract rule number
            rule_num = doc_id.replace("rule:", "")
            index.rule_to_doc[rule_num] = doc_id

        elif doc_type == "case":
            if doc_id not in index.case_to_chunks:
                index.case_to_chunks[doc_id] = []
            index.case_to_chunks[doc_id].append(chunk_id)

            # Index normalized citation text for lookup
            # Citation format: "Case Name, 366 So. 2d 1173"
            if "," in citation_text:
                parts = citation_text.split(",", 1)
                if len(parts) > 1:
                    cite_part = parts[1].strip()
                    cite_norm = normalize_case_citation_for_index(cite_part)
                    index.case_citation_to_id[cite_norm] = doc_id

        elif doc_type == "taa":
            if doc_id not in index.taa_to_chunks:
                index.taa_to_chunks[doc_id] = []
            index.taa_to_chunks[doc_id].append(chunk_id)

    return index


def get_parent_chunk(chunks: list[str]) -> str | None:
    """Get the parent chunk ID from a list of chunk IDs.

    Parent chunks have index 0 by convention (e.g., "chunk:statute:212.05:0").

    Args:
        chunks: List of chunk IDs for a document

    Returns:
        The parent chunk ID, or the first chunk if no parent found
    """
    if not chunks:
        return None

    # Look for the parent chunk (index 0)
    for chunk_id in chunks:
        if chunk_id.endswith(":0"):
            return chunk_id

    # Fallback to first chunk
    return chunks[0]


def resolve_citation(
    citation: Citation,
    index: CitationIndex,
) -> tuple[str | None, str | None, float]:
    """Resolve a citation to a document and chunk ID.

    Uses multi-tier resolution:
    1. Exact match by section/rule number (confidence 1.0)
    2. Partial match without subsection (confidence 0.8)
    3. Case citation string matching (confidence 0.9)

    Args:
        citation: The extracted citation
        index: The citation index

    Returns:
        Tuple of (doc_id, chunk_id, confidence)
    """
    if citation.citation_type == CitationType.STATUTE:
        # Try exact match first
        if citation.section:
            doc_id = f"statute:{citation.section}"
            if doc_id in index.statute_to_chunks:
                parent_chunk = get_parent_chunk(index.statute_to_chunks[doc_id])
                return doc_id, parent_chunk, 1.0

        # Try without subsection for partial match
        base_section = citation.section
        if base_section and "(" in base_section:
            base_section = base_section.split("(")[0]
            doc_id = f"statute:{base_section}"
            if doc_id in index.statute_to_chunks:
                parent_chunk = get_parent_chunk(index.statute_to_chunks[doc_id])
                return doc_id, parent_chunk, 0.8

        # Try looking up in section_to_doc
        if citation.normalized in index.section_to_doc:
            doc_id = index.section_to_doc[citation.normalized]
            if doc_id in index.statute_to_chunks:
                parent_chunk = get_parent_chunk(index.statute_to_chunks[doc_id])
                return doc_id, parent_chunk, 1.0

        # Try base section in section_to_doc
        if base_section and base_section in index.section_to_doc:
            doc_id = index.section_to_doc[base_section]
            if doc_id in index.statute_to_chunks:
                parent_chunk = get_parent_chunk(index.statute_to_chunks[doc_id])
                return doc_id, parent_chunk, 0.8

        return None, None, 0.0

    elif citation.citation_type == CitationType.RULE:
        # Try exact match
        if citation.section:
            doc_id = f"rule:{citation.section}"
            if doc_id in index.rule_to_chunks:
                parent_chunk = get_parent_chunk(index.rule_to_chunks[doc_id])
                return doc_id, parent_chunk, 1.0

        # Try without subsection
        base_rule = citation.section
        if base_rule and "(" in base_rule:
            base_rule = base_rule.split("(")[0]
            doc_id = f"rule:{base_rule}"
            if doc_id in index.rule_to_chunks:
                parent_chunk = get_parent_chunk(index.rule_to_chunks[doc_id])
                return doc_id, parent_chunk, 0.8

        # Try looking up in rule_to_doc
        if citation.normalized in index.rule_to_doc:
            doc_id = index.rule_to_doc[citation.normalized]
            if doc_id in index.rule_to_chunks:
                parent_chunk = get_parent_chunk(index.rule_to_chunks[doc_id])
                return doc_id, parent_chunk, 1.0

        return None, None, 0.0

    elif citation.citation_type == CitationType.CASE:
        # Try to match by normalized citation
        cite_norm = normalize_case_citation_for_index(citation.normalized)
        if cite_norm in index.case_citation_to_id:
            doc_id = index.case_citation_to_id[cite_norm]
            if doc_id in index.case_to_chunks:
                parent_chunk = get_parent_chunk(index.case_to_chunks[doc_id])
                return doc_id, parent_chunk, 0.9

        return None, None, 0.0

    elif citation.citation_type == CitationType.CHAPTER:
        # Chapter references don't resolve to specific documents
        # They're informational but not graph edges
        return None, None, 0.0

    return None, None, 0.0


def extract_chunk_citations(
    chunk: dict,
    index: CitationIndex,
) -> list[CitationRelation]:
    """Extract and resolve citations from a single chunk.

    Args:
        chunk: Chunk dictionary
        index: Citation index for resolution

    Returns:
        List of citation relations
    """
    relations = []

    text = chunk["text"]
    chunk_id = chunk["id"]
    doc_id = chunk["doc_id"]
    doc_type = chunk["doc_type"]

    # Get source section for self-reference filtering
    source_section = doc_id.split(":", 1)[1] if ":" in doc_id else ""

    # Extract all citations
    citations = extract_all_citations(text)

    for citation in citations:
        # Skip self-references
        if citation.citation_type == CitationType.STATUTE:
            if citation.section == source_section:
                continue
            # Also skip if base section matches
            base_section = (
                citation.section.split("(")[0]
                if citation.section and "(" in citation.section
                else citation.section
            )
            if base_section == source_section:
                continue

        if citation.citation_type == CitationType.RULE:
            if citation.section == source_section:
                continue
            base_rule = (
                citation.section.split("(")[0]
                if citation.section and "(" in citation.section
                else citation.section
            )
            if base_rule == source_section:
                continue

        # Get context for relation detection
        context = get_context(text, citation.start_pos, citation.end_pos)

        # Detect relation type
        relation_type = detect_relation_type(citation, context, doc_type)

        # Resolve to target
        target_doc_id, target_chunk_id, confidence = resolve_citation(citation, index)

        relations.append(
            CitationRelation(
                source_chunk_id=chunk_id,
                source_doc_id=doc_id,
                target_citation=citation,
                relation_type=relation_type,
                context=context,
                target_doc_id=target_doc_id,
            )
        )

    return relations


def build_citation_graph(
    chunks: list[dict],
    min_confidence: float = 0.5,
) -> tuple[list[ResolvedEdge], list[CitationRelation]]:
    """Build the full citation graph from all chunks.

    Args:
        chunks: All chunks from chunks.json
        min_confidence: Minimum confidence for edge inclusion

    Returns:
        Tuple of (resolved_edges, unresolved_relations)
    """
    # Build index
    index = build_citation_index(chunks)

    resolved_edges = []
    unresolved_relations = []

    for chunk in chunks:
        relations = extract_chunk_citations(chunk, index)

        for relation in relations:
            if relation.target_doc_id:
                # Re-resolve to get chunk ID and confidence
                target_doc_id, target_chunk_id, confidence = resolve_citation(
                    relation.target_citation, index
                )

                if confidence >= min_confidence:
                    resolved_edges.append(
                        ResolvedEdge(
                            source_chunk_id=relation.source_chunk_id,
                            source_doc_id=relation.source_doc_id,
                            target_doc_id=target_doc_id,
                            target_chunk_id=target_chunk_id,
                            relation_type=relation.relation_type,
                            citation_text=relation.target_citation.raw_text,
                            confidence=confidence,
                        )
                    )
            else:
                # Only add to unresolved if it's a citation type we care about
                if relation.target_citation.citation_type != CitationType.CHAPTER:
                    unresolved_relations.append(relation)

    return resolved_edges, unresolved_relations


def deduplicate_edges(edges: list[ResolvedEdge]) -> list[ResolvedEdge]:
    """Remove duplicate edges, keeping highest confidence.

    Args:
        edges: List of resolved edges

    Returns:
        Deduplicated list of edges
    """
    seen: dict[tuple, ResolvedEdge] = {}

    for edge in edges:
        key = (edge.source_doc_id, edge.target_doc_id, edge.relation_type.value)
        if key not in seen or edge.confidence > seen[key].confidence:
            seen[key] = edge

    return list(seen.values())
