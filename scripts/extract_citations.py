#!/usr/bin/env python3
"""Extract citations from chunks and build citation graph.

This script:
1. Loads chunks from data/processed/chunks.json
2. Extracts all citations from each chunk
3. Resolves citations to document/chunk IDs
4. Saves the citation graph to data/processed/citation_graph.json

Usage:
    python scripts/extract_citations.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.build_citation_graph import (
    ResolvedEdge,
    build_citation_graph,
    deduplicate_edges,
)
from src.ingestion.citation_extractor import CitationRelation, RelationType


def load_chunks(chunks_path: Path) -> list[dict]:
    """Load chunks from JSON file."""
    with open(chunks_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["chunks"]


def edge_to_dict(edge: ResolvedEdge) -> dict:
    """Convert ResolvedEdge to serializable dict."""
    return {
        "source_chunk_id": edge.source_chunk_id,
        "source_doc_id": edge.source_doc_id,
        "target_doc_id": edge.target_doc_id,
        "target_chunk_id": edge.target_chunk_id,
        "relation_type": edge.relation_type.value,
        "citation_text": edge.citation_text,
        "confidence": edge.confidence,
    }


def unresolved_to_dict(rel: CitationRelation) -> dict:
    """Convert unresolved CitationRelation to serializable dict."""
    return {
        "source_chunk_id": rel.source_chunk_id,
        "source_doc_id": rel.source_doc_id,
        "citation_type": rel.target_citation.citation_type.value,
        "citation_normalized": rel.target_citation.normalized,
        "citation_raw": rel.target_citation.raw_text,
    }


def main():
    """Main entry point."""
    chunks_path = PROJECT_ROOT / "data" / "processed" / "chunks.json"
    output_path = PROJECT_ROOT / "data" / "processed" / "citation_graph.json"

    print("=" * 60)
    print("CITATION EXTRACTION")
    print("=" * 60)
    print(f"Input: {chunks_path}")
    print(f"Output: {output_path}")
    print()

    # Load chunks
    print("Loading chunks...")
    chunks = load_chunks(chunks_path)
    print(f"  Loaded {len(chunks)} chunks")
    print()

    # Extract and resolve citations
    print("Extracting citations...")
    start_time = time.time()

    resolved_edges, unresolved = build_citation_graph(chunks)
    elapsed = time.time() - start_time

    total_citations = len(resolved_edges) + len(unresolved)
    print(f"  Extracted {total_citations} total citations in {elapsed:.2f}s")
    print(f"  Resolved: {len(resolved_edges)}")
    print(f"  Unresolved: {len(unresolved)}")
    print()

    # Deduplicate edges
    print("Deduplicating edges...")
    unique_edges = deduplicate_edges(resolved_edges)
    print(f"  {len(resolved_edges)} -> {len(unique_edges)} unique edges")
    print()

    # Statistics
    print("-" * 40)
    print("CITATION STATISTICS")
    print("-" * 40)

    # By relation type
    relation_counts = Counter(e.relation_type.value for e in unique_edges)
    print("By relation type:")
    for rel_type in RelationType:
        count = relation_counts.get(rel_type.value, 0)
        print(f"  {rel_type.value}: {count}")
    print()

    # By source doc type
    source_type_counts = Counter(e.source_doc_id.split(":")[0] for e in unique_edges)
    print("By source document type:")
    for doc_type, count in sorted(source_type_counts.items()):
        print(f"  {doc_type}: {count}")
    print()

    # By target doc type
    target_type_counts = Counter(e.target_doc_id.split(":")[0] for e in unique_edges)
    print("By target document type:")
    for doc_type, count in sorted(target_type_counts.items()):
        print(f"  {doc_type}: {count}")
    print()

    # Confidence distribution
    high_conf = sum(1 for e in unique_edges if e.confidence >= 0.9)
    med_conf = sum(1 for e in unique_edges if 0.7 <= e.confidence < 0.9)
    low_conf = sum(1 for e in unique_edges if e.confidence < 0.7)
    print("By confidence:")
    print(f"  High (>=0.9): {high_conf}")
    print(f"  Medium (0.7-0.9): {med_conf}")
    print(f"  Low (<0.7): {low_conf}")
    print()

    # Sample unresolved
    if unresolved:
        print("-" * 40)
        print("SAMPLE UNRESOLVED CITATIONS (first 10)")
        print("-" * 40)

        # Group by citation type
        unresolved_by_type = Counter(r.target_citation.citation_type.value for r in unresolved)
        print("Unresolved by type:")
        for ctype, count in sorted(unresolved_by_type.items()):
            print(f"  {ctype}: {count}")
        print()

        print("Examples:")
        for rel in unresolved[:10]:
            print(f"  From: {rel.source_doc_id}")
            print(f"    Citation: {rel.target_citation.normalized}")
            print(f"    Type: {rel.target_citation.citation_type.value}")
            print()

    # Save output
    print("-" * 40)
    print("SAVING OUTPUT")
    print("-" * 40)

    output_data = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "total_edges": len(unique_edges),
            "total_unresolved": len(unresolved),
            "by_relation_type": dict(relation_counts),
            "by_source_type": dict(source_type_counts),
            "by_target_type": dict(target_type_counts),
            "source_chunks": str(chunks_path.name),
        },
        "edges": [edge_to_dict(e) for e in unique_edges],
        "unresolved": [unresolved_to_dict(r) for r in unresolved],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    output_size = output_path.stat().st_size / 1024
    print(f"Saved to: {output_path}")
    print(f"File size: {output_size:.2f} KB")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
