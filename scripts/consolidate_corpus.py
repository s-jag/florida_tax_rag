#!/usr/bin/env python3
"""Consolidate all raw scraped data into a unified corpus.

This script loads all raw data from the various scrapers and
consolidates it into a single JSON file with a unified schema.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion import consolidate_all


def main():
    """Main entry point."""
    data_dir = PROJECT_ROOT / "data" / "raw"
    output_dir = PROJECT_ROOT / "data" / "processed"

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CORPUS CONSOLIDATION")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # Run consolidation
    start_time = time.time()
    corpus = consolidate_all(data_dir)
    elapsed = time.time() - start_time

    print()
    print("-" * 40)
    print(f"Consolidation complete in {elapsed:.2f}s")
    print(f"Total documents: {corpus.metadata.total_documents}")
    print(f"  - Statutes: {corpus.metadata.by_type.get('statute', 0)}")
    print(f"  - Rules: {corpus.metadata.by_type.get('rule', 0)}")
    print(f"  - TAAs: {corpus.metadata.by_type.get('taa', 0)}")
    print(f"  - Cases: {corpus.metadata.by_type.get('case', 0)}")

    # Compute cross-reference statistics
    total_statute_refs = 0
    total_rule_refs = 0
    total_case_refs = 0

    for doc in corpus.documents:
        total_statute_refs += len(doc.cites_statutes)
        total_rule_refs += len(doc.cites_rules)
        total_case_refs += len(doc.cites_cases)

    print()
    print("Cross-reference counts:")
    print(f"  - Statute citations: {total_statute_refs}")
    print(f"  - Rule citations: {total_rule_refs}")
    print(f"  - Case citations: {total_case_refs}")

    # Calculate average text lengths
    text_lengths = {"statute": [], "rule": [], "taa": [], "case": []}
    for doc in corpus.documents:
        text_lengths[doc.doc_type.value].append(len(doc.text))

    print()
    print("Average text lengths:")
    for doc_type, lengths in text_lengths.items():
        if lengths:
            avg = sum(lengths) // len(lengths)
            print(f"  - {doc_type}: {avg:,} chars")

    # Save corpus
    corpus_path = output_dir / "corpus.json"
    print()
    print(f"Saving corpus to: {corpus_path}")

    # Convert to dict for JSON serialization
    corpus_dict = {
        "metadata": {
            "created_at": corpus.metadata.created_at.isoformat(),
            "total_documents": corpus.metadata.total_documents,
            "by_type": corpus.metadata.by_type,
            "version": corpus.metadata.version,
        },
        "documents": [doc.model_dump(mode="json") for doc in corpus.documents],
    }

    with open(corpus_path, "w", encoding="utf-8") as f:
        json.dump(corpus_dict, f, indent=2, ensure_ascii=False, default=str)

    corpus_size = corpus_path.stat().st_size / (1024 * 1024)
    print(f"Corpus file size: {corpus_size:.2f} MB")

    # Save statistics separately
    stats_path = output_dir / "statistics.json"
    stats = {
        "consolidation_timestamp": datetime.now().isoformat(),
        "processing_time_seconds": round(elapsed, 2),
        "total_documents": corpus.metadata.total_documents,
        "by_type": corpus.metadata.by_type,
        "cross_references": {
            "statute_citations": total_statute_refs,
            "rule_citations": total_rule_refs,
            "case_citations": total_case_refs,
        },
        "text_lengths": {
            doc_type: {
                "count": len(lengths),
                "avg": sum(lengths) // len(lengths) if lengths else 0,
                "min": min(lengths) if lengths else 0,
                "max": max(lengths) if lengths else 0,
            }
            for doc_type, lengths in text_lengths.items()
        },
    }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Statistics saved to: {stats_path}")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
