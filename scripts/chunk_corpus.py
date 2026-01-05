#!/usr/bin/env python3
"""Chunk the consolidated corpus into hierarchical legal chunks.

This script loads the corpus from data/processed/corpus.json,
applies hierarchical chunking to all documents, and saves the
chunks to data/processed/chunks.json.
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

from src.ingestion import Corpus, LegalDocument, chunk_corpus, ChunkLevel


def load_corpus(corpus_path: Path) -> Corpus:
    """Load the corpus from JSON file."""
    with open(corpus_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Reconstruct Corpus from JSON
    from src.ingestion.models import CorpusMetadata, DocumentType

    metadata = CorpusMetadata(
        created_at=datetime.fromisoformat(data["metadata"]["created_at"]),
        total_documents=data["metadata"]["total_documents"],
        by_type=data["metadata"]["by_type"],
        version=data["metadata"].get("version", "1.0"),
    )

    documents = []
    for doc_data in data["documents"]:
        # Parse effective_date if present
        effective_date = None
        if doc_data.get("effective_date"):
            from datetime import date as date_type
            effective_date = date_type.fromisoformat(doc_data["effective_date"])

        # Parse scraped_at
        scraped_at = datetime.fromisoformat(doc_data["scraped_at"])

        doc = LegalDocument(
            id=doc_data["id"],
            doc_type=DocumentType(doc_data["doc_type"]),
            title=doc_data["title"],
            full_citation=doc_data["full_citation"],
            text=doc_data["text"],
            effective_date=effective_date,
            source_url=doc_data["source_url"],
            parent_id=doc_data.get("parent_id"),
            children_ids=doc_data.get("children_ids", []),
            cites_statutes=doc_data.get("cites_statutes", []),
            cites_rules=doc_data.get("cites_rules", []),
            cites_cases=doc_data.get("cites_cases", []),
            scraped_at=scraped_at,
            metadata=doc_data.get("metadata", {}),
        )
        documents.append(doc)

    return Corpus(metadata=metadata, documents=documents)


def main():
    """Main entry point."""
    corpus_path = PROJECT_ROOT / "data" / "processed" / "corpus.json"
    output_path = PROJECT_ROOT / "data" / "processed" / "chunks.json"

    print("=" * 60)
    print("CORPUS CHUNKING")
    print("=" * 60)
    print(f"Input: {corpus_path}")
    print(f"Output: {output_path}")
    print()

    # Load corpus
    print("Loading corpus...")
    corpus = load_corpus(corpus_path)
    print(f"  Loaded {len(corpus.documents)} documents")
    print()

    # Chunk all documents
    print("Chunking documents...")
    start_time = time.time()
    chunks = chunk_corpus(corpus)
    elapsed = time.time() - start_time
    print(f"  Created {len(chunks)} chunks in {elapsed:.2f}s")
    print()

    # Compute statistics
    level_counts = Counter(c.level.value for c in chunks)
    type_counts = Counter(c.doc_type for c in chunks)
    token_counts = [c.token_count for c in chunks]

    print("-" * 40)
    print("CHUNKING STATISTICS")
    print("-" * 40)
    print(f"Total chunks: {len(chunks)}")
    print()
    print("By level:")
    print(f"  Parent: {level_counts.get('parent', 0)}")
    print(f"  Child: {level_counts.get('child', 0)}")
    print()
    print("By document type:")
    for doc_type in ["statute", "rule", "taa", "case"]:
        print(f"  {doc_type}: {type_counts.get(doc_type, 0)}")
    print()
    print("Token statistics:")
    if token_counts:
        print(f"  Average: {sum(token_counts) // len(token_counts)} tokens")
        print(f"  Min: {min(token_counts)} tokens")
        print(f"  Max: {max(token_counts)} tokens")

        # Token distribution
        under_500 = sum(1 for t in token_counts if t < 500)
        under_1000 = sum(1 for t in token_counts if 500 <= t < 1000)
        under_2000 = sum(1 for t in token_counts if 1000 <= t < 2000)
        over_2000 = sum(1 for t in token_counts if t >= 2000)
        print()
        print("Token distribution:")
        print(f"  <500 tokens: {under_500} ({100*under_500//len(token_counts)}%)")
        print(f"  500-999 tokens: {under_1000} ({100*under_1000//len(token_counts)}%)")
        print(f"  1000-1999 tokens: {under_2000} ({100*under_2000//len(token_counts)}%)")
        print(f"  2000+ tokens: {over_2000} ({100*over_2000//len(token_counts)}%)")

    # Save chunks
    print()
    print(f"Saving chunks to: {output_path}")

    output_data = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "total_chunks": len(chunks),
            "by_level": dict(level_counts),
            "by_type": dict(type_counts),
            "avg_tokens": sum(token_counts) // len(token_counts) if token_counts else 0,
            "source_corpus": str(corpus_path.name),
        },
        "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)

    output_size = output_path.stat().st_size / (1024 * 1024)
    print(f"Output file size: {output_size:.2f} MB")
    print()
    print("Done!")


if __name__ == "__main__":
    main()
