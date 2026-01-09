#!/usr/bin/env python3
"""Verify Weaviate vector store is loaded correctly.

Usage:
    python scripts/verify_vector_store.py

This script:
1. Checks total object count
2. Runs sample hybrid searches
3. Tests metadata filtering
4. Verifies search relevance
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vector.client import WeaviateClient
from src.vector.embeddings import VoyageEmbedder


def print_section(title: str) -> None:
    """Print a section header."""
    print()
    print("-" * 60)
    print(title)
    print("-" * 60)


def print_result(result, idx: int) -> None:
    """Print a single search result."""
    print(f"\n  [{idx + 1}] {result.citation or result.chunk_id}")
    print(f"      Type: {result.doc_type} | Level: {result.level} | Score: {result.score:.4f}")
    text_preview = result.text[:150].replace("\n", " ")
    if len(result.text) > 150:
        text_preview += "..."
    print(f"      Text: {text_preview}")


def main() -> int:
    """Main entry point."""
    print("=" * 60)
    print("FLORIDA TAX RAG - VECTOR STORE VERIFICATION")
    print("=" * 60)

    # Connect to Weaviate
    print_section("CONNECTION CHECK")

    client = WeaviateClient()

    if not client.health_check():
        print("ERROR: Weaviate is not reachable")
        return 1

    print("Weaviate connection: OK")

    # Check collection info
    print_section("COLLECTION INFO")

    info = client.get_collection_info()
    if not info:
        print("ERROR: LegalChunk collection not found")
        return 1

    print(f"Collection: {info['name']}")
    print(f"Object count: {info['object_count']}")
    print(f"Properties: {len(info['properties'])}")

    expected_count = 3022
    if info["object_count"] != expected_count:
        print(f"WARNING: Expected {expected_count} objects, found {info['object_count']}")
    else:
        print("Count verification: PASSED")

    # Initialize embedder for query embedding
    print_section("EMBEDDER INITIALIZATION")

    try:
        embedder = VoyageEmbedder()
        print("Voyage AI embedder: OK")
    except Exception as e:
        print(f"ERROR: Could not initialize embedder: {e}")
        return 1

    # Test 1: Basic hybrid search
    print_section("TEST 1: HYBRID SEARCH - Sales Tax Rate")

    query = "What is the Florida sales tax rate?"
    query_vector = embedder.embed_query(query)
    results = client.hybrid_search(query, query_vector, alpha=0.5, limit=5)

    print(f"Query: {query}")
    print(f"Results: {len(results)}")

    if not results:
        print("ERROR: No results returned")
    else:
        for i, r in enumerate(results[:3]):
            print_result(r, i)
        print("\n  Relevance check: Looking for '212.05' or 'sales tax'...")
        relevant = any("212.05" in r.text or "sales tax" in r.text.lower() for r in results[:3])
        print(f"  Relevance: {'PASSED' if relevant else 'NEEDS REVIEW'}")

    # Test 2: Specific citation search
    print_section("TEST 2: KEYWORD SEARCH - Specific Citation")

    query = "212.08 exemptions"
    results = client.keyword_search(query, limit=5)

    print(f"Query: {query}")
    print(f"Results: {len(results)}")

    if results:
        for i, r in enumerate(results[:3]):
            print_result(r, i)
        relevant = any("212.08" in r.citation for r in results[:3])
        print(f"\n  Citation match: {'PASSED' if relevant else 'NEEDS REVIEW'}")
    else:
        print("  No results found")

    # Test 3: Filter by doc_type = statute
    print_section("TEST 3: FILTERED SEARCH - Statutes Only")

    query = "tax exemption"
    query_vector = embedder.embed_query(query)
    results = client.hybrid_search(
        query, query_vector, alpha=0.5, limit=5, filters={"doc_type": "statute"}
    )

    print(f"Query: {query}")
    print("Filter: doc_type = statute")
    print(f"Results: {len(results)}")

    if results:
        for i, r in enumerate(results[:3]):
            print_result(r, i)
        all_statutes = all(r.doc_type == "statute" for r in results)
        print(f"\n  Filter check: {'PASSED' if all_statutes else 'FAILED'}")
    else:
        print("  No results found")

    # Test 4: Filter by doc_type = rule
    print_section("TEST 4: FILTERED SEARCH - Rules Only")

    query = "administrative procedure"
    query_vector = embedder.embed_query(query)
    results = client.hybrid_search(
        query, query_vector, alpha=0.5, limit=5, filters={"doc_type": "rule"}
    )

    print(f"Query: {query}")
    print("Filter: doc_type = rule")
    print(f"Results: {len(results)}")

    if results:
        for i, r in enumerate(results[:3]):
            print_result(r, i)
        all_rules = all(r.doc_type == "rule" for r in results)
        print(f"\n  Filter check: {'PASSED' if all_rules else 'FAILED'}")
    else:
        print("  No results (may be expected if no rules match)")

    # Test 5: Filter by doc_type = case
    print_section("TEST 5: FILTERED SEARCH - Cases Only")

    query = "sales tax"
    query_vector = embedder.embed_query(query)
    results = client.hybrid_search(
        query, query_vector, alpha=0.5, limit=5, filters={"doc_type": "case"}
    )

    print(f"Query: {query}")
    print("Filter: doc_type = case")
    print(f"Results: {len(results)}")

    if results:
        for i, r in enumerate(results[:3]):
            print_result(r, i)
        all_cases = all(r.doc_type == "case" for r in results)
        print(f"\n  Filter check: {'PASSED' if all_cases else 'FAILED'}")
    else:
        print("  No results (may be expected if no cases match)")

    # Test 6: Filter by doc_type = taa
    print_section("TEST 6: FILTERED SEARCH - TAAs Only")

    query = "technical assistance"
    query_vector = embedder.embed_query(query)
    results = client.hybrid_search(
        query, query_vector, alpha=0.5, limit=5, filters={"doc_type": "taa"}
    )

    print(f"Query: {query}")
    print("Filter: doc_type = taa")
    print(f"Results: {len(results)}")

    if results:
        for i, r in enumerate(results[:3]):
            print_result(r, i)
        all_taa = all(r.doc_type == "taa" for r in results)
        print(f"\n  Filter check: {'PASSED' if all_taa else 'FAILED'}")
    else:
        print("  No results (may be expected if no TAAs match)")

    # Test 7: Vector-only search
    print_section("TEST 7: VECTOR SEARCH - Semantic Similarity")

    query = "corporate income tax deductions for businesses"
    query_vector = embedder.embed_query(query)
    results = client.vector_search(query_vector, limit=5)

    print(f"Query: {query}")
    print(f"Results: {len(results)}")

    if results:
        for i, r in enumerate(results[:3]):
            print_result(r, i)
        # Check semantic relevance (should find tax-related content)
        relevant = any(
            "tax" in r.text.lower() or "income" in r.text.lower() or "deduct" in r.text.lower()
            for r in results[:3]
        )
        print(f"\n  Semantic relevance: {'PASSED' if relevant else 'NEEDS REVIEW'}")
    else:
        print("  No results found")

    # Test 8: Multiple filters
    print_section("TEST 8: MULTIPLE FILTERS")

    query = "exemption"
    query_vector = embedder.embed_query(query)
    results = client.hybrid_search(
        query, query_vector, alpha=0.5, limit=5, filters={"doc_type": "statute", "level": "parent"}
    )

    print(f"Query: {query}")
    print("Filters: doc_type=statute, level=parent")
    print(f"Results: {len(results)}")

    if results:
        for i, r in enumerate(results[:3]):
            print_result(r, i)
        all_match = all(r.doc_type == "statute" and r.level == "parent" for r in results)
        print(f"\n  Multi-filter check: {'PASSED' if all_match else 'FAILED'}")
    else:
        print("  No results found")

    # Summary
    print_section("SUMMARY")

    print(f"Total objects in Weaviate: {info['object_count']}")
    print("All basic tests completed.")
    print("\nVector store verification: COMPLETE")

    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)

    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
