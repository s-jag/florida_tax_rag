#!/usr/bin/env python3
"""Test the hybrid retrieval system with sample queries.

Usage:
    python scripts/test_retrieval.py [--query "your query"] [--top-k 10]
    python scripts/test_retrieval.py --all  # Run all test queries

Options:
    --query     Custom query to test
    --top-k     Number of results to return (default: 5)
    --all       Run all predefined test queries
    --no-graph  Disable graph expansion
    --no-rerank Disable reranking
    --alpha     Vector vs keyword balance (0.0=keyword, 1.0=vector, default: 0.5)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.graph.client import Neo4jClient
from src.retrieval import HybridRetriever
from src.vector.client import WeaviateClient
from src.vector.embeddings import VoyageEmbedder

# Predefined test queries for different scenarios
TEST_QUERIES = [
    {
        "query": "What is the Florida sales tax rate?",
        "expected": "Should return § 212.05 or related statutes",
    },
    {
        "query": "Is software consulting taxable in Florida?",
        "expected": "Should return TAAs or rules about service taxation",
    },
    {
        "query": "212.08 exemptions",
        "expected": "Should return § 212.08 chunks about exemptions",
    },
    {
        "query": "What are the sales tax exemptions for agriculture?",
        "expected": "Should return agriculture exemption provisions",
    },
    {
        "query": "How does Florida tax cloud computing services?",
        "expected": "Should return TAAs or rules about digital services",
    },
    {
        "query": "What is the penalty for late sales tax filing?",
        "expected": "Should return penalty provisions from statutes/rules",
    },
    {
        "query": "Are nonprofits exempt from Florida sales tax?",
        "expected": "Should return nonprofit exemption provisions",
    },
]


def print_result(result, index: int, verbose: bool = False) -> None:
    """Print a single retrieval result."""
    print(f"\n  [{index}] {result.citation or result.chunk_id}")
    print(f"      Type: {result.doc_type} | Level: {result.level} | Score: {result.score:.4f}")

    # Show score breakdown if available
    scores = []
    if result.vector_score is not None:
        scores.append(f"vec={result.vector_score:.3f}")
    if result.keyword_score is not None:
        scores.append(f"kw={result.keyword_score:.3f}")
    if result.graph_boost > 0:
        scores.append(f"graph_boost={result.graph_boost:.3f}")
    if scores:
        print(f"      Scores: {', '.join(scores)}")

    # Show text preview
    text_preview = result.text[:150].replace("\n", " ")
    if len(result.text) > 150:
        text_preview += "..."
    print(f"      Text: {text_preview}")

    # Show graph context if available
    if result.citation_context:
        print(f"      Citations: {len(result.citation_context)} related docs")
    if result.related_chunk_ids:
        print(f"      Related chunks: {len(result.related_chunk_ids)}")

    if verbose and result.ancestry:
        print(f"      Ancestry: {result.ancestry}")


def run_query(
    retriever: HybridRetriever,
    query: str,
    top_k: int = 5,
    expand_graph: bool = True,
    rerank: bool = True,
    alpha: float = 0.5,
    verbose: bool = False,
) -> None:
    """Run a single query and print results."""
    print(f"\nQuery: {query}")
    print("-" * 70)

    try:
        results = retriever.retrieve(
            query,
            top_k=top_k,
            alpha=alpha,
            expand_graph=expand_graph,
            rerank=rerank,
        )

        print(f"Results: {len(results)}")

        for i, result in enumerate(results, 1):
            print_result(result, i, verbose=verbose)

    except Exception as e:
        print(f"ERROR: {e}")
        raise


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test the hybrid retrieval system"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Custom query to test",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all predefined test queries",
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Disable graph expansion",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable reranking",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Vector vs keyword balance (0.0=keyword, 1.0=vector, default: 0.5)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("FLORIDA TAX RAG - HYBRID RETRIEVAL TEST")
    print("=" * 70)

    # Initialize clients
    print("\nInitializing clients...")

    try:
        weaviate = WeaviateClient()
        if not weaviate.health_check():
            print("ERROR: Weaviate is not reachable")
            return 1
        print("  Weaviate: OK")

        neo4j = Neo4jClient()
        neo4j_ok = neo4j.health_check()
        if neo4j_ok:
            print("  Neo4j: OK")
        else:
            print("  Neo4j: NOT AVAILABLE (graph expansion disabled)")

        embedder = VoyageEmbedder()
        print("  Embedder: OK")

        retriever = HybridRetriever(
            weaviate_client=weaviate,
            neo4j_client=neo4j,
            embedder=embedder,
        )

    except Exception as e:
        print(f"ERROR initializing clients: {e}")
        return 1

    # Configuration
    expand_graph = not args.no_graph and neo4j_ok
    rerank = not args.no_rerank

    print(f"\nConfiguration:")
    print(f"  Alpha: {args.alpha} (0=keyword, 1=vector)")
    print(f"  Graph expansion: {'ON' if expand_graph else 'OFF'}")
    print(f"  Reranking: {'ON' if rerank else 'OFF'}")
    print(f"  Top K: {args.top_k}")

    # Run queries
    if args.query:
        # Single custom query
        run_query(
            retriever,
            args.query,
            top_k=args.top_k,
            expand_graph=expand_graph,
            rerank=rerank,
            alpha=args.alpha,
            verbose=args.verbose,
        )
    elif args.all:
        # All test queries
        for test in TEST_QUERIES:
            print(f"\n{'='*70}")
            print(f"Expected: {test['expected']}")
            run_query(
                retriever,
                test["query"],
                top_k=args.top_k,
                expand_graph=expand_graph,
                rerank=rerank,
                alpha=args.alpha,
                verbose=args.verbose,
            )
    else:
        # Default: first few test queries
        for test in TEST_QUERIES[:3]:
            run_query(
                retriever,
                test["query"],
                top_k=args.top_k,
                expand_graph=expand_graph,
                rerank=rerank,
                alpha=args.alpha,
                verbose=args.verbose,
            )

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)

    # Cleanup
    weaviate.close()
    neo4j.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
