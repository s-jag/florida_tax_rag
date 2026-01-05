#!/usr/bin/env python3
"""Test query decomposition with sample queries.

Usage:
    python scripts/test_decomposition.py [--query "your query"]
    python scripts/test_decomposition.py --all
    python scripts/test_decomposition.py --multi  # Test full multi-query retrieval

Options:
    --query     Custom query to test decomposition
    --all       Run all predefined test queries
    --multi     Also run multi-query retrieval (requires services)
    --no-llm    Skip LLM calls, only test heuristics
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Predefined test queries with expected behavior
TEST_QUERIES = [
    {
        "query": "Do I owe sales tax on software consulting services in Miami?",
        "expected_types": ["definition", "exemption", "local"],
        "expected_simple": False,
        "description": "Complex: software + consulting + Miami location",
    },
    {
        "query": "What are the homestead exemption requirements?",
        "expected_types": ["definition", "exemption"],
        "expected_simple": False,
        "description": "Moderate: exemption with requirements",
    },
    {
        "query": "How do I contest a property tax assessment?",
        "expected_types": ["procedure"],
        "expected_simple": False,
        "description": "Procedural question",
    },
    {
        "query": "Is there sales tax on SaaS products delivered to Florida customers?",
        "expected_types": ["definition", "exemption", "statute"],
        "expected_simple": False,
        "description": "Complex: SaaS taxation",
    },
    {
        "query": "What is the Florida sales tax rate?",
        "expected_types": [],
        "expected_simple": True,
        "description": "Simple: direct rate question",
    },
    {
        "query": "Are groceries taxable?",
        "expected_types": [],
        "expected_simple": True,
        "description": "Simple: yes/no exemption question",
    },
    {
        "query": "What are the penalties for late sales tax filing and also what interest accrues?",
        "expected_types": ["penalty", "temporal"],
        "expected_simple": False,
        "description": "Complex: penalties + interest",
    },
]


async def test_decomposition(query: str, verbose: bool = False) -> dict:
    """Test query decomposition for a single query.

    Args:
        query: The query to decompose
        verbose: Whether to print detailed output

    Returns:
        dict with decomposition results
    """
    from src.retrieval import QueryDecomposer, create_decomposer

    decomposer = create_decomposer()

    # Test heuristic first
    should_decompose = decomposer._should_decompose(query)

    if verbose:
        print(f"\n  Heuristic says decompose: {should_decompose}")

    # Run full decomposition
    result = await decomposer.decompose(query)

    return {
        "query": query,
        "heuristic": should_decompose,
        "is_simple": result.is_simple,
        "sub_query_count": result.query_count,
        "sub_queries": result.sub_queries,
        "reasoning": result.reasoning,
    }


async def test_multi_retrieval(query: str, top_k: int = 5) -> None:
    """Test full multi-query retrieval pipeline.

    Args:
        query: The query to test
        top_k: Number of results to return
    """
    from src.retrieval import create_multi_retriever

    print(f"\n{'='*70}")
    print("MULTI-QUERY RETRIEVAL TEST")
    print(f"{'='*70}")

    try:
        retriever = create_multi_retriever()
    except Exception as e:
        print(f"ERROR: Could not create retriever: {e}")
        return

    print(f"\nQuery: {query}")
    print("-" * 70)

    result = await retriever.retrieve(query, top_k=top_k)

    print(f"\nDecomposition: {'Simple' if result.decomposition.is_simple else 'Complex'}")
    print(f"Reasoning: {result.decomposition.reasoning}")

    if not result.decomposition.is_simple:
        print(f"\nSub-queries ({result.decomposition.query_count}):")
        for sq in result.decomposition.sub_queries:
            print(f"  [{sq.priority}] {sq.type.value}: {sq.text}")

        print(f"\nSub-query Results:")
        for sqr in result.sub_query_results:
            print(f"  {sqr.sub_query.type.value}: {sqr.result_count} results")

    print(f"\nMerged Results: {len(result.merged_results)}")
    print(f"Unique Documents: {result.unique_doc_ids}")
    print(f"Unique Chunks: {result.unique_chunk_ids}")

    print("\nTop Results:")
    for i, r in enumerate(result.merged_results[:5], 1):
        print(f"\n  [{i}] {r.citation or r.chunk_id}")
        print(f"      Type: {r.doc_type} | Score: {r.score:.4f}")
        text_preview = r.text[:100].replace("\n", " ")
        print(f"      Text: {text_preview}...")


async def run_all_tests(verbose: bool = False) -> None:
    """Run all test queries.

    Args:
        verbose: Whether to print detailed output
    """
    print("=" * 70)
    print("QUERY DECOMPOSITION TEST")
    print("=" * 70)

    results = []
    for test in TEST_QUERIES:
        print(f"\n{'-'*70}")
        print(f"Query: {test['query']}")
        print(f"Description: {test['description']}")
        print(f"Expected: simple={test['expected_simple']}, types={test['expected_types']}")

        try:
            result = await test_decomposition(test["query"], verbose=verbose)
            results.append(result)

            print(f"\nResult:")
            print(f"  Is Simple: {result['is_simple']}")
            print(f"  Reasoning: {result['reasoning']}")

            if not result["is_simple"]:
                print(f"  Sub-queries ({result['sub_query_count']}):")
                for sq in result["sub_queries"]:
                    print(f"    [{sq.priority}] {sq.type.value}: {sq.text}")

            # Check against expected
            if result["is_simple"] == test["expected_simple"]:
                print("\n  [PASS] Simple/complex classification matches expected")
            else:
                print(
                    f"\n  [WARN] Expected simple={test['expected_simple']}, "
                    f"got simple={result['is_simple']}"
                )

        except Exception as e:
            print(f"\n  [ERROR] {e}")
            results.append({"query": test["query"], "error": str(e)})

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    simple_count = sum(1 for r in results if r.get("is_simple", False))
    complex_count = len(results) - simple_count
    print(f"Total queries: {len(results)}")
    print(f"Simple: {simple_count}")
    print(f"Complex: {complex_count}")


async def run_single_query(query: str, multi: bool = False, top_k: int = 5) -> None:
    """Run decomposition for a single query.

    Args:
        query: The query to test
        multi: Whether to also run multi-query retrieval
        top_k: Number of results for multi-query retrieval
    """
    print("=" * 70)
    print("QUERY DECOMPOSITION TEST")
    print("=" * 70)

    print(f"\nQuery: {query}")
    print("-" * 70)

    result = await test_decomposition(query, verbose=True)

    print(f"\nResult:")
    print(f"  Heuristic: {'decompose' if result['heuristic'] else 'simple'}")
    print(f"  LLM says: {'simple' if result['is_simple'] else 'complex'}")
    print(f"  Reasoning: {result['reasoning']}")

    if not result["is_simple"]:
        print(f"\n  Sub-queries ({result['sub_query_count']}):")
        for sq in result["sub_queries"]:
            print(f"    [{sq.priority}] {sq.type.value}: {sq.text}")

    if multi:
        await test_multi_retrieval(query, top_k=top_k)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test query decomposition")
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Custom query to test",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all predefined test queries",
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Also run multi-query retrieval",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results for multi-query retrieval",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    if args.query:
        asyncio.run(run_single_query(args.query, multi=args.multi, top_k=args.top_k))
    elif args.all:
        asyncio.run(run_all_tests(verbose=args.verbose))
    else:
        # Default: run first 3 test queries
        async def default_run():
            print("Running first 3 test queries (use --all for all)...")
            for test in TEST_QUERIES[:3]:
                print(f"\n{'='*70}")
                print(f"Query: {test['query']}")
                result = await test_decomposition(test["query"])
                print(f"Simple: {result['is_simple']}")
                print(f"Sub-queries: {result['sub_query_count']}")
                if not result["is_simple"]:
                    for sq in result["sub_queries"]:
                        print(f"  - {sq.type.value}: {sq.text}")

        asyncio.run(default_run())

    return 0


if __name__ == "__main__":
    sys.exit(main())
