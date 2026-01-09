#!/usr/bin/env python3
"""Analyze retrieval quality for the Florida Tax RAG system.

Usage:
    python scripts/analyze_retrieval.py                    # Full analysis
    python scripts/analyze_retrieval.py --method vector    # Single method
    python scripts/analyze_retrieval.py --tune-alpha       # Alpha tuning only
    python scripts/analyze_retrieval.py --debug "query"    # Debug single query
    python scripts/analyze_retrieval.py --question eval_001  # Single question
    python scripts/analyze_retrieval.py --failed-only      # Show only failed queries
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import EvalDataset
from src.evaluation.retrieval_analysis import (
    RetrievalAnalyzer,
    debug_retrieval,
    generate_retrieval_markdown_report,
)
from src.graph.client import Neo4jClient
from src.retrieval.hybrid import HybridRetriever
from src.vector.client import WeaviateClient
from src.vector.embeddings import VoyageEmbedder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_dataset(path: str) -> EvalDataset:
    """Load evaluation dataset."""
    with open(path) as f:
        data = json.load(f)
    return EvalDataset(**data)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze retrieval quality for Florida Tax RAG")
    parser.add_argument(
        "--dataset",
        default="data/evaluation/golden_dataset.json",
        help="Path to golden dataset",
    )
    parser.add_argument(
        "--output",
        default="RETRIEVAL_ANALYSIS.md",
        help="Output file path",
    )
    parser.add_argument(
        "--method",
        choices=["vector", "keyword", "hybrid", "graph"],
        help="Analyze single method only",
    )
    parser.add_argument(
        "--tune-alpha",
        action="store_true",
        help="Run alpha tuning only",
    )
    parser.add_argument(
        "--alphas",
        type=str,
        default="0.0,0.25,0.5,0.75,1.0",
        help="Comma-separated alpha values to test",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of results to analyze",
    )
    parser.add_argument(
        "--debug",
        type=str,
        help="Debug a single query",
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Analyze single question by ID",
    )
    parser.add_argument(
        "--failed-only",
        action="store_true",
        help="Show only failed queries",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of markdown",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of questions to analyze",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("FLORIDA TAX RAG - RETRIEVAL ANALYSIS")
    print("=" * 60)

    # Initialize clients
    print("\nInitializing clients...")
    try:
        weaviate = WeaviateClient()
        neo4j = Neo4jClient()
        embedder = VoyageEmbedder()

        retriever = HybridRetriever(
            weaviate_client=weaviate,
            neo4j_client=neo4j,
            embedder=embedder,
        )
        print("  Weaviate: Connected")
        print(f"  Neo4j: {'Connected' if neo4j.health_check() else 'Not available'}")
        print("  Embedder: Initialized")
    except Exception as e:
        print(f"Error initializing clients: {e}")
        return 1

    # Debug mode
    if args.debug:
        print(f"\nDebugging query: {args.debug}")
        result = debug_retrieval(retriever, args.debug, top_k=args.top_k)

        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            for method, results in result["methods"].items():
                print(f"\n{method.upper()} Results:")
                if isinstance(results, dict) and "error" in results:
                    print(f"  Error: {results['error']}")
                    continue
                for r in results[:10]:
                    marker = " *" if r.get("matches_expected") else ""
                    print(f"  [{r['rank']:2d}] {r['doc_id']} (score={r['score']:.4f}){marker}")
                    print(f"       {r['text_preview'][:70]}...")

        weaviate.close()
        neo4j.close()
        return 0

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    try:
        dataset = load_dataset(args.dataset)
        print(f"  Loaded {len(dataset.questions)} questions")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        weaviate.close()
        neo4j.close()
        return 1

    # Apply limit if specified
    if args.limit:
        dataset.questions = dataset.questions[: args.limit]
        print(f"  Limited to {args.limit} questions")

    # Create analyzer
    analyzer = RetrievalAnalyzer(retriever, dataset)

    # Single question mode
    if args.question:
        question = next(
            (q for q in dataset.questions if q.id == args.question),
            None,
        )
        if not question:
            print(f"Question {args.question} not found")
            weaviate.close()
            neo4j.close()
            return 1

        print(f"\nAnalyzing question: {question.id}")
        comparison = analyzer.compare_retrieval_methods(question, top_k=args.top_k)

        print(f"\nQuery: {question.question}")
        print(f"Expected: {', '.join(comparison.expected_citations)}")
        print(f"\nBest method: {comparison.best_method} (MRR={comparison.best_mrr:.4f})")

        for method_name, result in [
            ("Vector", comparison.vector_result),
            ("Keyword", comparison.keyword_result),
            ("Hybrid", comparison.hybrid_result),
            ("Graph", comparison.graph_result),
        ]:
            print(f"\n{method_name}:")
            print(f"  MRR: {result.reciprocal_rank:.4f}")
            print(f"  NDCG@10: {result.ndcg_at_10:.4f}")
            print(f"  Recall@10: {result.recall_at_10:.4f}")
            print(f"  Found: {result.found_expected}")

        weaviate.close()
        neo4j.close()
        return 0

    # Alpha tuning only
    if args.tune_alpha:
        alphas = [float(a) for a in args.alphas.split(",")]
        print(f"\nTuning alpha with values: {alphas}")

        def progress(alpha, i, n):
            print(f"  alpha={alpha:.2f}: {i}/{n} questions", end="\r")

        results = analyzer.tune_alpha(alphas=alphas, top_k=args.top_k, progress_callback=progress)

        print("\n\nAlpha Tuning Results:")
        print("-" * 60)
        print(f"{'Alpha':>8} | {'MRR':>8} | {'Recall@5':>10} | {'Recall@10':>10} | {'NDCG@10':>10}")
        print("-" * 60)
        for r in results:
            print(
                f"{r.alpha:>8.2f} | {r.mrr:>8.4f} | {r.recall_at_5:>10.4f} | "
                f"{r.recall_at_10:>10.4f} | {r.ndcg_at_10:>10.4f}"
            )

        best = max(results, key=lambda r: r.mrr)
        print("-" * 60)
        print(f"\nOptimal alpha: {best.alpha} (MRR={best.mrr:.4f})")

        weaviate.close()
        neo4j.close()
        return 0

    # Single method analysis
    if args.method:
        print(f"\nAnalyzing {args.method} retrieval...")

        all_results = []
        for i, question in enumerate(dataset.questions):
            print(f"  {i + 1}/{len(dataset.questions)}: {question.id}", end="\r")
            result = analyzer.analyze_query(question, top_k=args.top_k, method=args.method)
            all_results.append(result)

        print("\n\nResults:")
        print("-" * 50)

        n = len(all_results)
        avg_mrr = sum(r.reciprocal_rank for r in all_results) / n
        avg_recall_10 = sum(r.recall_at_10 for r in all_results) / n
        avg_ndcg_10 = sum(r.ndcg_at_10 for r in all_results) / n

        print(f"Method: {args.method}")
        print(f"MRR: {avg_mrr:.4f}")
        print(f"Recall@10: {avg_recall_10:.4f}")
        print(f"NDCG@10: {avg_ndcg_10:.4f}")

        # Show failed queries
        failed = [r for r in all_results if r.recall_at_20 == 0]
        if failed:
            print(f"\nFailed Queries ({len(failed)}):")
            for r in failed:
                print(f"  - {r.question_id}: {r.question_text[:50]}...")

        weaviate.close()
        neo4j.close()
        return 0

    # Full analysis
    print("\nRunning full retrieval analysis...")
    alphas = [float(a) for a in args.alphas.split(",")]

    def progress(step, i, n):
        print(f"  {step}: {i}/{n}", end="\r")

    report = analyzer.run_full_analysis(alphas=alphas, top_k=args.top_k, progress_callback=progress)

    print("\n")

    # Failed-only mode
    if args.failed_only:
        if not report.failed_queries:
            print("No failed queries!")
        else:
            print(f"Failed Queries ({len(report.failed_queries)}):")
            print("-" * 60)
            for q in report.failed_queries:
                print(f"\n{q.question_id}: {q.question_text}")
                print(f"  Expected: {', '.join(q.expected_citations)}")
                print(f"  Top-5 Retrieved: {', '.join(q.retrieved_doc_ids[:5])}")

        weaviate.close()
        neo4j.close()
        return 0

    # Output report
    if args.json:
        output = report.model_dump_json(indent=2)
    else:
        output = generate_retrieval_markdown_report(report)

    # Save to file
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        f.write(output)
    print(f"Report saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Best Method: {report.best_overall_method} (MRR={report.best_method_mrr:.4f})")
    print(f"Optimal Alpha: {report.optimal_alpha}")
    print(f"Failed Queries: {len(report.failed_queries)}")
    print()

    print("Method Comparison:")
    print("-" * 60)
    print(f"{'Method':>10} | {'MRR':>8} | {'Recall@5':>10} | {'Recall@10':>10}")
    print("-" * 60)
    for method, metrics in [
        ("Vector", report.vector_metrics),
        ("Keyword", report.keyword_metrics),
        ("Hybrid", report.hybrid_metrics),
        ("Graph", report.graph_metrics),
    ]:
        print(
            f"{method:>10} | {metrics.mrr:>8.4f} | {metrics.recall_at_5:>10.4f} | "
            f"{metrics.recall_at_10:>10.4f}"
        )
    print("=" * 60)

    # Cleanup
    weaviate.close()
    neo4j.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
