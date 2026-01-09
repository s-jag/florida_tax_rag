#!/usr/bin/env python3
"""Run evaluation pipeline on the Florida Tax RAG golden dataset.

Usage:
    python scripts/run_evaluation.py                    # Full evaluation with GPT-4 judge
    python scripts/run_evaluation.py --limit 5          # Quick check with 5 questions
    python scripts/run_evaluation.py --no-judge         # Skip LLM judge (faster)
    python scripts/run_evaluation.py --output reports/  # Custom output directory
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import get_settings
from src.agent.graph import create_tax_agent_graph
from src.evaluation import EvaluationRunner, LLMJudge, generate_markdown_report


async def main() -> int:
    """Run the evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="Run Florida Tax RAG evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="data/evaluation/golden_dataset.json",
        help="Path to evaluation dataset",
    )
    parser.add_argument(
        "--output",
        default="data/evaluation/reports",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit to N questions (for quick check)",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip LLM judge evaluation (faster, metrics only)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Timeout per question in seconds",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("FLORIDA TAX RAG EVALUATION")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output}")
    print(f"Questions: {args.limit or 'all'}")
    print(f"LLM Judge: {'disabled' if args.no_judge else 'enabled'}")
    print(f"Timeout: {args.timeout}s per question")
    print("=" * 60)
    print()

    settings = get_settings()

    # Initialize agent graph
    print("Initializing agent graph...")
    try:
        graph = create_tax_agent_graph()
        print("  Agent graph ready")
    except Exception as e:
        print(f"  ERROR: Failed to create agent graph: {e}")
        return 1

    # Initialize LLM judge
    judge = None
    if not args.no_judge:
        if settings.openai_api_key:
            print("Initializing GPT-4 judge...")
            judge = LLMJudge(api_key=settings.openai_api_key.get_secret_value())
            print("  GPT-4 judge ready")
        else:
            print("  WARNING: OPENAI_API_KEY not set, skipping LLM judge")
            print("  Set OPENAI_API_KEY or use --no-judge flag")

    # Create runner
    print("Loading evaluation dataset...")
    try:
        runner = EvaluationRunner(
            agent=graph,
            judge=judge,
            dataset_path=args.dataset,
        )
        print(f"  Loaded {len(runner.dataset.questions)} questions")
    except Exception as e:
        print(f"  ERROR: Failed to load dataset: {e}")
        return 1

    # Run evaluation
    print()
    print("Running evaluation...")
    print("-" * 60)

    try:
        report = await runner.run_all(
            limit=args.limit,
            timeout_per_question=args.timeout,
        )
    except Exception as e:
        print(f"ERROR: Evaluation failed: {e}")
        return 1

    print("-" * 60)
    print()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON report
    json_path = output_dir / f"eval_{report.run_id}.json"
    with open(json_path, "w") as f:
        json.dump(report.model_dump(), f, indent=2, default=str)
    print(f"Saved JSON report: {json_path}")

    # Save markdown report
    md_path = output_dir / f"eval_{report.run_id}.md"
    markdown = generate_markdown_report(report)
    with open(md_path, "w") as f:
        f.write(markdown)
    print(f"Saved Markdown report: {md_path}")

    # Print summary
    print()
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Run ID: {report.run_id}")
    print(f"  Questions: {report.successful_evaluations}/{report.total_questions}")
    if report.failed_evaluations > 0:
        print(f"  Failed: {report.failed_evaluations}")
    print()
    print(f"  Avg Overall Score: {report.avg_overall_score:.1f}/10")
    print(f"  Pass Rate: {report.pass_rate:.1%}")
    print()
    print(f"  Citation Precision: {report.avg_citation_precision:.1%}")
    print(f"  Citation Recall: {report.avg_citation_recall:.1%}")
    print(f"  Citation F1: {report.avg_citation_f1:.1%}")
    print(f"  Answer Contains: {report.avg_answer_contains:.1%}")
    print()
    print(f"  Avg Latency: {report.avg_latency_ms:,.0f}ms")
    print(f"  Total Hallucinations: {report.total_hallucinations}")

    # Print by difficulty breakdown
    if report.metrics_by_difficulty:
        print()
        print("By Difficulty:")
        for diff in ["easy", "medium", "hard"]:
            if diff in report.metrics_by_difficulty:
                m = report.metrics_by_difficulty[diff]
                print(
                    f"  {diff:8s}: {m.count} questions, "
                    f"score={m.avg_score:.1f}, "
                    f"pass={m.pass_rate:.0%}"
                )

    # Print by category breakdown
    if report.metrics_by_category:
        print()
        print("By Category:")
        for cat, m in sorted(report.metrics_by_category.items()):
            print(
                f"  {cat:15s}: {m.count} questions, score={m.avg_score:.1f}, pass={m.pass_rate:.0%}"
            )

    print("=" * 60)

    # Return exit code based on pass rate
    if report.pass_rate >= 0.7:
        print("\nResult: PASS (>=70% pass rate)")
        return 0
    else:
        print(f"\nResult: FAIL ({report.pass_rate:.0%} < 70% pass rate)")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
