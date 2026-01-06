#!/usr/bin/env python3
"""Comprehensive benchmark script for the Florida Tax RAG pipeline.

Runs diverse queries, measures per-stage timing, and generates a report.

Usage:
    python scripts/benchmark.py --num-queries 50
    python scripts/benchmark.py --output benchmark_results.md
    python scripts/benchmark.py --compare-before results_before.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Optional

import httpx

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Diverse test queries covering different query types
BENCHMARK_QUERIES = [
    # Simple factual queries
    {"query": "What is the Florida sales tax rate?", "category": "factual", "difficulty": "easy"},
    {"query": "What is the corporate income tax rate in Florida?", "category": "factual", "difficulty": "easy"},
    {"query": "What is the documentary stamp tax rate?", "category": "factual", "difficulty": "easy"},

    # Exemption queries
    {"query": "Are groceries exempt from sales tax in Florida?", "category": "exemption", "difficulty": "medium"},
    {"query": "Is clothing exempt from Florida sales tax?", "category": "exemption", "difficulty": "medium"},
    {"query": "What medical items are exempt from Florida sales tax?", "category": "exemption", "difficulty": "medium"},

    # Complex multi-step queries
    {"query": "How does a business determine if it has sales tax nexus in Florida?", "category": "nexus", "difficulty": "hard"},
    {"query": "What are the requirements for claiming a resale exemption certificate?", "category": "procedural", "difficulty": "hard"},
    {"query": "How is use tax calculated when purchasing from out-of-state vendors?", "category": "calculation", "difficulty": "hard"},

    # SaaS/Technology queries
    {"query": "How does Florida tax SaaS subscriptions?", "category": "technology", "difficulty": "hard"},
    {"query": "Are cloud computing services taxable in Florida?", "category": "technology", "difficulty": "hard"},

    # Rental/Lease queries
    {"query": "How is commercial real estate rental taxed in Florida?", "category": "rental", "difficulty": "medium"},
    {"query": "What is the tax treatment of equipment leases in Florida?", "category": "rental", "difficulty": "medium"},

    # Specific statute queries
    {"query": "What does Florida Statute 212.05 say about sales tax?", "category": "statute", "difficulty": "easy"},
    {"query": "What are the penalties for late sales tax filing under Chapter 212?", "category": "statute", "difficulty": "medium"},
]


@dataclass
class BenchmarkResult:
    """Result of a single benchmark query."""

    query: str
    category: str
    difficulty: str
    success: bool
    latency_ms: float
    stage_timings: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    citation_count: int = 0
    error: Optional[str] = None
    request_id: Optional[str] = None


@dataclass
class BenchmarkSummary:
    """Aggregated benchmark results."""

    results: list[BenchmarkResult] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    warmup_count: int = 0

    @property
    def successful_results(self) -> list[BenchmarkResult]:
        return [r for r in self.results if r.success]

    @property
    def latencies(self) -> list[float]:
        return [r.latency_ms for r in self.successful_results]

    @property
    def total_queries(self) -> int:
        return len(self.results)

    @property
    def success_count(self) -> int:
        return len(self.successful_results)

    @property
    def success_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return (self.success_count / self.total_queries) * 100

    @property
    def avg_latency_ms(self) -> float:
        return mean(self.latencies) if self.latencies else 0.0

    @property
    def p50_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_l = sorted(self.latencies)
        return sorted_l[len(sorted_l) // 2]

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_l = sorted(self.latencies)
        idx = int(len(sorted_l) * 0.95)
        return sorted_l[min(idx, len(sorted_l) - 1)]

    @property
    def p99_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_l = sorted(self.latencies)
        idx = int(len(sorted_l) * 0.99)
        return sorted_l[min(idx, len(sorted_l) - 1)]

    def get_stage_averages(self) -> dict[str, float]:
        """Calculate average time for each pipeline stage."""
        stage_times: dict[str, list[float]] = {}

        for result in self.successful_results:
            for stage, timing in result.stage_timings.items():
                if stage not in stage_times:
                    stage_times[stage] = []
                stage_times[stage].append(timing)

        return {
            stage: mean(times) for stage, times in stage_times.items()
        }

    def get_metrics_by_category(self) -> dict[str, dict[str, float]]:
        """Get metrics broken down by query category."""
        categories: dict[str, list[BenchmarkResult]] = {}

        for result in self.results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)

        metrics = {}
        for category, results in categories.items():
            latencies = [r.latency_ms for r in results if r.success]
            metrics[category] = {
                "count": len(results),
                "success_rate": (len(latencies) / len(results)) * 100 if results else 0,
                "avg_latency_ms": mean(latencies) if latencies else 0,
            }

        return metrics

    def get_metrics_by_difficulty(self) -> dict[str, dict[str, float]]:
        """Get metrics broken down by difficulty level."""
        difficulties: dict[str, list[BenchmarkResult]] = {}

        for result in self.results:
            if result.difficulty not in difficulties:
                difficulties[result.difficulty] = []
            difficulties[result.difficulty].append(result)

        metrics = {}
        for difficulty, results in difficulties.items():
            latencies = [r.latency_ms for r in results if r.success]
            metrics[difficulty] = {
                "count": len(results),
                "success_rate": (len(latencies) / len(results)) * 100 if results else 0,
                "avg_latency_ms": mean(latencies) if latencies else 0,
            }

        return metrics


async def run_query(
    client: httpx.AsyncClient,
    url: str,
    query_info: dict[str, str],
    timeout: float,
) -> BenchmarkResult:
    """Run a single benchmark query."""
    start_time = time.perf_counter()

    try:
        response = await client.post(
            url,
            json={
                "query": query_info["query"],
                "options": {"timeout_seconds": int(timeout), "include_reasoning": True},
            },
            timeout=timeout,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        if response.status_code == 200:
            data = response.json()
            return BenchmarkResult(
                query=query_info["query"],
                category=query_info["category"],
                difficulty=query_info["difficulty"],
                success=True,
                latency_ms=latency_ms,
                stage_timings=data.get("stage_timings", {}),
                confidence=data.get("confidence", 0.0),
                citation_count=len(data.get("citations", [])),
                request_id=data.get("request_id"),
            )
        else:
            error_msg = f"HTTP {response.status_code}"
            try:
                error_data = response.json()
                error_msg = error_data.get("error", error_msg)
            except Exception:
                pass

            return BenchmarkResult(
                query=query_info["query"],
                category=query_info["category"],
                difficulty=query_info["difficulty"],
                success=False,
                latency_ms=latency_ms,
                error=error_msg,
            )

    except httpx.TimeoutException:
        return BenchmarkResult(
            query=query_info["query"],
            category=query_info["category"],
            difficulty=query_info["difficulty"],
            success=False,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            error="TIMEOUT",
        )
    except Exception as e:
        return BenchmarkResult(
            query=query_info["query"],
            category=query_info["category"],
            difficulty=query_info["difficulty"],
            success=False,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            error=str(e),
        )


async def run_benchmark(
    url: str,
    num_queries: int,
    warmup_count: int,
    timeout: float,
) -> BenchmarkSummary:
    """Run the full benchmark suite."""
    summary = BenchmarkSummary()
    summary.warmup_count = warmup_count

    async with httpx.AsyncClient() as client:
        # Warmup phase
        if warmup_count > 0:
            print(f"\nRunning {warmup_count} warmup queries...")
            for i in range(warmup_count):
                query_info = BENCHMARK_QUERIES[i % len(BENCHMARK_QUERIES)]
                await run_query(client, url, query_info, timeout)
                print(f"  Warmup {i + 1}/{warmup_count} completed")

            print("Warmup complete. Waiting 2 seconds...")
            await asyncio.sleep(2)

        # Build query list
        queries_to_run = []
        for i in range(num_queries):
            queries_to_run.append(BENCHMARK_QUERIES[i % len(BENCHMARK_QUERIES)])

        # Run benchmark
        print(f"\nRunning {num_queries} benchmark queries...")
        summary.start_time = time.perf_counter()

        for i, query_info in enumerate(queries_to_run):
            result = await run_query(client, url, query_info, timeout)
            summary.results.append(result)

            status = "OK" if result.success else f"FAIL: {result.error}"
            print(f"  [{i + 1}/{num_queries}] {result.latency_ms:.0f}ms - {status}")

        summary.end_time = time.perf_counter()

    return summary


def generate_report(summary: BenchmarkSummary, comparison: Optional[dict] = None) -> str:
    """Generate a markdown report from benchmark results."""
    lines = [
        "# Florida Tax RAG Benchmark Report",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total Queries:** {summary.total_queries}",
        f"**Warmup Queries:** {summary.warmup_count}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Success Rate | {summary.success_rate:.1f}% |",
        f"| Average Latency | {summary.avg_latency_ms:.0f}ms |",
        f"| P50 Latency | {summary.p50_latency_ms:.0f}ms |",
        f"| P95 Latency | {summary.p95_latency_ms:.0f}ms |",
        f"| P99 Latency | {summary.p99_latency_ms:.0f}ms |",
    ]

    if summary.latencies:
        lines.extend([
            f"| Min Latency | {min(summary.latencies):.0f}ms |",
            f"| Max Latency | {max(summary.latencies):.0f}ms |",
        ])

    # Stage timing breakdown
    stage_avgs = summary.get_stage_averages()
    if stage_avgs:
        lines.extend([
            "",
            "## Pipeline Stage Timing (Average)",
            "",
            "| Stage | Average (ms) |",
            "|-------|-------------|",
        ])
        for stage, avg in sorted(stage_avgs.items(), key=lambda x: -x[1]):
            lines.append(f"| {stage} | {avg:.0f} |")

    # By category
    category_metrics = summary.get_metrics_by_category()
    if category_metrics:
        lines.extend([
            "",
            "## Results by Category",
            "",
            "| Category | Count | Success Rate | Avg Latency |",
            "|----------|-------|--------------|-------------|",
        ])
        for cat, metrics in sorted(category_metrics.items()):
            lines.append(
                f"| {cat} | {metrics['count']} | {metrics['success_rate']:.0f}% | {metrics['avg_latency_ms']:.0f}ms |"
            )

    # By difficulty
    diff_metrics = summary.get_metrics_by_difficulty()
    if diff_metrics:
        lines.extend([
            "",
            "## Results by Difficulty",
            "",
            "| Difficulty | Count | Success Rate | Avg Latency |",
            "|------------|-------|--------------|-------------|",
        ])
        for diff in ["easy", "medium", "hard"]:
            if diff in diff_metrics:
                metrics = diff_metrics[diff]
                lines.append(
                    f"| {diff} | {metrics['count']} | {metrics['success_rate']:.0f}% | {metrics['avg_latency_ms']:.0f}ms |"
                )

    # Comparison if provided
    if comparison:
        lines.extend([
            "",
            "## Comparison with Previous Run",
            "",
            "| Metric | Before | After | Change |",
            "|--------|--------|-------|--------|",
        ])

        before_p95 = comparison.get("p95_latency_ms", 0)
        after_p95 = summary.p95_latency_ms
        change_p95 = ((after_p95 - before_p95) / before_p95 * 100) if before_p95 else 0

        lines.append(
            f"| P95 Latency | {before_p95:.0f}ms | {after_p95:.0f}ms | {change_p95:+.1f}% |"
        )

        before_avg = comparison.get("avg_latency_ms", 0)
        after_avg = summary.avg_latency_ms
        change_avg = ((after_avg - before_avg) / before_avg * 100) if before_avg else 0

        lines.append(
            f"| Avg Latency | {before_avg:.0f}ms | {after_avg:.0f}ms | {change_avg:+.1f}% |"
        )

    return "\n".join(lines)


def save_results(summary: BenchmarkSummary, path: Path) -> None:
    """Save benchmark results to JSON for later comparison."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "total_queries": summary.total_queries,
        "success_rate": summary.success_rate,
        "avg_latency_ms": summary.avg_latency_ms,
        "p50_latency_ms": summary.p50_latency_ms,
        "p95_latency_ms": summary.p95_latency_ms,
        "p99_latency_ms": summary.p99_latency_ms,
        "stage_averages": summary.get_stage_averages(),
        "by_category": summary.get_metrics_by_category(),
        "by_difficulty": summary.get_metrics_by_difficulty(),
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


async def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the Florida Tax RAG pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000/api/v1/query",
        help="API endpoint URL",
    )
    parser.add_argument(
        "--num-queries",
        "-n",
        type=int,
        default=50,
        help="Number of queries to run",
    )
    parser.add_argument(
        "--warmup",
        "-w",
        type=int,
        default=5,
        help="Number of warmup queries",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=float,
        default=120.0,
        help="Query timeout in seconds",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output markdown report to file",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        help="Save results to JSON for comparison",
    )
    parser.add_argument(
        "--compare-before",
        type=str,
        help="Compare with previous results JSON",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("FLORIDA TAX RAG BENCHMARK")
    print("=" * 60)
    print(f"URL: {args.url}")
    print(f"Queries: {args.num_queries} + {args.warmup} warmup")
    print(f"Timeout: {args.timeout}s")

    # Load comparison if provided
    comparison = None
    if args.compare_before:
        try:
            with open(args.compare_before) as f:
                comparison = json.load(f)
            print(f"Comparing with: {args.compare_before}")
        except Exception as e:
            print(f"Warning: Could not load comparison file: {e}")

    # Run benchmark
    summary = await run_benchmark(
        url=args.url,
        num_queries=args.num_queries,
        warmup_count=args.warmup,
        timeout=args.timeout,
    )

    # Generate report
    report = generate_report(summary, comparison)

    print("\n" + report)

    # Save outputs
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(report)
        print(f"\nReport saved to: {output_path}")

    if args.save_json:
        json_path = Path(args.save_json)
        save_results(summary, json_path)
        print(f"Results saved to: {json_path}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
