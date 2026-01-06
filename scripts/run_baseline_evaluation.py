#!/usr/bin/env python3
"""Run frontier models on golden dataset for baseline comparison.

This script evaluates GPT-4 and Claude on the same questions as Margen,
WITHOUT any RAG augmentation, to establish a fair baseline for comparison.

The same LLM-as-judge evaluates all models to ensure consistency.

Usage:
    python scripts/run_baseline_evaluation.py --models gpt4 claude
    python scripts/run_baseline_evaluation.py --limit 5  # Quick test with 5 questions
    python scripts/run_baseline_evaluation.py --output data/evaluation/baseline_comparison.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

from src.evaluation.llm_judge import LLMJudge
from src.evaluation.models import EvalQuestion, EvalDataset
from src.evaluation.metrics import extract_citations_from_answer, citation_precision


@dataclass
class BaselineResult:
    """Result from running a single question through a frontier model."""

    question_id: str
    model: str
    answer: str
    latency_ms: int
    citations_found: list[str]
    citation_precision: float
    correctness: int
    completeness: int
    clarity: int
    citation_accuracy: int
    hallucinations: list[str]
    fabricated_citations: list[str]
    overall_score: int
    passed: bool
    reasoning: str


@dataclass
class BaselineModelSummary:
    """Aggregated results for a single model."""

    name: str
    model_id: str
    questions_evaluated: int
    hallucination_count: int
    hallucination_rate: float
    fabrication_count: int
    fabrication_rate: float
    avg_citation_precision: float
    avg_overall_score: float
    pass_rate: float
    avg_latency_ms: float
    results: list[dict]


BASELINE_PROMPT = """You are a Florida tax law expert. Answer the following question accurately, citing any relevant Florida Statutes (Chapter 212 for sales tax, Chapter 220 for corporate tax, etc.) or Florida Administrative Code rules (12A for revenue rules).

Be specific and cite exact statute section numbers when applicable.

Question: {question}

Provide your answer with specific legal citations."""


async def call_gpt4(
    client: AsyncOpenAI,
    question: str,
    model: str = "gpt-4-turbo-preview",
) -> tuple[str, int]:
    """Call GPT-4 and return response with latency."""
    start = time.perf_counter()

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert in Florida tax law."},
            {"role": "user", "content": BASELINE_PROMPT.format(question=question)},
        ],
        temperature=0.0,
        max_tokens=1000,
    )

    latency_ms = int((time.perf_counter() - start) * 1000)
    answer = response.choices[0].message.content or ""

    return answer, latency_ms


async def call_claude(
    client: AsyncAnthropic,
    question: str,
    model: str = "claude-sonnet-4-20250514",
) -> tuple[str, int]:
    """Call Claude and return response with latency."""
    start = time.perf_counter()

    response = await client.messages.create(
        model=model,
        max_tokens=1000,
        messages=[
            {"role": "user", "content": BASELINE_PROMPT.format(question=question)},
        ],
        system="You are an expert in Florida tax law.",
    )

    latency_ms = int((time.perf_counter() - start) * 1000)
    answer = response.content[0].text if response.content else ""

    return answer, latency_ms


def detect_fabricated_citations(
    citations: list[str],
    known_statutes: list[str],
    known_rules: list[str],
) -> list[str]:
    """Detect citations that appear fabricated (don't match expected patterns)."""
    fabricated = []

    # Known valid statute chapters
    valid_chapters = {"212", "213", "215", "220", "196"}

    for citation in citations:
        citation_lower = citation.lower()

        # Check if it looks like a Florida statute
        if any(ch in citation for ch in valid_chapters):
            # It references a valid chapter, consider it valid
            continue

        # Check if it's a rule reference (12A, 12-26, etc.)
        if "12a" in citation_lower or "12-" in citation_lower:
            continue

        # If it has numbers and periods (looks like a citation) but isn't recognizable
        if any(c.isdigit() for c in citation) and "." in citation:
            # Could be fabricated
            fabricated.append(citation)

    return fabricated


async def evaluate_model(
    model_name: str,
    model_id: str,
    questions: list[EvalQuestion],
    call_fn,
    client,
    judge: LLMJudge,
    semaphore: asyncio.Semaphore,
) -> BaselineModelSummary:
    """Evaluate a single model on all questions."""

    results: list[BaselineResult] = []

    async def process_question(q: EvalQuestion) -> Optional[BaselineResult]:
        async with semaphore:
            try:
                print(f"  [{model_name}] Evaluating: {q.id} - {q.question[:50]}...")

                # Call the model
                answer, latency_ms = await call_fn(client, q.question)

                # Extract citations from answer
                citations = extract_citations_from_answer(answer)

                # Calculate citation precision
                precision = citation_precision(
                    citations,
                    q.expected_statutes,
                    q.expected_rules,
                )

                # Detect fabricated citations
                fabricated = detect_fabricated_citations(
                    citations,
                    q.expected_statutes,
                    q.expected_rules,
                )

                # Use LLM judge to evaluate
                judgment = await judge.judge_answer(q, answer)

                passed = judgment.overall_score >= 7 and len(judgment.hallucinations) == 0

                return BaselineResult(
                    question_id=q.id,
                    model=model_name,
                    answer=answer,
                    latency_ms=latency_ms,
                    citations_found=citations,
                    citation_precision=precision,
                    correctness=judgment.correctness,
                    completeness=judgment.completeness,
                    clarity=judgment.clarity,
                    citation_accuracy=judgment.citation_accuracy,
                    hallucinations=judgment.hallucinations,
                    fabricated_citations=fabricated,
                    overall_score=judgment.overall_score,
                    passed=passed,
                    reasoning=judgment.reasoning,
                )

            except Exception as e:
                print(f"    -> ERROR: {str(e)[:100]}")
                return None

    # Process all questions concurrently
    tasks = [process_question(q) for q in questions]
    raw_results = await asyncio.gather(*tasks)

    # Filter out None results
    results = [r for r in raw_results if r is not None]

    # Calculate aggregates
    total = len(results)
    if total == 0:
        return BaselineModelSummary(
            name=model_name,
            model_id=model_id,
            questions_evaluated=0,
            hallucination_count=0,
            hallucination_rate=0.0,
            fabrication_count=0,
            fabrication_rate=0.0,
            avg_citation_precision=0.0,
            avg_overall_score=0.0,
            pass_rate=0.0,
            avg_latency_ms=0.0,
            results=[],
        )

    hallucination_count = sum(len(r.hallucinations) for r in results)
    queries_with_hallucinations = sum(1 for r in results if r.hallucinations)
    fabrication_count = sum(len(r.fabricated_citations) for r in results)
    queries_with_fabrications = sum(1 for r in results if r.fabricated_citations)

    return BaselineModelSummary(
        name=model_name,
        model_id=model_id,
        questions_evaluated=total,
        hallucination_count=hallucination_count,
        hallucination_rate=queries_with_hallucinations / total,
        fabrication_count=fabrication_count,
        fabrication_rate=queries_with_fabrications / total,
        avg_citation_precision=sum(r.citation_precision for r in results) / total,
        avg_overall_score=sum(r.overall_score for r in results) / total,
        pass_rate=sum(1 for r in results if r.passed) / total,
        avg_latency_ms=sum(r.latency_ms for r in results) / total,
        results=[asdict(r) for r in results],
    )


async def run_baseline_evaluation(
    dataset_path: str,
    output_path: str,
    models: list[str],
    limit: Optional[int] = None,
    max_concurrent: int = 3,
) -> dict:
    """Run baseline evaluation on specified models."""

    # Load dataset
    with open(dataset_path) as f:
        data = json.load(f)
    dataset = EvalDataset(**data)

    questions = dataset.questions[:limit] if limit else dataset.questions
    print(f"\nEvaluating {len(questions)} questions against baseline models: {models}")

    # Initialize clients
    openai_api_key = os.getenv("OPENAI_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not set")

    openai_client = AsyncOpenAI(api_key=openai_api_key)
    anthropic_client = AsyncAnthropic(api_key=anthropic_api_key) if anthropic_api_key else None

    # Initialize judge (uses GPT-4)
    judge = LLMJudge(api_key=openai_api_key)

    semaphore = asyncio.Semaphore(max_concurrent)

    results = {
        "evaluation_date": datetime.now().isoformat(),
        "dataset_version": dataset.metadata.get("version", "1.0.0"),
        "questions_evaluated": len(questions),
        "baselines": {},
    }

    # Evaluate each model
    for model in models:
        if model == "gpt4":
            if not openai_api_key:
                print("  Skipping GPT-4: No API key")
                continue
            print(f"\n--- Evaluating GPT-4 Turbo ---")
            summary = await evaluate_model(
                "GPT-4 Turbo (No RAG)",
                "gpt-4-turbo-preview",
                questions,
                call_gpt4,
                openai_client,
                judge,
                semaphore,
            )
            results["baselines"]["gpt4_vanilla"] = asdict(summary)

        elif model == "claude":
            if not anthropic_client:
                print("  Skipping Claude: No API key")
                continue
            print(f"\n--- Evaluating Claude Sonnet 4 ---")
            summary = await evaluate_model(
                "Claude Sonnet 4 (No RAG)",
                "claude-sonnet-4-20250514",
                questions,
                call_claude,
                anthropic_client,
                judge,
                semaphore,
            )
            results["baselines"]["claude_vanilla"] = asdict(summary)

    # Save results
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to: {output}")

    # Print summary
    print("\n" + "=" * 60)
    print("BASELINE EVALUATION SUMMARY")
    print("=" * 60)

    for model_key, summary in results["baselines"].items():
        print(f"\n{summary['name']}:")
        print(f"  Questions: {summary['questions_evaluated']}")
        print(f"  Hallucination Rate: {summary['hallucination_rate']:.1%}")
        print(f"  Fabrication Rate: {summary['fabrication_rate']:.1%}")
        print(f"  Citation Precision: {summary['avg_citation_precision']:.1%}")
        print(f"  Pass Rate: {summary['pass_rate']:.1%}")
        print(f"  Avg Latency: {summary['avg_latency_ms']:.0f}ms")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run frontier models on golden dataset for baseline comparison"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/evaluation/golden_dataset.json",
        help="Path to golden dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/evaluation/baseline_comparison.json",
        help="Output path for results",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["gpt4", "claude"],
        choices=["gpt4", "claude"],
        help="Models to evaluate",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions (for testing)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Maximum concurrent API calls per model",
    )

    args = parser.parse_args()

    asyncio.run(run_baseline_evaluation(
        dataset_path=args.dataset,
        output_path=args.output,
        models=args.models,
        limit=args.limit,
        max_concurrent=args.max_concurrent,
    ))


if __name__ == "__main__":
    main()
