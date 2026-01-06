"""Evaluation runner for the Florida Tax RAG system."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

from .llm_judge import LLMJudge
from .metrics import (
    answer_contains_expected,
    citation_precision,
    citation_recall,
    extract_citations_from_answer,
    f1_score,
)
from .models import EvalDataset, EvalQuestion, EvalResult
from .report import (
    CategoryMetrics,
    DifficultyMetrics,
    FullEvaluationReport,
    QuestionSummary,
)


class EvaluationRunner:
    """Orchestrates evaluation of the RAG system against a golden dataset."""

    def __init__(
        self,
        agent: Any,
        judge: Optional[LLMJudge],
        dataset_path: str,
    ):
        """Initialize the evaluation runner.

        Args:
            agent: Compiled LangGraph agent with ainvoke method
            judge: Optional LLMJudge for GPT-4 evaluation
            dataset_path: Path to golden dataset JSON file
        """
        self.agent = agent
        self.judge = judge
        self.dataset = self._load_dataset(dataset_path)
        self._questions_map = {q.id: q for q in self.dataset.questions}

    def _load_dataset(self, path: str) -> EvalDataset:
        """Load evaluation dataset from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return EvalDataset(**data)

    async def run_single(
        self,
        question: EvalQuestion,
        timeout: Optional[int] = None,
    ) -> EvalResult:
        """Run evaluation on a single question.

        Args:
            question: The evaluation question
            timeout: Optional timeout in seconds

        Returns:
            EvalResult with metrics and optional judgment
        """
        start = time.perf_counter()

        # Build initial state for agent
        initial_state = {
            "original_query": question.question,
            "retrieved_chunks": [],
            "errors": [],
            "reasoning_steps": [],
        }

        # Invoke the agent
        result = await self.agent.ainvoke(initial_state)

        latency_ms = int((time.perf_counter() - start) * 1000)

        # Extract answer
        answer = result.get("final_answer") or ""

        # Extract citations from the answer text
        generated_citations = extract_citations_from_answer(answer)

        # Calculate citation metrics
        precision = citation_precision(
            generated_citations,
            question.expected_statutes,
            question.expected_rules,
        )
        recall = citation_recall(
            generated_citations,
            question.expected_statutes,
            question.expected_rules,
        )

        # Calculate answer contains score
        contains_score = answer_contains_expected(
            answer,
            question.expected_answer_contains,
        )

        # Get LLM judgment if judge is available
        judgment = None
        if self.judge and answer:
            judgment = await self.judge.judge_answer(question, answer)

        return EvalResult(
            question_id=question.id,
            generated_answer=answer,
            generated_citations=generated_citations,
            citation_precision=precision,
            citation_recall=recall,
            answer_contains_score=contains_score,
            judgment=judgment,
            latency_ms=latency_ms,
        )

    async def run_all(
        self,
        limit: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        timeout_per_question: int = 120,
    ) -> FullEvaluationReport:
        """Run evaluation on all questions in the dataset.

        Args:
            limit: Optional limit on number of questions to evaluate
            progress_callback: Optional callback(current, total) for progress updates
            timeout_per_question: Timeout per question in seconds

        Returns:
            FullEvaluationReport with aggregated metrics
        """
        questions = self.dataset.questions[:limit] if limit else self.dataset.questions

        results: list[EvalResult] = []
        failed: list[str] = []
        errors: list[str] = []

        for i, question in enumerate(questions):
            try:
                print(f"  [{i+1}/{len(questions)}] Evaluating: {question.id} - {question.question[:50]}...")
                result = await self.run_single(question, timeout=timeout_per_question)
                results.append(result)

                # Print quick status
                if result.judgment:
                    status = "PASS" if result.judgment.passed else "FAIL"
                    print(f"    -> Score: {result.judgment.overall_score}/10 [{status}]")
                else:
                    print(f"    -> Precision: {result.citation_precision:.0%}, Recall: {result.citation_recall:.0%}")

            except Exception as e:
                failed.append(question.id)
                errors.append(f"{question.id}: {str(e)}")
                print(f"    -> ERROR: {str(e)[:100]}")

            if progress_callback:
                progress_callback(i + 1, len(questions))

        return self._compile_report(questions, results, failed, errors)

    def _compile_report(
        self,
        questions: list[EvalQuestion],
        results: list[EvalResult],
        failed: list[str],
        errors: list[str],
    ) -> FullEvaluationReport:
        """Compile evaluation results into a full report.

        Args:
            questions: List of questions that were evaluated
            results: List of evaluation results
            failed: List of failed question IDs
            errors: List of error messages

        Returns:
            FullEvaluationReport with all metrics
        """
        # Build question summaries
        summaries = []
        for result in results:
            question = self._questions_map.get(result.question_id)
            if question:
                summary = QuestionSummary(
                    question_id=result.question_id,
                    question_text=question.question,
                    category=question.category.value,
                    difficulty=question.difficulty.value,
                    score=result.judgment.overall_score if result.judgment else None,
                    passed=result.judgment.passed if result.judgment else False,
                    hallucinations=result.judgment.hallucinations if result.judgment else [],
                    missing_concepts=result.judgment.missing_concepts if result.judgment else [],
                    latency_ms=result.latency_ms,
                )
                summaries.append(summary)

        # Calculate aggregate metrics
        total_questions = len(questions)
        successful = len(results)
        failed_count = len(failed)

        avg_precision = sum(r.citation_precision for r in results) / max(len(results), 1)
        avg_recall = sum(r.citation_recall for r in results) / max(len(results), 1)
        avg_f1 = f1_score(avg_precision, avg_recall)
        avg_contains = sum(r.answer_contains_score for r in results) / max(len(results), 1)
        avg_latency = sum(r.latency_ms for r in results) / max(len(results), 1)

        # Calculate judgment-based metrics
        judged = [r for r in results if r.judgment]
        avg_score = sum(r.judgment.overall_score for r in judged) / max(len(judged), 1)
        passed_count = sum(1 for r in judged if r.judgment.passed)
        pass_rate = passed_count / max(len(judged), 1)
        total_hallucinations = sum(len(r.judgment.hallucinations) for r in judged)

        # Calculate metrics by category
        metrics_by_category = self._calculate_category_metrics(results)

        # Calculate metrics by difficulty
        metrics_by_difficulty = self._calculate_difficulty_metrics(results)

        return FullEvaluationReport(
            dataset_version=self.dataset.metadata.get("version", "1.0.0"),
            total_questions=total_questions,
            successful_evaluations=successful,
            failed_evaluations=failed_count,
            avg_citation_precision=avg_precision,
            avg_citation_recall=avg_recall,
            avg_citation_f1=avg_f1,
            avg_answer_contains=avg_contains,
            avg_overall_score=avg_score,
            avg_latency_ms=avg_latency,
            total_hallucinations=total_hallucinations,
            pass_rate=pass_rate,
            metrics_by_category=metrics_by_category,
            metrics_by_difficulty=metrics_by_difficulty,
            results=results,
            question_summaries=summaries,
            failed_questions=failed,
            errors=errors,
        )

    def _calculate_category_metrics(
        self,
        results: list[EvalResult],
    ) -> dict[str, CategoryMetrics]:
        """Calculate metrics grouped by category."""
        by_category: dict[str, list[EvalResult]] = defaultdict(list)

        for result in results:
            question = self._questions_map.get(result.question_id)
            if question:
                by_category[question.category.value].append(result)

        metrics = {}
        for cat, cat_results in by_category.items():
            judged = [r for r in cat_results if r.judgment]

            metrics[cat] = CategoryMetrics(
                category=cat,
                count=len(cat_results),
                avg_precision=sum(r.citation_precision for r in cat_results) / max(len(cat_results), 1),
                avg_recall=sum(r.citation_recall for r in cat_results) / max(len(cat_results), 1),
                avg_score=sum(r.judgment.overall_score for r in judged) / max(len(judged), 1),
                pass_rate=sum(1 for r in judged if r.judgment.passed) / max(len(judged), 1),
                total_hallucinations=sum(len(r.judgment.hallucinations) for r in judged),
            )

        return metrics

    def _calculate_difficulty_metrics(
        self,
        results: list[EvalResult],
    ) -> dict[str, DifficultyMetrics]:
        """Calculate metrics grouped by difficulty."""
        by_difficulty: dict[str, list[EvalResult]] = defaultdict(list)

        for result in results:
            question = self._questions_map.get(result.question_id)
            if question:
                by_difficulty[question.difficulty.value].append(result)

        metrics = {}
        for diff, diff_results in by_difficulty.items():
            judged = [r for r in diff_results if r.judgment]

            metrics[diff] = DifficultyMetrics(
                difficulty=diff,
                count=len(diff_results),
                avg_precision=sum(r.citation_precision for r in diff_results) / max(len(diff_results), 1),
                avg_recall=sum(r.citation_recall for r in diff_results) / max(len(diff_results), 1),
                avg_score=sum(r.judgment.overall_score for r in judged) / max(len(judged), 1),
                pass_rate=sum(1 for r in judged if r.judgment.passed) / max(len(judged), 1),
                avg_latency_ms=sum(r.latency_ms for r in diff_results) / max(len(diff_results), 1),
            )

        return metrics
