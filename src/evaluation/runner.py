"""Evaluation runner for the Florida Tax RAG system."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Optional

from .authority_metrics import compute_authority_metrics, aggregate_authority_metrics
from .correction_metrics import CorrectionTracker, CorrectionAction
from .llm_judge import LLMJudge
from .metrics import (
    answer_contains_expected,
    citation_precision,
    citation_recall,
    extract_citations_from_answer,
    f1_score,
)
from .models import (
    AuthorityMetricsResult,
    CorrectionMetricsResult,
    EvalDataset,
    EvalQuestion,
    EvalResult,
    FaithfulnessMetricsResult,
)
from .report import (
    AuthorityAnalysis,
    CategoryMetrics,
    CorrectionAnalysis,
    DifficultyMetrics,
    FaithfulnessAnalysis,
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
        enable_authority_metrics: bool = True,
        enable_faithfulness_check: bool = False,
        enable_correction_tracking: bool = True,
    ):
        """Initialize the evaluation runner.

        Args:
            agent: Compiled LangGraph agent with ainvoke method
            judge: Optional LLMJudge for GPT-4 evaluation
            dataset_path: Path to golden dataset JSON file
            enable_authority_metrics: Track authority-weighted ranking metrics
            enable_faithfulness_check: Run LLM faithfulness checking (slower)
            enable_correction_tracking: Track self-correction metrics
        """
        self.agent = agent
        self.judge = judge
        self.dataset = self._load_dataset(dataset_path)
        self._questions_map = {q.id: q for q in self.dataset.questions}

        # Feature flags
        self.enable_authority_metrics = enable_authority_metrics
        self.enable_faithfulness_check = enable_faithfulness_check
        self.enable_correction_tracking = enable_correction_tracking

        # Correction tracker for aggregating across queries
        self.correction_tracker = CorrectionTracker() if enable_correction_tracking else None

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

        # Compute authority metrics from retrieved chunks
        authority_metrics_result = None
        if self.enable_authority_metrics:
            chunks = result.get("retrieved_chunks", [])
            doc_types = [
                chunk.get("doc_type", "unknown")
                for chunk in chunks
            ]
            if doc_types:
                auth_metrics = compute_authority_metrics(doc_types)
                authority_metrics_result = AuthorityMetricsResult(
                    authority_ndcg_at_5=auth_metrics.authority_ndcg_at_5,
                    authority_ndcg_at_10=auth_metrics.authority_ndcg_at_10,
                    hierarchy_alignment_score=auth_metrics.hierarchy_alignment_score,
                    primary_authority_rate_at_5=auth_metrics.primary_authority_rate_at_5,
                    doc_types_retrieved=doc_types,
                )

        # Track self-correction metrics
        correction_metrics_result = None
        if self.enable_correction_tracking and self.correction_tracker:
            self.correction_tracker.record_from_state(question.id, result)

            # Extract for this single result
            validation_result = result.get("validation_result", {})
            correction_result = result.get("correction_result", {})
            hallucinations = validation_result.get("hallucinations", [])

            if result.get("validation_passed", True):
                action = "none"
            elif correction_result.get("corrections_made"):
                action = "corrected"
            elif result.get("regeneration_count", 0) > 0:
                action = "regenerated"
            else:
                action = "failed"

            correction_metrics_result = CorrectionMetricsResult(
                action_taken=action,
                issues_detected=len(hallucinations),
                issues_corrected=len(correction_result.get("corrections_made", [])),
                severity_scores=[h.get("severity", 0.5) for h in hallucinations],
                hallucination_types=[h.get("type", "unknown") for h in hallucinations],
                confidence_before=result.get("original_confidence", result.get("confidence", 0.0)),
                confidence_after=result.get("confidence", 0.0),
            )

        return EvalResult(
            question_id=question.id,
            generated_answer=answer,
            generated_citations=generated_citations,
            citation_precision=precision,
            citation_recall=recall,
            answer_contains_score=contains_score,
            authority_metrics=authority_metrics_result,
            correction_metrics=correction_metrics_result,
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
                    faithfulness_score=result.faithfulness_metrics.faithfulness_score if result.faithfulness_metrics else None,
                    correction_action=result.correction_metrics.action_taken if result.correction_metrics else None,
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

        # Aggregate authority metrics
        authority_analysis = None
        if self.enable_authority_metrics:
            auth_results = [r for r in results if r.authority_metrics]
            if auth_results:
                from .authority_metrics import AuthorityMetrics as AuthMetrics
                auth_metrics_list = []
                total_doc_type_dist: dict[str, int] = {}
                total_by_rank: dict[int, dict[str, int]] = {}

                for r in auth_results:
                    am = r.authority_metrics
                    # Aggregate doc type distribution
                    for dt in am.doc_types_retrieved:
                        total_doc_type_dist[dt] = total_doc_type_dist.get(dt, 0) + 1

                authority_analysis = AuthorityAnalysis(
                    avg_authority_ndcg_at_5=sum(r.authority_metrics.authority_ndcg_at_5 for r in auth_results) / len(auth_results),
                    avg_authority_ndcg_at_10=sum(r.authority_metrics.authority_ndcg_at_10 for r in auth_results) / len(auth_results),
                    avg_hierarchy_alignment=sum(r.authority_metrics.hierarchy_alignment_score for r in auth_results) / len(auth_results),
                    avg_primary_authority_rate=sum(r.authority_metrics.primary_authority_rate_at_5 for r in auth_results) / len(auth_results),
                    doc_type_distribution=total_doc_type_dist,
                    authority_by_rank=total_by_rank,
                )

        # Aggregate correction metrics
        correction_analysis = None
        if self.enable_correction_tracking and self.correction_tracker:
            corr_metrics = self.correction_tracker.compute_metrics()
            correction_analysis = CorrectionAnalysis(
                total_queries=corr_metrics.total_queries,
                queries_with_issues=corr_metrics.queries_with_issues,
                queries_corrected=corr_metrics.queries_corrected,
                queries_regenerated=corr_metrics.queries_regenerated,
                queries_failed=corr_metrics.queries_failed,
                intervention_rate=corr_metrics.intervention_rate,
                correction_success_rate=corr_metrics.correction_success_rate,
                total_issues_detected=corr_metrics.total_issues_detected,
                total_issues_corrected=corr_metrics.total_issues_corrected,
                issues_by_type=corr_metrics.issues_by_type,
                avg_severity=corr_metrics.avg_severity,
            )

        # Calculate faithfulness metrics
        faithfulness_analysis = None
        faith_results = [r for r in results if r.faithfulness_metrics]
        if faith_results:
            faithfulness_analysis = FaithfulnessAnalysis(
                total_claims_checked=sum(r.faithfulness_metrics.total_claims for r in faith_results),
                total_supported=sum(r.faithfulness_metrics.supported_claims for r in faith_results),
                total_partially_supported=sum(r.faithfulness_metrics.partially_supported_claims for r in faith_results),
                total_unsupported=sum(r.faithfulness_metrics.unsupported_claims for r in faith_results),
                total_contradicted=sum(r.faithfulness_metrics.contradicted_claims for r in faith_results),
                avg_faithfulness_score=sum(r.faithfulness_metrics.faithfulness_score for r in faith_results) / len(faith_results),
            )

        # Calculate aggregate faithfulness score
        avg_faithfulness = 1.0
        if faith_results:
            avg_faithfulness = sum(r.faithfulness_metrics.faithfulness_score for r in faith_results) / len(faith_results)

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
            avg_faithfulness_score=avg_faithfulness,
            metrics_by_category=metrics_by_category,
            metrics_by_difficulty=metrics_by_difficulty,
            authority_analysis=authority_analysis,
            faithfulness_analysis=faithfulness_analysis,
            correction_analysis=correction_analysis,
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
