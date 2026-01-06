"""Extended evaluation report models and markdown generation."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from .models import Category, Difficulty, EvalResult


class CategoryMetrics(BaseModel):
    """Metrics aggregated by category."""

    category: str = Field(..., description="Category name")
    count: int = Field(default=0, description="Number of questions")
    avg_precision: float = Field(default=0.0, description="Average citation precision")
    avg_recall: float = Field(default=0.0, description="Average citation recall")
    avg_score: float = Field(default=0.0, description="Average overall score")
    pass_rate: float = Field(default=0.0, description="Percentage of passed questions")
    total_hallucinations: int = Field(default=0, description="Total hallucinations")
    avg_faithfulness: float = Field(default=1.0, description="Average faithfulness score")


class DifficultyMetrics(BaseModel):
    """Metrics aggregated by difficulty level."""

    difficulty: str = Field(..., description="Difficulty level")
    count: int = Field(default=0, description="Number of questions")
    avg_precision: float = Field(default=0.0, description="Average citation precision")
    avg_recall: float = Field(default=0.0, description="Average citation recall")
    avg_score: float = Field(default=0.0, description="Average overall score")
    pass_rate: float = Field(default=0.0, description="Percentage of passed questions")
    avg_latency_ms: float = Field(default=0.0, description="Average latency in ms")


class QuestionSummary(BaseModel):
    """Summary of a single question result for reporting."""

    question_id: str
    question_text: str
    category: str
    difficulty: str
    score: Optional[int] = None
    passed: bool = False
    hallucinations: list[str] = Field(default_factory=list)
    missing_concepts: list[str] = Field(default_factory=list)
    latency_ms: int = 0
    faithfulness_score: Optional[float] = None
    correction_action: Optional[str] = None


class AuthorityAnalysis(BaseModel):
    """Aggregated authority-aware metrics."""

    avg_authority_ndcg_at_5: float = Field(default=0.0)
    avg_authority_ndcg_at_10: float = Field(default=0.0)
    avg_hierarchy_alignment: float = Field(default=0.0)
    avg_primary_authority_rate: float = Field(default=0.0)
    doc_type_distribution: dict[str, int] = Field(default_factory=dict)
    authority_by_rank: dict[int, dict[str, int]] = Field(default_factory=dict)


class FaithfulnessAnalysis(BaseModel):
    """Aggregated faithfulness metrics."""

    total_claims_checked: int = Field(default=0)
    total_supported: int = Field(default=0)
    total_partially_supported: int = Field(default=0)
    total_unsupported: int = Field(default=0)
    total_contradicted: int = Field(default=0)
    avg_faithfulness_score: float = Field(default=1.0)
    fabrication_rate: float = Field(default=0.0, description="% of citations that don't exist")


class CorrectionAnalysis(BaseModel):
    """Aggregated self-correction metrics."""

    total_queries: int = Field(default=0)
    queries_with_issues: int = Field(default=0)
    queries_corrected: int = Field(default=0)
    queries_regenerated: int = Field(default=0)
    queries_failed: int = Field(default=0)
    intervention_rate: float = Field(default=0.0)
    correction_success_rate: float = Field(default=1.0)
    total_issues_detected: int = Field(default=0)
    total_issues_corrected: int = Field(default=0)
    issues_by_type: dict[str, int] = Field(default_factory=dict)
    avg_severity: float = Field(default=0.0)


class FullEvaluationReport(BaseModel):
    """Complete evaluation report with breakdowns."""

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    dataset_version: str = Field(default="1.0.0")

    # Aggregate metrics
    total_questions: int = Field(default=0)
    successful_evaluations: int = Field(default=0)
    failed_evaluations: int = Field(default=0)

    avg_citation_precision: float = Field(default=0.0)
    avg_citation_recall: float = Field(default=0.0)
    avg_citation_f1: float = Field(default=0.0)
    avg_answer_contains: float = Field(default=0.0)
    avg_overall_score: float = Field(default=0.0)
    avg_latency_ms: float = Field(default=0.0)
    total_hallucinations: int = Field(default=0)
    pass_rate: float = Field(default=0.0)

    # New aggregate metrics
    avg_faithfulness_score: float = Field(default=1.0)
    fabrication_rate: float = Field(default=0.0)

    # Breakdowns
    metrics_by_category: dict[str, CategoryMetrics] = Field(default_factory=dict)
    metrics_by_difficulty: dict[str, DifficultyMetrics] = Field(default_factory=dict)

    # New analysis sections
    authority_analysis: Optional[AuthorityAnalysis] = Field(default=None)
    faithfulness_analysis: Optional[FaithfulnessAnalysis] = Field(default=None)
    correction_analysis: Optional[CorrectionAnalysis] = Field(default=None)

    # Individual results
    results: list[EvalResult] = Field(default_factory=list)

    # Question summaries for reporting
    question_summaries: list[QuestionSummary] = Field(default_factory=list)

    # Failures
    failed_questions: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def generate_markdown_report(
    report: FullEvaluationReport,
    questions_map: Optional[dict] = None,
) -> str:
    """Generate human-readable markdown report.

    Args:
        report: The full evaluation report
        questions_map: Optional mapping of question_id to EvalQuestion for details

    Returns:
        Formatted markdown string
    """
    lines = []

    # Header
    lines.append("# Florida Tax RAG Evaluation Report")
    lines.append("")
    lines.append(f"**Run ID:** {report.run_id}")
    lines.append(f"**Date:** {report.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"**Dataset Version:** {report.dataset_version}")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Questions Evaluated | {report.successful_evaluations}/{report.total_questions} |")
    lines.append(f"| Pass Rate | {report.pass_rate:.1%} |")
    lines.append(f"| Avg Overall Score | {report.avg_overall_score:.1f}/10 |")
    lines.append(f"| Citation Precision | {report.avg_citation_precision:.1%} |")
    lines.append(f"| Citation Recall | {report.avg_citation_recall:.1%} |")
    lines.append(f"| Citation F1 | {report.avg_citation_f1:.1%} |")
    lines.append(f"| Faithfulness Score | {report.avg_faithfulness_score:.1%} |")
    lines.append(f"| Fabrication Rate | {report.fabrication_rate:.1%} |")
    lines.append(f"| Answer Contains Score | {report.avg_answer_contains:.1%} |")
    lines.append(f"| Avg Latency | {report.avg_latency_ms:,.0f}ms |")
    lines.append(f"| Total Hallucinations | {report.total_hallucinations} |")
    lines.append("")

    # Authority Analysis
    if report.authority_analysis:
        lines.append("## Authority Awareness Analysis")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Authority-Weighted NDCG@5 | {report.authority_analysis.avg_authority_ndcg_at_5:.3f} |")
        lines.append(f"| Authority-Weighted NDCG@10 | {report.authority_analysis.avg_authority_ndcg_at_10:.3f} |")
        lines.append(f"| Hierarchy Alignment | {report.authority_analysis.avg_hierarchy_alignment:.1%} |")
        lines.append(f"| Primary Authority Rate | {report.authority_analysis.avg_primary_authority_rate:.1%} |")
        lines.append("")

        if report.authority_analysis.doc_type_distribution:
            lines.append("**Document Type Distribution:**")
            lines.append("")
            for doc_type, count in sorted(report.authority_analysis.doc_type_distribution.items()):
                lines.append(f"- {doc_type}: {count}")
            lines.append("")

    # Faithfulness Analysis
    if report.faithfulness_analysis:
        lines.append("## Faithfulness Analysis")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total Claims Checked | {report.faithfulness_analysis.total_claims_checked} |")
        lines.append(f"| Supported | {report.faithfulness_analysis.total_supported} |")
        lines.append(f"| Partially Supported | {report.faithfulness_analysis.total_partially_supported} |")
        lines.append(f"| Unsupported | {report.faithfulness_analysis.total_unsupported} |")
        lines.append(f"| Contradicted | {report.faithfulness_analysis.total_contradicted} |")
        lines.append(f"| Avg Faithfulness Score | {report.faithfulness_analysis.avg_faithfulness_score:.1%} |")
        lines.append(f"| Fabrication Rate | {report.faithfulness_analysis.fabrication_rate:.1%} |")
        lines.append("")

    # Self-Correction Analysis
    if report.correction_analysis:
        lines.append("## Self-Correction Analysis")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Total Queries | {report.correction_analysis.total_queries} |")
        lines.append(f"| Queries with Issues | {report.correction_analysis.queries_with_issues} |")
        lines.append(f"| Auto-Corrected | {report.correction_analysis.queries_corrected} |")
        lines.append(f"| Regenerated | {report.correction_analysis.queries_regenerated} |")
        lines.append(f"| Failed Corrections | {report.correction_analysis.queries_failed} |")
        lines.append(f"| Intervention Rate | {report.correction_analysis.intervention_rate:.1%} |")
        lines.append(f"| Correction Success Rate | {report.correction_analysis.correction_success_rate:.1%} |")
        lines.append("")

        if report.correction_analysis.issues_by_type:
            lines.append("**Issues by Type:**")
            lines.append("")
            for issue_type, count in sorted(report.correction_analysis.issues_by_type.items()):
                lines.append(f"- {issue_type}: {count}")
            lines.append("")

    # By Category
    if report.metrics_by_category:
        lines.append("## Results by Category")
        lines.append("")
        lines.append("| Category | Count | Avg Score | Pass Rate | Precision | Recall | Hallucinations |")
        lines.append("|----------|-------|-----------|-----------|-----------|--------|----------------|")
        for cat_name, metrics in sorted(report.metrics_by_category.items()):
            lines.append(
                f"| {cat_name} | {metrics.count} | {metrics.avg_score:.1f} | "
                f"{metrics.pass_rate:.0%} | {metrics.avg_precision:.0%} | "
                f"{metrics.avg_recall:.0%} | {metrics.total_hallucinations} |"
            )
        lines.append("")

    # By Difficulty
    if report.metrics_by_difficulty:
        lines.append("## Results by Difficulty")
        lines.append("")
        lines.append("| Difficulty | Count | Avg Score | Pass Rate | Precision | Recall | Avg Latency |")
        lines.append("|------------|-------|-----------|-----------|-----------|--------|-------------|")
        for diff_name in ["easy", "medium", "hard"]:
            if diff_name in report.metrics_by_difficulty:
                metrics = report.metrics_by_difficulty[diff_name]
                lines.append(
                    f"| {diff_name} | {metrics.count} | {metrics.avg_score:.1f} | "
                    f"{metrics.pass_rate:.0%} | {metrics.avg_precision:.0%} | "
                    f"{metrics.avg_recall:.0%} | {metrics.avg_latency_ms:,.0f}ms |"
                )
        lines.append("")

    # Worst Performing Questions
    failed_or_low = [
        s for s in report.question_summaries
        if not s.passed or (s.score is not None and s.score < 7)
    ]
    failed_or_low.sort(key=lambda x: x.score if x.score is not None else -1)

    if failed_or_low:
        lines.append("## Worst Performing Questions")
        lines.append("")
        for i, summary in enumerate(failed_or_low[:5], 1):
            score_str = f"{summary.score}/10" if summary.score is not None else "N/A"
            lines.append(f"### {i}. {summary.question_id} ({summary.difficulty}, {summary.category}) - Score: {score_str}")
            lines.append("")
            lines.append(f"**Question:** {summary.question_text}")
            lines.append("")
            if summary.hallucinations:
                lines.append(f"**Hallucinations:**")
                for h in summary.hallucinations:
                    lines.append(f"- {h}")
                lines.append("")
            if summary.missing_concepts:
                lines.append(f"**Missing Concepts:**")
                for m in summary.missing_concepts:
                    lines.append(f"- {m}")
                lines.append("")
        lines.append("")

    # Hallucination Details
    all_hallucinations = []
    for summary in report.question_summaries:
        for h in summary.hallucinations:
            all_hallucinations.append((summary.question_id, h))

    if all_hallucinations:
        lines.append("## All Hallucinations")
        lines.append("")
        lines.append("| Question | Hallucination |")
        lines.append("|----------|---------------|")
        for qid, h in all_hallucinations:
            # Escape pipe characters in hallucination text
            h_escaped = h.replace("|", "\\|")
            lines.append(f"| {qid} | {h_escaped} |")
        lines.append("")

    # Failed Questions
    if report.failed_questions:
        lines.append("## Failed Questions")
        lines.append("")
        lines.append("The following questions failed to evaluate:")
        lines.append("")
        for qid in report.failed_questions:
            lines.append(f"- {qid}")
        lines.append("")
        if report.errors:
            lines.append("**Errors:**")
            lines.append("")
            for err in report.errors:
                lines.append(f"- {err}")
            lines.append("")

    # Individual Results (abbreviated)
    lines.append("## Individual Results")
    lines.append("")
    lines.append("| ID | Category | Difficulty | Score | Passed | Precision | Recall | Latency |")
    lines.append("|----|----------|------------|-------|--------|-----------|--------|---------|")
    for result in report.results:
        score = result.judgment.overall_score if result.judgment else "N/A"
        passed = "Yes" if result.judgment and result.judgment.passed else "No"
        # Find matching summary for category/difficulty
        summary = next((s for s in report.question_summaries if s.question_id == result.question_id), None)
        cat = summary.category if summary else "?"
        diff = summary.difficulty if summary else "?"
        lines.append(
            f"| {result.question_id} | {cat} | {diff} | {score} | {passed} | "
            f"{result.citation_precision:.0%} | {result.citation_recall:.0%} | "
            f"{result.latency_ms:,}ms |"
        )
    lines.append("")

    return "\n".join(lines)
