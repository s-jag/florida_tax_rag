"""Metrics for evaluating the self-correction system.

Tracks intervention rates, correction success, and the value added by
the validation loop compared to not having it (ablation testing).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class CorrectionAction(Enum):
    """Actions taken by the self-correction system."""

    NONE = "none"  # No issues detected, passed through
    CORRECTED = "corrected"  # Issues found and corrected
    REGENERATED = "regenerated"  # Severe issues, answer regenerated
    FAILED = "failed"  # Correction attempted but failed


@dataclass
class CorrectionEvent:
    """Record of a single correction event."""

    query_id: str
    action: CorrectionAction
    issues_detected: int
    issues_corrected: int
    severity_scores: list[float] = field(default_factory=list)
    hallucination_types: list[str] = field(default_factory=list)
    confidence_before: float = 0.0
    confidence_after: float = 0.0


@dataclass
class CorrectionMetrics:
    """Aggregated metrics for self-correction performance."""

    # Intervention metrics
    total_queries: int
    queries_with_issues: int
    queries_corrected: int
    queries_regenerated: int
    queries_failed: int

    # Rates
    intervention_rate: float  # % of queries triggering validation action
    correction_rate: float  # % of issues that were corrected
    regeneration_rate: float  # % that required regeneration

    # Issue breakdown
    total_issues_detected: int
    total_issues_corrected: int
    issues_by_type: dict[str, int]

    # Severity distribution
    avg_severity: float
    max_severity: float
    severity_distribution: dict[str, int]  # low/medium/high/critical counts

    # Confidence impact
    avg_confidence_delta: float  # How much correction affects confidence

    # Individual events for drill-down
    events: list[CorrectionEvent] = field(default_factory=list)

    @property
    def correction_success_rate(self) -> float:
        """Percentage of detected issues that were successfully corrected."""
        if self.total_issues_detected == 0:
            return 1.0
        return self.total_issues_corrected / self.total_issues_detected


@dataclass
class AblationResult:
    """Results from running with/without validation."""

    validation_enabled: bool
    total_queries: int
    hallucinations_detected: int  # By external judge
    fabricated_citations: int
    avg_faithfulness: float
    avg_confidence: float


@dataclass
class AblationComparison:
    """Comparison between validation ON vs OFF."""

    with_validation: AblationResult
    without_validation: AblationResult

    # Deltas (positive = validation helps)
    hallucination_delta: int
    fabrication_delta: int
    faithfulness_delta: float

    @property
    def errors_prevented(self) -> int:
        """Number of errors prevented by validation."""
        return max(0, self.hallucination_delta)

    @property
    def validation_value(self) -> str:
        """Human-readable statement of validation value."""
        if self.hallucination_delta > 0:
            return (
                f"Validation prevented {self.hallucination_delta} hallucinations "
                f"({self.hallucination_delta / max(1, self.without_validation.hallucinations_detected) * 100:.0f}% reduction)"
            )
        elif self.hallucination_delta == 0:
            return "Validation had no measurable impact on hallucination rate"
        else:
            return "Validation unexpectedly increased hallucination rate (investigate)"


class CorrectionTracker:
    """Tracks correction events during evaluation."""

    def __init__(self):
        self.events: list[CorrectionEvent] = []

    def record_event(
        self,
        query_id: str,
        action: CorrectionAction,
        issues_detected: int = 0,
        issues_corrected: int = 0,
        severity_scores: list[float] | None = None,
        hallucination_types: list[str] | None = None,
        confidence_before: float = 0.0,
        confidence_after: float = 0.0,
    ) -> None:
        """Record a correction event.

        Args:
            query_id: Identifier for the query
            action: What action was taken
            issues_detected: Number of issues found
            issues_corrected: Number of issues successfully corrected
            severity_scores: Severity score for each issue
            hallucination_types: Type of each hallucination
            confidence_before: Confidence before correction
            confidence_after: Confidence after correction
        """
        self.events.append(
            CorrectionEvent(
                query_id=query_id,
                action=action,
                issues_detected=issues_detected,
                issues_corrected=issues_corrected,
                severity_scores=severity_scores or [],
                hallucination_types=hallucination_types or [],
                confidence_before=confidence_before,
                confidence_after=confidence_after,
            )
        )

    def record_from_state(
        self,
        query_id: str,
        state: dict,
    ) -> None:
        """Record event from agent state dictionary.

        Args:
            query_id: Identifier for the query
            state: Agent state with validation/correction results
        """
        validation_result = state.get("validation_result", {})
        correction_result = state.get("correction_result", {})

        # Determine action
        if state.get("validation_passed", True):
            action = CorrectionAction.NONE
        elif correction_result.get("corrections_made"):
            action = CorrectionAction.CORRECTED
        elif state.get("regeneration_count", 0) > 0:
            action = CorrectionAction.REGENERATED
        else:
            action = CorrectionAction.FAILED

        # Extract hallucination info
        hallucinations = validation_result.get("hallucinations", [])
        severity_scores = [h.get("severity", 0.5) for h in hallucinations]
        hallucination_types = [h.get("type", "unknown") for h in hallucinations]

        self.record_event(
            query_id=query_id,
            action=action,
            issues_detected=len(hallucinations),
            issues_corrected=len(correction_result.get("corrections_made", [])),
            severity_scores=severity_scores,
            hallucination_types=hallucination_types,
            confidence_before=state.get("original_confidence", state.get("confidence", 0.0)),
            confidence_after=state.get("confidence", 0.0),
        )

    def compute_metrics(self) -> CorrectionMetrics:
        """Compute aggregated correction metrics.

        Returns:
            CorrectionMetrics with all computed values
        """
        if not self.events:
            return CorrectionMetrics(
                total_queries=0,
                queries_with_issues=0,
                queries_corrected=0,
                queries_regenerated=0,
                queries_failed=0,
                intervention_rate=0.0,
                correction_rate=0.0,
                regeneration_rate=0.0,
                total_issues_detected=0,
                total_issues_corrected=0,
                issues_by_type={},
                avg_severity=0.0,
                max_severity=0.0,
                severity_distribution={},
                avg_confidence_delta=0.0,
                events=[],
            )

        total = len(self.events)
        with_issues = sum(1 for e in self.events if e.action != CorrectionAction.NONE)
        corrected = sum(1 for e in self.events if e.action == CorrectionAction.CORRECTED)
        regenerated = sum(1 for e in self.events if e.action == CorrectionAction.REGENERATED)
        failed = sum(1 for e in self.events if e.action == CorrectionAction.FAILED)

        total_detected = sum(e.issues_detected for e in self.events)
        total_corrected = sum(e.issues_corrected for e in self.events)

        # Aggregate hallucination types
        issues_by_type: dict[str, int] = {}
        for event in self.events:
            for ht in event.hallucination_types:
                issues_by_type[ht] = issues_by_type.get(ht, 0) + 1

        # Severity stats
        all_severities = [s for e in self.events for s in e.severity_scores]
        avg_severity = sum(all_severities) / len(all_severities) if all_severities else 0.0
        max_severity = max(all_severities) if all_severities else 0.0

        # Severity distribution
        severity_dist = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for s in all_severities:
            if s < 0.3:
                severity_dist["low"] += 1
            elif s < 0.5:
                severity_dist["medium"] += 1
            elif s < 0.7:
                severity_dist["high"] += 1
            else:
                severity_dist["critical"] += 1

        # Confidence delta
        confidence_deltas = [
            e.confidence_after - e.confidence_before
            for e in self.events
            if e.action != CorrectionAction.NONE
        ]
        avg_delta = sum(confidence_deltas) / len(confidence_deltas) if confidence_deltas else 0.0

        return CorrectionMetrics(
            total_queries=total,
            queries_with_issues=with_issues,
            queries_corrected=corrected,
            queries_regenerated=regenerated,
            queries_failed=failed,
            intervention_rate=with_issues / total if total > 0 else 0.0,
            correction_rate=total_corrected / total_detected if total_detected > 0 else 1.0,
            regeneration_rate=regenerated / total if total > 0 else 0.0,
            total_issues_detected=total_detected,
            total_issues_corrected=total_corrected,
            issues_by_type=issues_by_type,
            avg_severity=avg_severity,
            max_severity=max_severity,
            severity_distribution=severity_dist,
            avg_confidence_delta=avg_delta,
            events=self.events,
        )

    def reset(self) -> None:
        """Clear all recorded events."""
        self.events = []


def compute_ablation_comparison(
    with_val: AblationResult,
    without_val: AblationResult,
) -> AblationComparison:
    """Compare results with and without validation.

    Args:
        with_val: Results with validation enabled
        without_val: Results with validation disabled

    Returns:
        AblationComparison with deltas
    """
    return AblationComparison(
        with_validation=with_val,
        without_validation=without_val,
        hallucination_delta=without_val.hallucinations_detected - with_val.hallucinations_detected,
        fabrication_delta=without_val.fabricated_citations - with_val.fabricated_citations,
        faithfulness_delta=with_val.avg_faithfulness - without_val.avg_faithfulness,
    )


def format_correction_funnel(metrics: CorrectionMetrics) -> dict:
    """Format correction metrics for funnel visualization.

    Args:
        metrics: CorrectionMetrics instance

    Returns:
        Dict with funnel stages and counts
    """
    return {
        "stages": [
            "Total Queries",
            "Issues Detected",
            "Auto-Corrected",
            "Regenerated",
            "Final Hallucinations",
        ],
        "values": [
            metrics.total_queries,
            metrics.queries_with_issues,
            metrics.queries_corrected,
            metrics.queries_regenerated,
            metrics.queries_failed,  # Assuming failed = final hallucinations
        ],
    }
