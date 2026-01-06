"""Evaluation models for RAG quality assessment."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, computed_field


class Difficulty(str, Enum):
    """Question difficulty level."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class AnswerType(str, Enum):
    """Expected answer type."""

    YES_NO = "yes/no"
    NUMERIC = "numeric"
    IT_DEPENDS = "it_depends"
    EXPLANATION = "explanation"


class Category(str, Enum):
    """Tax topic category."""

    SALES_TAX = "sales_tax"
    PROPERTY_TAX = "property_tax"
    CORPORATE_TAX = "corporate_tax"
    EXEMPTIONS = "exemptions"
    PROCEDURES = "procedures"


class EvalQuestion(BaseModel):
    """A single evaluation question with expected answers."""

    id: str = Field(..., description="Unique question ID (e.g., 'eval_001')")
    question: str = Field(..., description="The tax law question to evaluate")
    category: Category = Field(..., description="Tax topic category")
    difficulty: Difficulty = Field(..., description="Question difficulty level")
    expected_statutes: list[str] = Field(
        default_factory=list,
        description="Expected statute citations (e.g., ['212.05', '212.08(1)'])",
    )
    expected_rules: list[str] = Field(
        default_factory=list,
        description="Expected rule citations (e.g., ['12A-1.001'])",
    )
    expected_answer_contains: list[str] = Field(
        default_factory=list,
        description="Key phrases that should appear in answer",
    )
    expected_answer_type: AnswerType = Field(..., description="Type of expected answer")
    notes: str = Field(default="", description="Explanation for evaluators")


class JudgmentResult(BaseModel):
    """LLM judge's evaluation of an answer."""

    correctness: int = Field(..., ge=0, le=10, description="Factual correctness (0-10)")
    completeness: int = Field(..., ge=0, le=10, description="Answer completeness (0-10)")
    clarity: int = Field(..., ge=0, le=10, description="Clarity and organization (0-10)")
    citation_accuracy: int = Field(
        ..., ge=0, le=10, description="Citation accuracy (0-10)"
    )
    hallucinations: list[str] = Field(
        default_factory=list, description="List of hallucinated facts"
    )
    missing_concepts: list[str] = Field(
        default_factory=list, description="Expected but missing concepts"
    )
    overall_score: int = Field(..., ge=0, le=10, description="Overall quality (0-10)")
    reasoning: str = Field(..., description="Judge's reasoning")

    @computed_field
    @property
    def passed(self) -> bool:
        """Whether the answer passes (overall >= 7 and no critical hallucinations)."""
        return self.overall_score >= 7 and len(self.hallucinations) == 0


class EvalResult(BaseModel):
    """Complete evaluation result for a single question."""

    question_id: str = Field(..., description="Reference to EvalQuestion.id")
    generated_answer: str = Field(..., description="The RAG system's answer")
    generated_citations: list[str] = Field(
        default_factory=list, description="Citations extracted from answer"
    )

    # Computed metrics
    citation_precision: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Correct citations / All generated citations",
    )
    citation_recall: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Correct citations / Expected citations",
    )
    answer_contains_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of expected phrases found",
    )

    # LLM judgment
    judgment: Optional[JudgmentResult] = Field(
        default=None, description="LLM judge evaluation"
    )

    # Performance
    latency_ms: int = Field(..., description="Query latency in milliseconds")

    # Metadata
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)
    model_version: str = Field(default="", description="RAG system version")


class EvalDataset(BaseModel):
    """Collection of evaluation questions."""

    metadata: dict = Field(default_factory=dict)
    questions: list[EvalQuestion] = Field(default_factory=list)


class EvalReport(BaseModel):
    """Aggregated evaluation results."""

    metadata: dict = Field(default_factory=dict)
    results: list[EvalResult] = Field(default_factory=list)

    @computed_field
    @property
    def avg_citation_precision(self) -> float:
        """Average citation precision across all results."""
        if not self.results:
            return 0.0
        return sum(r.citation_precision for r in self.results) / len(self.results)

    @computed_field
    @property
    def avg_citation_recall(self) -> float:
        """Average citation recall across all results."""
        if not self.results:
            return 0.0
        return sum(r.citation_recall for r in self.results) / len(self.results)

    @computed_field
    @property
    def avg_overall_score(self) -> float:
        """Average overall score from LLM judge."""
        scored = [r for r in self.results if r.judgment]
        if not scored:
            return 0.0
        return sum(r.judgment.overall_score for r in scored) / len(scored)

    @computed_field
    @property
    def pass_rate(self) -> float:
        """Percentage of answers that passed evaluation."""
        scored = [r for r in self.results if r.judgment]
        if not scored:
            return 0.0
        return sum(1 for r in scored if r.judgment.passed) / len(scored)
