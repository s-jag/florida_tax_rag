"""Testing and evaluation metrics for RAG quality."""

from .models import (
    AnswerType,
    Category,
    Difficulty,
    EvalDataset,
    EvalQuestion,
    EvalReport,
    EvalResult,
    JudgmentResult,
)
from .metrics import (
    answer_contains_expected,
    citation_precision,
    citation_recall,
    citations_match,
    extract_citations_from_answer,
    f1_score,
    get_base_citation,
    normalize_citation,
)
from .llm_judge import LLMJudge
from .report import (
    CategoryMetrics,
    DifficultyMetrics,
    FullEvaluationReport,
    QuestionSummary,
    generate_markdown_report,
)
from .runner import EvaluationRunner

__all__ = [
    # Enums
    "AnswerType",
    "Category",
    "Difficulty",
    # Models
    "EvalDataset",
    "EvalQuestion",
    "EvalReport",
    "EvalResult",
    "JudgmentResult",
    # Metrics
    "answer_contains_expected",
    "citation_precision",
    "citation_recall",
    "citations_match",
    "extract_citations_from_answer",
    "f1_score",
    "get_base_citation",
    "normalize_citation",
    # Judge
    "LLMJudge",
    # Report
    "CategoryMetrics",
    "DifficultyMetrics",
    "FullEvaluationReport",
    "QuestionSummary",
    "generate_markdown_report",
    # Runner
    "EvaluationRunner",
]
