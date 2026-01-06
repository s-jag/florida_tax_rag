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
from .retrieval_metrics import (
    AlphaTuningResult,
    MethodComparisonResult,
    QueryRetrievalResult,
    RetrievalAnalysisReport,
    RetrievalMetrics,
    find_rank,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from .retrieval_analysis import (
    RetrievalAnalyzer,
    debug_retrieval,
    generate_retrieval_markdown_report,
)

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
    # Retrieval Metrics
    "AlphaTuningResult",
    "MethodComparisonResult",
    "QueryRetrievalResult",
    "RetrievalAnalysisReport",
    "RetrievalMetrics",
    "find_rank",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    # Retrieval Analysis
    "RetrievalAnalyzer",
    "debug_retrieval",
    "generate_retrieval_markdown_report",
]
