"""Retrieval-specific evaluation metrics: MRR, NDCG, Recall@k, Precision@k."""

from __future__ import annotations

import math
from collections.abc import Callable

from pydantic import BaseModel, Field

# =============================================================================
# Data Models
# =============================================================================


class RetrievalMetrics(BaseModel):
    """Aggregated retrieval metrics for a set of queries."""

    mrr: float = Field(default=0.0, description="Mean Reciprocal Rank")
    ndcg_at_5: float = Field(default=0.0, description="NDCG@5")
    ndcg_at_10: float = Field(default=0.0, description="NDCG@10")
    ndcg_at_20: float = Field(default=0.0, description="NDCG@20")
    recall_at_5: float = Field(default=0.0, description="Recall@5")
    recall_at_10: float = Field(default=0.0, description="Recall@10")
    recall_at_20: float = Field(default=0.0, description="Recall@20")
    precision_at_5: float = Field(default=0.0, description="Precision@5")
    precision_at_10: float = Field(default=0.0, description="Precision@10")


class QueryRetrievalResult(BaseModel):
    """Detailed retrieval analysis for a single query."""

    question_id: str = Field(..., description="Evaluation question ID")
    question_text: str = Field(..., description="The query text")
    expected_citations: list[str] = Field(default_factory=list)

    # Retrieved documents info
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    retrieved_citations: list[str] = Field(default_factory=list)
    retrieved_scores: list[float] = Field(default_factory=list)

    # Which expected docs were found and at what rank
    found_expected: dict[str, int] = Field(
        default_factory=dict,
        description="Map of expected citation to rank (1-indexed, 0 if not found)",
    )

    # Metrics for this query
    reciprocal_rank: float = Field(default=0.0)
    ndcg_at_5: float = Field(default=0.0)
    ndcg_at_10: float = Field(default=0.0)
    recall_at_5: float = Field(default=0.0)
    recall_at_10: float = Field(default=0.0)
    recall_at_20: float = Field(default=0.0)

    # Irrelevant docs in top-k
    irrelevant_in_top_5: list[str] = Field(default_factory=list)
    irrelevant_in_top_10: list[str] = Field(default_factory=list)


class MethodComparisonResult(BaseModel):
    """Comparison of retrieval methods for a single query."""

    question_id: str
    question_text: str
    expected_citations: list[str] = Field(default_factory=list)

    # Results by method
    vector_result: QueryRetrievalResult | None = None
    keyword_result: QueryRetrievalResult | None = None
    hybrid_result: QueryRetrievalResult | None = None
    graph_result: QueryRetrievalResult | None = None

    # Best method for this query
    best_method: str = Field(default="", description="Method with highest MRR")
    best_mrr: float = Field(default=0.0)


class AlphaTuningResult(BaseModel):
    """Result of alpha parameter tuning."""

    alpha: float = Field(..., description="Alpha value tested")
    mrr: float = Field(default=0.0)
    recall_at_5: float = Field(default=0.0)
    recall_at_10: float = Field(default=0.0)
    ndcg_at_10: float = Field(default=0.0)

    # Per-query breakdown
    per_query_mrr: dict[str, float] = Field(default_factory=dict)


class RetrievalAnalysisReport(BaseModel):
    """Full retrieval analysis report."""

    # Aggregate metrics by method
    vector_metrics: RetrievalMetrics = Field(default_factory=RetrievalMetrics)
    keyword_metrics: RetrievalMetrics = Field(default_factory=RetrievalMetrics)
    hybrid_metrics: RetrievalMetrics = Field(default_factory=RetrievalMetrics)
    graph_metrics: RetrievalMetrics = Field(default_factory=RetrievalMetrics)

    # Alpha tuning results
    alpha_tuning: list[AlphaTuningResult] = Field(default_factory=list)
    optimal_alpha: float = Field(default=0.5)

    # Queries where retrieval failed
    failed_queries: list[QueryRetrievalResult] = Field(default_factory=list)

    # Method comparison per query
    comparisons: list[MethodComparisonResult] = Field(default_factory=list)

    # Summary
    best_overall_method: str = Field(default="hybrid")
    best_method_mrr: float = Field(default=0.0)


# =============================================================================
# Metric Functions
# =============================================================================


def mean_reciprocal_rank(
    expected: list[str],
    retrieved: list[str],
    matcher: Callable[[str, str], bool] | None = None,
) -> float:
    """Calculate Reciprocal Rank for a single query.

    RR = 1/rank of the first relevant document, or 0 if no match found.

    Args:
        expected: List of expected (relevant) document identifiers
        retrieved: List of retrieved document identifiers (in rank order)
        matcher: Optional function to match expected to retrieved
                 (default: exact string equality)

    Returns:
        Reciprocal rank (1/rank of first match, or 0 if no match)
    """
    if not expected or not retrieved:
        return 0.0

    if matcher is None:

        def matcher(a, b):
            return a == b

    for rank, doc_id in enumerate(retrieved, start=1):
        for exp in expected:
            if matcher(doc_id, exp):
                return 1.0 / rank
    return 0.0


def ndcg_at_k(
    expected: list[str],
    retrieved: list[str],
    k: int,
    matcher: Callable[[str, str], bool] | None = None,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain at k.

    NDCG@k = DCG@k / IDCG@k
    DCG@k = sum(rel_i / log2(i+1)) for i in 1..k

    Args:
        expected: List of expected (relevant) documents
        retrieved: List of retrieved documents in rank order
        k: Cutoff rank
        matcher: Function to match expected to retrieved

    Returns:
        NDCG score between 0 and 1
    """
    if not expected or not retrieved or k <= 0:
        return 0.0

    if matcher is None:

        def matcher(a, b):
            return a == b

    # Calculate DCG@k
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        is_relevant = any(matcher(doc_id, exp) for exp in expected)
        if is_relevant:
            dcg += 1.0 / math.log2(i + 1)

    # Calculate IDCG@k (perfect ranking - all relevant docs at top)
    ideal_relevant = min(len(expected), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_relevant + 1))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def recall_at_k(
    expected: list[str],
    retrieved: list[str],
    k: int,
    matcher: Callable[[str, str], bool] | None = None,
) -> float:
    """Calculate Recall at k.

    Recall@k = |relevant docs in top-k| / |total relevant docs|

    Args:
        expected: List of expected (relevant) documents
        retrieved: List of retrieved documents in rank order
        k: Cutoff rank
        matcher: Function to match expected to retrieved

    Returns:
        Recall score between 0 and 1
    """
    if not expected:
        return 1.0  # No expected = perfect recall (vacuously true)
    if not retrieved or k <= 0:
        return 0.0

    if matcher is None:

        def matcher(a, b):
            return a == b

    found = 0
    for exp in expected:
        for doc_id in retrieved[:k]:
            if matcher(doc_id, exp):
                found += 1
                break

    return found / len(expected)


def precision_at_k(
    expected: list[str],
    retrieved: list[str],
    k: int,
    matcher: Callable[[str, str], bool] | None = None,
) -> float:
    """Calculate Precision at k.

    Precision@k = |relevant docs in top-k| / k

    Args:
        expected: List of expected (relevant) documents
        retrieved: List of retrieved documents in rank order
        k: Cutoff rank
        matcher: Function to match expected to retrieved

    Returns:
        Precision score between 0 and 1
    """
    if not retrieved or k <= 0:
        return 0.0

    if matcher is None:

        def matcher(a, b):
            return a == b

    top_k = retrieved[:k]
    relevant_count = 0

    for doc_id in top_k:
        if any(matcher(doc_id, exp) for exp in expected):
            relevant_count += 1

    return relevant_count / min(k, len(top_k))


def find_rank(
    target: str,
    retrieved: list[str],
    matcher: Callable[[str, str], bool] | None = None,
) -> int:
    """Find the rank of a target document in retrieved list.

    Args:
        target: The document to find
        retrieved: List of retrieved documents in rank order
        matcher: Function to match target to retrieved

    Returns:
        1-indexed rank, or 0 if not found
    """
    if matcher is None:

        def matcher(a, b):
            return a == b

    for rank, doc_id in enumerate(retrieved, start=1):
        if matcher(doc_id, target):
            return rank
    return 0
