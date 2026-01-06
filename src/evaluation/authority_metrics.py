"""Authority-aware metrics for evaluating legal document retrieval.

These metrics account for the legal hierarchy where primary authority
(statutes) should be ranked higher than secondary sources (advisories).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

# Authority weights for different document types
# Higher weight = more authoritative source
AUTHORITY_WEIGHTS = {
    "statute": 3.0,   # Primary law - binding
    "rule": 2.0,      # Implementing authority - binding
    "case": 2.0,      # Interpretive authority - binding precedent
    "taa": 1.0,       # Technical Assistance Advisement - advisory only
}

# Authority hierarchy for ranking checks
AUTHORITY_HIERARCHY = ["statute", "rule", "case", "taa"]


@dataclass
class AuthorityMetrics:
    """Results from authority-aware evaluation."""

    # Authority-weighted NDCG
    authority_ndcg_at_5: float
    authority_ndcg_at_10: float

    # Hierarchy alignment
    hierarchy_alignment_score: float  # % times higher authority ranked first

    # Primary authority rate
    primary_authority_rate_at_5: float  # % of top-5 that are statutes/rules
    primary_authority_rate_at_10: float

    # Document type distribution
    doc_type_distribution: dict[str, int]  # Counts per type in results

    # Per-rank breakdown (for heatmap)
    authority_by_rank: dict[int, dict[str, int]]  # rank -> {doc_type: count}


def get_authority_weight(doc_type: str) -> float:
    """Get the authority weight for a document type.

    Args:
        doc_type: Document type (statute, rule, case, taa)

    Returns:
        Authority weight (higher = more authoritative)
    """
    return AUTHORITY_WEIGHTS.get(doc_type.lower(), 0.5)


def get_authority_rank(doc_type: str) -> int:
    """Get the hierarchy rank for a document type.

    Args:
        doc_type: Document type

    Returns:
        Rank in hierarchy (0 = highest authority)
    """
    try:
        return AUTHORITY_HIERARCHY.index(doc_type.lower())
    except ValueError:
        return len(AUTHORITY_HIERARCHY)  # Unknown types at bottom


def authority_weighted_dcg(
    doc_types: list[str],
    relevance_scores: Optional[list[float]] = None,
    k: int = 10,
) -> float:
    """Calculate Discounted Cumulative Gain weighted by authority.

    This extends standard DCG by multiplying relevance by authority weight,
    so high-authority documents contribute more to the score.

    Args:
        doc_types: List of document types in ranked order
        relevance_scores: Optional relevance scores (default: 1.0 for all)
        k: Cutoff for evaluation

    Returns:
        Authority-weighted DCG score
    """
    if not doc_types:
        return 0.0

    if relevance_scores is None:
        relevance_scores = [1.0] * len(doc_types)

    dcg = 0.0
    for i, (doc_type, rel) in enumerate(zip(doc_types[:k], relevance_scores[:k])):
        authority = get_authority_weight(doc_type)
        # Standard DCG formula: rel / log2(rank + 1)
        # We multiply relevance by authority weight
        weighted_rel = rel * authority
        dcg += weighted_rel / math.log2(i + 2)  # i+2 because ranks start at 1

    return dcg


def ideal_authority_weighted_dcg(
    doc_types: list[str],
    relevance_scores: Optional[list[float]] = None,
    k: int = 10,
) -> float:
    """Calculate ideal (maximum possible) authority-weighted DCG.

    This assumes perfect ranking where highest authority documents
    with highest relevance come first.

    Args:
        doc_types: List of document types
        relevance_scores: Optional relevance scores
        k: Cutoff for evaluation

    Returns:
        Ideal authority-weighted DCG score
    """
    if not doc_types:
        return 0.0

    if relevance_scores is None:
        relevance_scores = [1.0] * len(doc_types)

    # Create (authority_weight * relevance) pairs and sort descending
    weighted_pairs = [
        (get_authority_weight(dt) * rel, dt)
        for dt, rel in zip(doc_types, relevance_scores)
    ]
    weighted_pairs.sort(reverse=True, key=lambda x: x[0])

    # Calculate DCG with ideal ordering
    idcg = 0.0
    for i, (weighted_rel, _) in enumerate(weighted_pairs[:k]):
        idcg += weighted_rel / math.log2(i + 2)

    return idcg


def authority_weighted_ndcg(
    doc_types: list[str],
    relevance_scores: Optional[list[float]] = None,
    k: int = 10,
) -> float:
    """Calculate Normalized DCG weighted by document authority.

    NDCG = DCG / IDCG, normalized to [0, 1].

    Args:
        doc_types: List of document types in ranked order
        relevance_scores: Optional relevance scores
        k: Cutoff for evaluation

    Returns:
        Authority-weighted NDCG score (0-1)
    """
    dcg = authority_weighted_dcg(doc_types, relevance_scores, k)
    idcg = ideal_authority_weighted_dcg(doc_types, relevance_scores, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def hierarchy_alignment_score(
    doc_types: list[str],
    k: int = 10,
) -> float:
    """Calculate how well the ranking respects authority hierarchy.

    For each pair of documents, check if higher authority is ranked first.
    Returns the percentage of pairs that are correctly ordered.

    Args:
        doc_types: List of document types in ranked order
        k: Cutoff for evaluation

    Returns:
        Alignment score (0-1), where 1 = perfect hierarchy alignment
    """
    types = doc_types[:k]
    if len(types) < 2:
        return 1.0

    correct_pairs = 0
    total_pairs = 0

    for i in range(len(types)):
        for j in range(i + 1, len(types)):
            rank_i = get_authority_rank(types[i])
            rank_j = get_authority_rank(types[j])

            # Higher authority (lower rank number) should come first
            # If i < j in position, then rank_i should be <= rank_j
            total_pairs += 1
            if rank_i <= rank_j:
                correct_pairs += 1

    return correct_pairs / total_pairs if total_pairs > 0 else 1.0


def primary_authority_rate(
    doc_types: list[str],
    k: int = 5,
) -> float:
    """Calculate percentage of top-k results that are primary authority.

    Primary authority = statutes and rules (binding law).

    Args:
        doc_types: List of document types in ranked order
        k: Cutoff for evaluation

    Returns:
        Rate (0-1) of primary authority documents in top-k
    """
    types = doc_types[:k]
    if not types:
        return 0.0

    primary_count = sum(
        1 for dt in types
        if dt.lower() in ("statute", "rule")
    )

    return primary_count / len(types)


def doc_type_distribution(doc_types: list[str]) -> dict[str, int]:
    """Count documents by type.

    Args:
        doc_types: List of document types

    Returns:
        Dictionary mapping doc_type to count
    """
    distribution: dict[str, int] = {}
    for dt in doc_types:
        key = dt.lower()
        distribution[key] = distribution.get(key, 0) + 1
    return distribution


def authority_by_rank(
    doc_types: list[str],
    max_rank: int = 5,
) -> dict[int, dict[str, int]]:
    """Get document type distribution at each rank position.

    This is useful for generating heatmaps showing which document
    types appear at which rank positions.

    Args:
        doc_types: List of document types in ranked order
        max_rank: Maximum rank to track

    Returns:
        Dictionary mapping rank (1-indexed) to doc_type counts
    """
    result: dict[int, dict[str, int]] = {}

    for rank in range(1, max_rank + 1):
        result[rank] = {dt: 0 for dt in AUTHORITY_HIERARCHY}

    for i, dt in enumerate(doc_types[:max_rank]):
        rank = i + 1
        key = dt.lower()
        if key in result[rank]:
            result[rank][key] += 1
        else:
            result[rank][key] = 1

    return result


def compute_authority_metrics(
    doc_types: list[str],
    relevance_scores: Optional[list[float]] = None,
) -> AuthorityMetrics:
    """Compute all authority-aware metrics for a single query result.

    Args:
        doc_types: List of document types in ranked order
        relevance_scores: Optional relevance scores for each document

    Returns:
        AuthorityMetrics dataclass with all computed metrics
    """
    return AuthorityMetrics(
        authority_ndcg_at_5=authority_weighted_ndcg(doc_types, relevance_scores, k=5),
        authority_ndcg_at_10=authority_weighted_ndcg(doc_types, relevance_scores, k=10),
        hierarchy_alignment_score=hierarchy_alignment_score(doc_types, k=10),
        primary_authority_rate_at_5=primary_authority_rate(doc_types, k=5),
        primary_authority_rate_at_10=primary_authority_rate(doc_types, k=10),
        doc_type_distribution=doc_type_distribution(doc_types),
        authority_by_rank=authority_by_rank(doc_types, max_rank=5),
    )


def aggregate_authority_metrics(
    metrics_list: list[AuthorityMetrics],
) -> AuthorityMetrics:
    """Aggregate authority metrics across multiple queries.

    Args:
        metrics_list: List of AuthorityMetrics from individual queries

    Returns:
        Aggregated AuthorityMetrics with averaged scores
    """
    if not metrics_list:
        return AuthorityMetrics(
            authority_ndcg_at_5=0.0,
            authority_ndcg_at_10=0.0,
            hierarchy_alignment_score=0.0,
            primary_authority_rate_at_5=0.0,
            primary_authority_rate_at_10=0.0,
            doc_type_distribution={},
            authority_by_rank={},
        )

    n = len(metrics_list)

    # Average numeric metrics
    avg_ndcg_5 = sum(m.authority_ndcg_at_5 for m in metrics_list) / n
    avg_ndcg_10 = sum(m.authority_ndcg_at_10 for m in metrics_list) / n
    avg_hierarchy = sum(m.hierarchy_alignment_score for m in metrics_list) / n
    avg_primary_5 = sum(m.primary_authority_rate_at_5 for m in metrics_list) / n
    avg_primary_10 = sum(m.primary_authority_rate_at_10 for m in metrics_list) / n

    # Sum distributions
    total_distribution: dict[str, int] = {}
    for m in metrics_list:
        for dt, count in m.doc_type_distribution.items():
            total_distribution[dt] = total_distribution.get(dt, 0) + count

    # Sum rank distributions
    total_by_rank: dict[int, dict[str, int]] = {}
    for m in metrics_list:
        for rank, dist in m.authority_by_rank.items():
            if rank not in total_by_rank:
                total_by_rank[rank] = {}
            for dt, count in dist.items():
                total_by_rank[rank][dt] = total_by_rank[rank].get(dt, 0) + count

    return AuthorityMetrics(
        authority_ndcg_at_5=avg_ndcg_5,
        authority_ndcg_at_10=avg_ndcg_10,
        hierarchy_alignment_score=avg_hierarchy,
        primary_authority_rate_at_5=avg_primary_5,
        primary_authority_rate_at_10=avg_primary_10,
        doc_type_distribution=total_distribution,
        authority_by_rank=total_by_rank,
    )
