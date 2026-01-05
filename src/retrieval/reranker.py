"""Legal-specific re-ranking for retrieval results."""

from __future__ import annotations

from datetime import date
from typing import Optional

from .models import RetrievalResult


class LegalReranker:
    """Re-ranks results using legal-specific heuristics.

    This reranker applies domain-specific knowledge to improve retrieval
    results for legal research, including:
    - Primary authority preference (statutes > rules > cases > TAAs)
    - Recency boost for more recent documents
    - Diversity penalty to avoid duplicate content from same document
    """

    # Document type priority weights
    # Primary authorities (statutes) are most authoritative
    # Implementing authorities (rules) are next
    # Interpretive authorities (cases, TAAs) are least authoritative
    DOC_TYPE_PRIORITY = {
        "statute": 1.0,  # Primary authority - highest weight
        "rule": 0.9,  # Implementing authority
        "case": 0.8,  # Interpretive authority
        "taa": 0.7,  # Technical Assistance Advisement - advisory only
    }

    # Year range for recency calculation (documents older than this get min boost)
    RECENCY_YEARS = 10

    def __init__(
        self,
        doc_type_priority: Optional[dict[str, float]] = None,
        recency_years: int = 10,
    ):
        """Initialize the reranker.

        Args:
            doc_type_priority: Custom priority weights for document types
            recency_years: Number of years for recency calculation
        """
        self.doc_type_priority = doc_type_priority or self.DOC_TYPE_PRIORITY
        self.recency_years = recency_years

    def rerank(
        self,
        results: list[RetrievalResult],
        prefer_recent: bool = True,
        prefer_primary: bool = True,
        diversity_penalty: float = 0.1,
    ) -> list[RetrievalResult]:
        """Re-rank results with legal heuristics.

        Scoring factors applied:
        1. Original retrieval score (base)
        2. Document type priority multiplier (if prefer_primary)
        3. Recency boost (if prefer_recent)
        4. Graph relationship boost
        5. Diversity penalty for same doc_id

        Args:
            results: List of retrieval results to rerank
            prefer_recent: Whether to boost newer documents
            prefer_primary: Whether to boost primary authority (statutes)
            diversity_penalty: Score reduction for duplicate doc_ids

        Returns:
            Re-ranked list of retrieval results
        """
        if not results:
            return results

        # Track seen doc_ids for diversity penalty
        seen_doc_ids: dict[str, int] = {}

        for result in results:
            adjusted_score = result.score

            # 1. Apply document type priority
            if prefer_primary:
                type_weight = self.doc_type_priority.get(result.doc_type, 0.5)
                adjusted_score *= type_weight

            # 2. Apply recency boost
            if prefer_recent and result.effective_date:
                recency_boost = self._calculate_recency_boost(result.effective_date)
                adjusted_score *= recency_boost

            # 3. Add graph boost
            adjusted_score += result.graph_boost

            # 4. Apply diversity penalty for duplicate doc_ids
            doc_id = result.doc_id
            if doc_id in seen_doc_ids:
                # Apply increasing penalty for each duplicate
                penalty = diversity_penalty * seen_doc_ids[doc_id]
                adjusted_score -= penalty
                seen_doc_ids[doc_id] += 1
            else:
                seen_doc_ids[doc_id] = 1

            # Update the score
            result.score = max(0.0, adjusted_score)  # Ensure non-negative

        # Sort by adjusted score (descending)
        return sorted(results, key=lambda r: r.score, reverse=True)

    def _calculate_recency_boost(self, effective_date: date) -> float:
        """Calculate recency boost based on effective date.

        Recent documents get a higher boost (up to 1.1x).
        Old documents get a lower boost (down to 0.9x).

        Args:
            effective_date: The document's effective date

        Returns:
            Boost multiplier (0.9 to 1.1)
        """
        today = date.today()
        age_days = (today - effective_date).days
        age_years = age_days / 365.25

        if age_years <= 0:
            return 1.1  # Future or current date
        elif age_years >= self.recency_years:
            return 0.9  # Old document
        else:
            # Linear interpolation: 1.1 at 0 years, 0.9 at recency_years
            return 1.1 - (0.2 * age_years / self.recency_years)

    def boost_by_query_match(
        self,
        results: list[RetrievalResult],
        query_terms: list[str],
        boost: float = 0.1,
    ) -> list[RetrievalResult]:
        """Additional boost for results that contain query terms in citation.

        This helps exact citation searches rank higher.

        Args:
            results: List of retrieval results
            query_terms: Terms from the query to match
            boost: Score boost per matching term

        Returns:
            Boosted results
        """
        query_terms_lower = [t.lower() for t in query_terms]

        for result in results:
            citation = (result.citation or "").lower()
            text = result.text[:500].lower()  # Check first 500 chars

            for term in query_terms_lower:
                if term in citation:
                    result.score += boost * 2  # Higher boost for citation match
                elif term in text:
                    result.score += boost

        return sorted(results, key=lambda r: r.score, reverse=True)
