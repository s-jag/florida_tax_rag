"""Tests for src/evaluation/retrieval_metrics.py."""

from __future__ import annotations

import pytest

from src.evaluation.retrieval_metrics import (
    find_rank,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

# =============================================================================
# Mean Reciprocal Rank Tests
# =============================================================================


class TestMeanReciprocalRank:
    """Test mean_reciprocal_rank function."""

    def test_first_position(self) -> None:
        """First relevant doc at rank 1 should give RR=1."""
        expected = ["doc1"]
        retrieved = ["doc1", "doc2", "doc3"]
        result = mean_reciprocal_rank(expected, retrieved)
        assert result == 1.0

    def test_second_position(self) -> None:
        """First relevant doc at rank 2 should give RR=0.5."""
        expected = ["doc2"]
        retrieved = ["doc1", "doc2", "doc3"]
        result = mean_reciprocal_rank(expected, retrieved)
        assert result == 0.5

    def test_third_position(self) -> None:
        """First relevant doc at rank 3 should give RR=1/3."""
        expected = ["doc3"]
        retrieved = ["doc1", "doc2", "doc3"]
        result = mean_reciprocal_rank(expected, retrieved)
        assert result == pytest.approx(1 / 3)

    def test_not_found(self) -> None:
        """No relevant doc found should give RR=0."""
        expected = ["doc4"]
        retrieved = ["doc1", "doc2", "doc3"]
        result = mean_reciprocal_rank(expected, retrieved)
        assert result == 0.0

    def test_empty_expected(self) -> None:
        """Empty expected list should give RR=0."""
        expected: list[str] = []
        retrieved = ["doc1", "doc2"]
        result = mean_reciprocal_rank(expected, retrieved)
        assert result == 0.0

    def test_empty_retrieved(self) -> None:
        """Empty retrieved list should give RR=0."""
        expected = ["doc1"]
        retrieved: list[str] = []
        result = mean_reciprocal_rank(expected, retrieved)
        assert result == 0.0

    def test_multiple_expected_first_match(self) -> None:
        """With multiple expected, first match determines RR."""
        expected = ["doc3", "doc2"]
        retrieved = ["doc1", "doc2", "doc3"]
        result = mean_reciprocal_rank(expected, retrieved)
        assert result == 0.5  # doc2 found at rank 2

    def test_custom_matcher(self) -> None:
        """Should support custom matcher function."""
        expected = ["212.05"]
        retrieved = ["statute:212.05", "statute:212.08"]

        def matcher(retrieved_id: str, expected_id: str) -> bool:
            return expected_id in retrieved_id

        result = mean_reciprocal_rank(expected, retrieved, matcher=matcher)
        assert result == 1.0


# =============================================================================
# NDCG@k Tests
# =============================================================================


class TestNDCGAtK:
    """Test ndcg_at_k function."""

    def test_perfect_ranking(self) -> None:
        """Perfect ranking should give NDCG=1."""
        expected = ["doc1", "doc2"]
        retrieved = ["doc1", "doc2", "doc3"]
        result = ndcg_at_k(expected, retrieved, k=5)
        assert result == 1.0

    def test_imperfect_ranking(self) -> None:
        """Imperfect ranking should give NDCG < 1."""
        expected = ["doc2"]
        retrieved = ["doc1", "doc2", "doc3"]
        result = ndcg_at_k(expected, retrieved, k=5)
        # doc2 is at position 2, IDCG has it at position 1
        assert 0 < result < 1

    def test_no_relevant_docs(self) -> None:
        """No relevant docs should give NDCG=0."""
        expected = ["doc4"]
        retrieved = ["doc1", "doc2", "doc3"]
        result = ndcg_at_k(expected, retrieved, k=5)
        assert result == 0.0

    def test_empty_expected(self) -> None:
        """Empty expected list should give NDCG=0."""
        expected: list[str] = []
        retrieved = ["doc1", "doc2"]
        result = ndcg_at_k(expected, retrieved, k=5)
        assert result == 0.0

    def test_k_equals_zero(self) -> None:
        """k=0 should give NDCG=0."""
        expected = ["doc1"]
        retrieved = ["doc1", "doc2"]
        result = ndcg_at_k(expected, retrieved, k=0)
        assert result == 0.0

    def test_k_limits_results(self) -> None:
        """k should limit which results are considered."""
        expected = ["doc3"]
        retrieved = ["doc1", "doc2", "doc3"]
        # With k=2, doc3 at position 3 is not considered
        result = ndcg_at_k(expected, retrieved, k=2)
        assert result == 0.0

    def test_custom_matcher(self) -> None:
        """Should support custom matcher function."""
        expected = ["212.05"]
        retrieved = ["statute:212.05", "statute:212.08"]

        def matcher(retrieved_id: str, expected_id: str) -> bool:
            return expected_id in retrieved_id

        result = ndcg_at_k(expected, retrieved, k=5, matcher=matcher)
        assert result == 1.0


# =============================================================================
# Recall@k Tests
# =============================================================================


class TestRecallAtK:
    """Test recall_at_k function."""

    def test_perfect_recall(self) -> None:
        """All expected docs found should give Recall=1."""
        expected = ["doc1", "doc2"]
        retrieved = ["doc1", "doc2", "doc3"]
        result = recall_at_k(expected, retrieved, k=5)
        assert result == 1.0

    def test_partial_recall(self) -> None:
        """Some expected docs found should give partial recall."""
        expected = ["doc1", "doc4"]
        retrieved = ["doc1", "doc2", "doc3"]
        result = recall_at_k(expected, retrieved, k=5)
        assert result == 0.5  # 1 of 2 found

    def test_no_recall(self) -> None:
        """No expected docs found should give Recall=0."""
        expected = ["doc4", "doc5"]
        retrieved = ["doc1", "doc2", "doc3"]
        result = recall_at_k(expected, retrieved, k=5)
        assert result == 0.0

    def test_empty_expected(self) -> None:
        """Empty expected list should give Recall=1 (vacuously true)."""
        expected: list[str] = []
        retrieved = ["doc1", "doc2"]
        result = recall_at_k(expected, retrieved, k=5)
        assert result == 1.0

    def test_empty_retrieved(self) -> None:
        """Empty retrieved list should give Recall=0."""
        expected = ["doc1"]
        retrieved: list[str] = []
        result = recall_at_k(expected, retrieved, k=5)
        assert result == 0.0

    def test_k_limits_search(self) -> None:
        """k should limit which results are searched."""
        expected = ["doc3"]
        retrieved = ["doc1", "doc2", "doc3"]
        # With k=2, doc3 at position 3 is not found
        result = recall_at_k(expected, retrieved, k=2)
        assert result == 0.0

    def test_custom_matcher(self) -> None:
        """Should support custom matcher function."""
        expected = ["212.05", "212.08"]
        retrieved = ["statute:212.05", "statute:212.08", "rule:12A-1.001"]

        def matcher(retrieved_id: str, expected_id: str) -> bool:
            return expected_id in retrieved_id

        result = recall_at_k(expected, retrieved, k=5, matcher=matcher)
        assert result == 1.0


# =============================================================================
# Precision@k Tests
# =============================================================================


class TestPrecisionAtK:
    """Test precision_at_k function."""

    def test_perfect_precision(self) -> None:
        """All top-k docs relevant should give Precision=1."""
        expected = ["doc1", "doc2", "doc3"]
        retrieved = ["doc1", "doc2"]
        result = precision_at_k(expected, retrieved, k=2)
        assert result == 1.0

    def test_partial_precision(self) -> None:
        """Some top-k docs relevant should give partial precision."""
        expected = ["doc1", "doc3"]
        retrieved = ["doc1", "doc2", "doc3"]
        result = precision_at_k(expected, retrieved, k=3)
        assert result == pytest.approx(2 / 3)

    def test_no_precision(self) -> None:
        """No top-k docs relevant should give Precision=0."""
        expected = ["doc4"]
        retrieved = ["doc1", "doc2", "doc3"]
        result = precision_at_k(expected, retrieved, k=3)
        assert result == 0.0

    def test_empty_retrieved(self) -> None:
        """Empty retrieved list should give Precision=0."""
        expected = ["doc1"]
        retrieved: list[str] = []
        result = precision_at_k(expected, retrieved, k=5)
        assert result == 0.0

    def test_k_equals_zero(self) -> None:
        """k=0 should give Precision=0."""
        expected = ["doc1"]
        retrieved = ["doc1"]
        result = precision_at_k(expected, retrieved, k=0)
        assert result == 0.0

    def test_k_larger_than_retrieved(self) -> None:
        """k larger than retrieved should use actual retrieved length."""
        expected = ["doc1"]
        retrieved = ["doc1", "doc2"]
        # k=5 but only 2 retrieved, precision = 1/2
        result = precision_at_k(expected, retrieved, k=5)
        assert result == 0.5

    def test_custom_matcher(self) -> None:
        """Should support custom matcher function."""
        expected = ["212.05"]
        retrieved = ["statute:212.05", "statute:212.08"]

        def matcher(retrieved_id: str, expected_id: str) -> bool:
            return expected_id in retrieved_id

        result = precision_at_k(expected, retrieved, k=2, matcher=matcher)
        assert result == 0.5  # 1 of 2 is relevant


# =============================================================================
# Find Rank Tests
# =============================================================================


class TestFindRank:
    """Test find_rank function."""

    def test_finds_first_position(self) -> None:
        """Should find target at position 1."""
        result = find_rank("doc1", ["doc1", "doc2", "doc3"])
        assert result == 1

    def test_finds_second_position(self) -> None:
        """Should find target at position 2."""
        result = find_rank("doc2", ["doc1", "doc2", "doc3"])
        assert result == 2

    def test_finds_last_position(self) -> None:
        """Should find target at last position."""
        result = find_rank("doc3", ["doc1", "doc2", "doc3"])
        assert result == 3

    def test_returns_zero_if_not_found(self) -> None:
        """Should return 0 if target not found."""
        result = find_rank("doc4", ["doc1", "doc2", "doc3"])
        assert result == 0

    def test_empty_list(self) -> None:
        """Should return 0 for empty list."""
        result = find_rank("doc1", [])
        assert result == 0

    def test_custom_matcher(self) -> None:
        """Should support custom matcher function."""

        def matcher(doc_id: str, target: str) -> bool:
            return target in doc_id

        result = find_rank("212.05", ["statute:212.05", "statute:212.08"], matcher=matcher)
        assert result == 1
