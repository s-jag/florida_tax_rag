"""Evaluation metrics for citation and answer quality."""

from __future__ import annotations

import re


def normalize_citation(citation: str) -> str:
    """Normalize a citation for comparison.

    Handles various formats:
    - "Fla. Stat. § 212.05" -> "212.05"
    - "§ 212.05(1)(a)" -> "212.05(1)(a)"
    - "Rule 12A-1.001" -> "12a-1.001"
    - "F.A.C. 12A-1.001" -> "12a-1.001"
    """
    normalized = citation.lower().strip()
    # Remove common prefixes
    normalized = re.sub(r"^(fla\.?\s*stat\.?\s*)?§?\s*", "", normalized)
    normalized = re.sub(r"^(rule\s+|f\.?a\.?c\.?\s*r?\.?\s*)", "", normalized)
    # Remove all whitespace
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def citation_precision(
    generated: list[str],
    expected_statutes: list[str],
    expected_rules: list[str],
) -> float:
    """Calculate precision: correct citations / all generated citations.

    Args:
        generated: Citations extracted from generated answer
        expected_statutes: Expected statute citations
        expected_rules: Expected rule citations

    Returns:
        Precision score between 0.0 and 1.0
    """
    if not generated:
        return 1.0  # No citations = no false positives

    expected_normalized = set(
        normalize_citation(c) for c in expected_statutes + expected_rules
    )

    correct = sum(
        1 for g in generated if normalize_citation(g) in expected_normalized
    )
    return correct / len(generated)


def citation_recall(
    generated: list[str],
    expected_statutes: list[str],
    expected_rules: list[str],
) -> float:
    """Calculate recall: correct citations / expected citations.

    Args:
        generated: Citations extracted from generated answer
        expected_statutes: Expected statute citations
        expected_rules: Expected rule citations

    Returns:
        Recall score between 0.0 and 1.0
    """
    expected = expected_statutes + expected_rules
    if not expected:
        return 1.0  # No expected citations = perfect recall

    generated_normalized = set(normalize_citation(g) for g in generated)

    found = sum(
        1 for e in expected if normalize_citation(e) in generated_normalized
    )
    return found / len(expected)


def answer_contains_expected(
    answer: str,
    expected_phrases: list[str],
) -> float:
    """Calculate fraction of expected phrases found in answer.

    Args:
        answer: Generated answer text
        expected_phrases: Key phrases that should appear

    Returns:
        Score between 0.0 and 1.0
    """
    if not expected_phrases:
        return 1.0

    answer_lower = answer.lower()
    found = sum(1 for phrase in expected_phrases if phrase.lower() in answer_lower)
    return found / len(expected_phrases)


def extract_citations_from_answer(answer: str) -> list[str]:
    """Extract statute and rule citations from generated answer.

    Args:
        answer: Generated answer text

    Returns:
        List of extracted citations (deduplicated)
    """
    citations = []

    # Statute patterns: § 212.05, Fla. Stat. § 212.05(1), etc.
    statute_pattern = r"(?:Fla\.?\s*Stat\.?\s*)?§\s*(\d+\.\d+(?:\([^)]+\))*)"
    for match in re.finditer(statute_pattern, answer, re.IGNORECASE):
        citations.append(match.group(1))

    # Rule patterns: Rule 12A-1.005, F.A.C. 12A-1.005
    rule_pattern = r"(?:Rule\s+|F\.?A\.?C\.?\s*R?\.?\s*)(\d+[A-Z]?-\d+\.\d+)"
    for match in re.finditer(rule_pattern, answer, re.IGNORECASE):
        citations.append(match.group(1))

    return list(set(citations))  # Deduplicate


def f1_score(precision: float, recall: float) -> float:
    """Calculate F1 score from precision and recall.

    Args:
        precision: Precision score (0.0 to 1.0)
        recall: Recall score (0.0 to 1.0)

    Returns:
        F1 score between 0.0 and 1.0
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)
