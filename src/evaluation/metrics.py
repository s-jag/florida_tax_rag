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


def get_base_citation(citation: str) -> str:
    """Extract the base citation without subsections.

    E.g., "212.05(1)(a)" -> "212.05"
    E.g., "12a-1.001" -> "12a-1.001"
    """
    normalized = normalize_citation(citation)
    # Remove subsection references like (1)(a)(b)
    base = re.sub(r"\([^)]*\)+$", "", normalized)
    # Also handle repeated subsections
    base = re.sub(r"\([^)]*\)", "", base)
    return base


def citations_match(generated: str, expected: str) -> bool:
    """Check if a generated citation matches an expected citation.

    Handles cases where generated has more specificity:
    - Expected "212.05", Generated "212.05(1)(a)" -> True
    - Expected "212.08(1)", Generated "212.08(1)(a)" -> True
    - Expected "12A-1.001", Generated "12A-1.001" -> True
    """
    gen_norm = normalize_citation(generated)
    exp_norm = normalize_citation(expected)

    # Exact match
    if gen_norm == exp_norm:
        return True

    # Check if generated starts with expected (handling subsection specificity)
    if gen_norm.startswith(exp_norm):
        # Make sure it's actually a subsection, not a different statute
        # e.g., 212.05 should match 212.05(1), but not 212.051
        remainder = gen_norm[len(exp_norm):]
        if not remainder or remainder.startswith("("):
            return True

    # Check if they share the same base citation
    gen_base = get_base_citation(generated)
    exp_base = get_base_citation(expected)
    if gen_base == exp_base:
        return True

    return False


def citation_precision(
    generated: list[str],
    expected_statutes: list[str],
    expected_rules: list[str],
) -> float:
    """Calculate precision: correct citations / all generated citations.

    A generated citation is "correct" if it matches any expected citation,
    accounting for subsection specificity (e.g., 212.05(1)(a) matches 212.05).

    Args:
        generated: Citations extracted from generated answer
        expected_statutes: Expected statute citations
        expected_rules: Expected rule citations

    Returns:
        Precision score between 0.0 and 1.0
    """
    if not generated:
        return 1.0  # No citations = no false positives

    expected = expected_statutes + expected_rules
    if not expected:
        return 1.0  # No expected = any citation is fine

    correct = sum(
        1 for g in generated
        if any(citations_match(g, e) for e in expected)
    )
    return correct / len(generated)


def citation_recall(
    generated: list[str],
    expected_statutes: list[str],
    expected_rules: list[str],
) -> float:
    """Calculate recall: expected citations found / total expected citations.

    An expected citation is "found" if any generated citation matches it,
    accounting for subsection specificity.

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

    found = sum(
        1 for e in expected
        if any(citations_match(g, e) for g in generated)
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
