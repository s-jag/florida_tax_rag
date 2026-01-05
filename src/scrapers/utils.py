"""Utility functions for scraping and parsing legal documents."""

from __future__ import annotations

import re
from datetime import date

# Regex patterns for Florida legal citations

# Matches statute citations like:
# - § 212.05
# - §212.05
# - Section 212.05
# - s. 212.05
# - Fla. Stat. § 212.05
# - F.S. 212.05
# - 212.05, F.S.
# - § 212.05(1)(a)2.
STATUTE_PATTERN = re.compile(
    r"""
    (?:
        (?:Fla(?:\.|\s)?(?:Stat(?:\.|\s)?)?)?  # Optional "Fla. Stat." prefix
        (?:§|section|s\.)\s*                    # Section symbol or word
        (\d{1,3}\.\d{2,4})                      # Chapter.Section (e.g., 212.05)
        ((?:\([0-9a-z]+\))*(?:\d+\.?[a-z]?)?)  # Optional subsections
    |
        (\d{1,3}\.\d{2,4})                      # Chapter.Section
        ((?:\([0-9a-z]+\))*(?:\d+\.?[a-z]?)?)  # Optional subsections
        ,?\s*F\.?S\.?                           # F.S. suffix
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Matches rule citations like:
# - Rule 12A-1.005
# - R. 12A-1.005
# - Fla. Admin. Code R. 12A-1.005
# - F.A.C. 12A-1.005
# - 12A-1.005, F.A.C.
RULE_PATTERN = re.compile(
    r"""
    (?:
        (?:Fla(?:\.|\s)?Admin(?:\.|\s)?Code\s*)?  # Optional "Fla. Admin. Code" prefix
        (?:R(?:ule)?\.?\s*)                        # Rule or R.
        (\d{1,2}[A-Z]?-\d+\.\d+)                  # Rule number (e.g., 12A-1.005)
        ((?:\([0-9a-z]+\))*)                      # Optional subsections
    |
        (\d{1,2}[A-Z]?-\d+\.\d+)                  # Rule number
        ((?:\([0-9a-z]+\))*)                      # Optional subsections
        ,?\s*F\.?A\.?C\.?                         # F.A.C. suffix
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Date patterns
DATE_PATTERNS = [
    # Month DD, YYYY (e.g., "January 1, 2024")
    (
        re.compile(
            r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b",
            re.IGNORECASE,
        ),
        lambda m: _parse_month_day_year(m.group(1), m.group(2), m.group(3)),
    ),
    # MM/DD/YYYY or MM-DD-YYYY
    (
        re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b"),
        lambda m: _parse_numeric_date(m.group(1), m.group(2), m.group(3)),
    ),
    # YYYY-MM-DD (ISO format)
    (
        re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),
        lambda m: _parse_iso_date(m.group(1), m.group(2), m.group(3)),
    ),
]

MONTH_MAP = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def _parse_month_day_year(month: str, day: str, year: str) -> date | None:
    """Parse a date from month name, day, and year."""
    try:
        month_num = MONTH_MAP.get(month.lower())
        if month_num is None:
            return None
        return date(int(year), month_num, int(day))
    except ValueError:
        return None


def _parse_numeric_date(month: str, day: str, year: str) -> date | None:
    """Parse a date from numeric MM/DD/YYYY."""
    try:
        return date(int(year), int(month), int(day))
    except ValueError:
        return None


def _parse_iso_date(year: str, month: str, day: str) -> date | None:
    """Parse a date from ISO format YYYY-MM-DD."""
    try:
        return date(int(year), int(month), int(day))
    except ValueError:
        return None


def parse_statute_citation(text: str) -> list[str]:
    """Extract Florida Statute citations from text.

    Args:
        text: The text to search for citations.

    Returns:
        List of unique statute citations in normalized format (e.g., "212.05(1)(a)").
    """
    citations = set()

    for match in STATUTE_PATTERN.finditer(text):
        # Groups 1,2 are from the first alternative (§ prefix)
        # Groups 3,4 are from the second alternative (F.S. suffix)
        section = match.group(1) or match.group(3)
        subsection = match.group(2) or match.group(4) or ""

        if section:
            # Normalize: remove spaces, lowercase subsection letters
            citation = section + subsection.lower().replace(" ", "")
            citations.add(citation)

    return sorted(citations)


def parse_rule_citation(text: str) -> list[str]:
    """Extract Florida Administrative Code rule citations from text.

    Args:
        text: The text to search for citations.

    Returns:
        List of unique rule citations in normalized format (e.g., "12A-1.005").
    """
    citations = set()

    for match in RULE_PATTERN.finditer(text):
        # Groups 1,2 are from the first alternative (Rule prefix)
        # Groups 3,4 are from the second alternative (F.A.C. suffix)
        rule_num = match.group(1) or match.group(3)
        subsection = match.group(2) or match.group(4) or ""

        if rule_num:
            # Normalize: uppercase the letter in rule number
            normalized = re.sub(
                r"(\d+)([a-z])-",
                lambda m: f"{m.group(1)}{m.group(2).upper()}-",
                rule_num,
                flags=re.IGNORECASE,
            )
            citation = normalized + subsection.lower().replace(" ", "")
            citations.add(citation)

    return sorted(citations)


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text.

    - Collapses multiple spaces/tabs to single space
    - Normalizes line endings
    - Strips leading/trailing whitespace from each line
    - Removes excessive blank lines (max 2 consecutive)

    Args:
        text: The text to normalize.

    Returns:
        Normalized text.
    """
    # Normalize line endings to \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Process line by line
    lines = []
    blank_count = 0

    for line in text.split("\n"):
        # Collapse multiple spaces/tabs to single space, strip
        line = re.sub(r"[ \t]+", " ", line).strip()

        if not line:
            blank_count += 1
            # Allow max 2 consecutive blank lines
            if blank_count <= 2:
                lines.append("")
        else:
            blank_count = 0
            lines.append(line)

    # Join and strip leading/trailing blank lines
    result = "\n".join(lines).strip()

    return result


def extract_dates(text: str) -> list[date]:
    """Extract dates from text.

    Args:
        text: The text to search for dates.

    Returns:
        List of unique dates found, sorted chronologically.
    """
    dates = set()

    for pattern, parser in DATE_PATTERNS:
        for match in pattern.finditer(text):
            parsed = parser(match)
            if parsed is not None:
                dates.add(parsed)

    return sorted(dates)


def extract_section_number(text: str) -> str | None:
    """Extract a statute section number from text (e.g., from a heading).

    Args:
        text: Text that may contain a section number.

    Returns:
        The section number (e.g., "212.05") or None.
    """
    match = re.search(r"\b(\d{1,3}\.\d{2,4})\b", text)
    return match.group(1) if match else None


def extract_chapter_number(text: str) -> int | None:
    """Extract a chapter number from text.

    Args:
        text: Text that may contain a chapter reference.

    Returns:
        The chapter number or None.
    """
    match = re.search(r"\bChapter\s+(\d+)\b", text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # Try just a number at word boundary
    match = re.search(r"\b(\d{3})\b", text)
    if match:
        return int(match.group(1))

    return None


def clean_html_text(html_text: str) -> str:
    """Clean text extracted from HTML.

    Removes common HTML artifacts and normalizes the result.

    Args:
        html_text: Text that was extracted from HTML.

    Returns:
        Cleaned text.
    """
    # Remove common HTML entities that might remain
    replacements = {
        "&nbsp;": " ",
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&#39;": "'",
        "&mdash;": "—",
        "&ndash;": "–",
        "&sect;": "§",
    }

    for entity, replacement in replacements.items():
        html_text = html_text.replace(entity, replacement)

    # Remove any remaining HTML tags that slipped through
    html_text = re.sub(r"<[^>]+>", "", html_text)

    return normalize_whitespace(html_text)
