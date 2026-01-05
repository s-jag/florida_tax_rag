"""Scrapers for Florida tax law data sources."""

from src.scrapers.admin_code import FloridaAdminCodeScraper
from src.scrapers.base import BaseScraper, FetchError, ScraperError
from src.scrapers.models import (
    RawRule,
    RawStatute,
    RawTAA,
    RuleMetadata,
    ScrapedDocument,
    StatuteMetadata,
    TAAMetadata,
)
from src.scrapers.utils import (
    clean_html_text,
    extract_dates,
    normalize_whitespace,
    parse_rule_citation,
    parse_statute_citation,
)

__all__ = [
    # Base classes
    "BaseScraper",
    "ScraperError",
    "FetchError",
    # Scrapers
    "FloridaAdminCodeScraper",
    # Models
    "StatuteMetadata",
    "RawStatute",
    "RuleMetadata",
    "RawRule",
    "TAAMetadata",
    "RawTAA",
    "ScrapedDocument",
    # Utilities
    "parse_statute_citation",
    "parse_rule_citation",
    "normalize_whitespace",
    "extract_dates",
    "clean_html_text",
]
