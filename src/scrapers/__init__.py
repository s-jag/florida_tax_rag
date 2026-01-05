"""Scrapers for Florida tax law data sources."""

from src.scrapers.admin_code import FloridaAdminCodeScraper
from src.scrapers.base import BaseScraper, FetchError, ScraperError
from src.scrapers.case_law import FloridaCaseLawScraper
from src.scrapers.taa import FloridaTAAScraper
from src.scrapers.models import (
    CaseMetadata,
    RawCase,
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
    "FloridaCaseLawScraper",
    "FloridaTAAScraper",
    # Models
    "StatuteMetadata",
    "RawStatute",
    "RuleMetadata",
    "RawRule",
    "TAAMetadata",
    "RawTAA",
    "CaseMetadata",
    "RawCase",
    "ScrapedDocument",
    # Utilities
    "parse_statute_citation",
    "parse_rule_citation",
    "normalize_whitespace",
    "extract_dates",
    "clean_html_text",
]
