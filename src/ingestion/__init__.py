"""Ingestion module for data consolidation and processing."""

from .consolidate import (
    consolidate_all,
    consolidate_cases,
    consolidate_rules,
    consolidate_statutes,
    consolidate_taas,
)
from .models import Corpus, CorpusMetadata, DocumentType, LegalDocument

__all__ = [
    "Corpus",
    "CorpusMetadata",
    "DocumentType",
    "LegalDocument",
    "consolidate_all",
    "consolidate_cases",
    "consolidate_rules",
    "consolidate_statutes",
    "consolidate_taas",
]
