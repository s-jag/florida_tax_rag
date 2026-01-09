"""Consolidation functions to convert raw scraped data to unified LegalDocument format."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

from .models import Corpus, CorpusMetadata, DocumentType, LegalDocument


def parse_date(date_str: str | None) -> date | None:
    """Parse a date string into a date object."""
    if date_str is None:
        return None

    # Handle ISO format with timezone
    if "T" in str(date_str):
        date_str = str(date_str).split("T")[0]

    try:
        return datetime.strptime(str(date_str), "%Y-%m-%d").date()
    except ValueError:
        return None


def parse_datetime(dt_str: str | None) -> datetime:
    """Parse a datetime string, defaulting to now if invalid."""
    if dt_str is None:
        return datetime.now()

    try:
        # Handle ISO format with timezone
        if "+" in dt_str:
            dt_str = dt_str.split("+")[0]
        if dt_str.endswith("Z"):
            dt_str = dt_str[:-1]
        return datetime.fromisoformat(dt_str)
    except ValueError:
        return datetime.now()


def load_json_file(path: Path) -> dict | None:
    """Load a JSON file."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, FileNotFoundError):
        return None


def consolidate_statute(data: dict, file_path: Path) -> LegalDocument | None:
    """Convert a raw statute to a LegalDocument."""
    metadata = data.get("metadata", {})
    section = metadata.get("section", "")

    if not section:
        return None

    # Build parent_id from chapter
    chapter = metadata.get("chapter")
    parent_id = f"statute:chapter_{chapter}" if chapter else None

    # Extract history to metadata dict
    extra_metadata = {
        "title_number": metadata.get("title_number"),
        "title_name": metadata.get("title"),
        "chapter": chapter,
        "chapter_name": metadata.get("chapter_name", ""),
        "section_name": metadata.get("section_name", ""),
        "subsection": metadata.get("subsection"),
        "history": metadata.get("history", []),
    }

    return LegalDocument(
        id=f"statute:{section}",
        doc_type=DocumentType.STATUTE,
        title=metadata.get("section_name", "") or f"Section {section}",
        full_citation=metadata.get("full_citation", f"Fla. Stat. § {section}"),
        text=data.get("text", ""),
        effective_date=parse_date(metadata.get("effective_date")),
        source_url=data.get("source_url", ""),
        parent_id=parent_id,
        children_ids=[],  # Would need additional processing to build hierarchy
        cites_statutes=[],  # Statutes don't have explicit cross-references in our data
        cites_rules=[],
        cites_cases=[],
        scraped_at=parse_datetime(data.get("scraped_at")),
        metadata=extra_metadata,
    )


def consolidate_rule(data: dict, file_path: Path) -> LegalDocument | None:
    """Convert a raw rule to a LegalDocument."""
    metadata = data.get("metadata", {})
    rule_number = metadata.get("rule_number", "")

    if not rule_number:
        return None

    # Build parent_id from chapter
    chapter = metadata.get("chapter", "")
    parent_id = f"rule:chapter_{chapter}" if chapter else None

    # Extract statute citations from rulemaking_authority and law_implemented
    cites_statutes = list(
        set(
            metadata.get("rulemaking_authority", [])
            + metadata.get("law_implemented", [])
            + metadata.get("references_statutes", [])
        )
    )

    extra_metadata = {
        "chapter": chapter,
        "rulemaking_authority": metadata.get("rulemaking_authority", []),
        "law_implemented": metadata.get("law_implemented", []),
        "references_statutes": metadata.get("references_statutes", []),
    }

    return LegalDocument(
        id=f"rule:{rule_number}",
        doc_type=DocumentType.RULE,
        title=metadata.get("title", "") or f"Rule {rule_number}",
        full_citation=metadata.get("full_citation", f"Fla. Admin. Code R. {rule_number}"),
        text=data.get("text", ""),
        effective_date=parse_date(metadata.get("effective_date")),
        source_url=data.get("source_url", ""),
        parent_id=parent_id,
        children_ids=[],
        cites_statutes=cites_statutes,
        cites_rules=[],  # Rules don't reference other rules in our data
        cites_cases=[],
        scraped_at=parse_datetime(data.get("scraped_at")),
        metadata=extra_metadata,
    )


def consolidate_taa(data: dict, file_path: Path) -> LegalDocument | None:
    """Convert a raw TAA to a LegalDocument."""
    metadata = data.get("metadata", {})
    taa_number = metadata.get("taa_number", "")

    if not taa_number:
        return None

    # Normalize TAA number for ID (replace spaces with underscores)
    taa_id = taa_number.replace(" ", "_")

    extra_metadata = {
        "tax_type": metadata.get("tax_type", ""),
        "tax_type_code": metadata.get("tax_type_code", ""),
        "topics": metadata.get("topics", []),
        "question": metadata.get("question", ""),
        "answer": metadata.get("answer", ""),
        "pdf_path": data.get("pdf_path"),
    }

    return LegalDocument(
        id=f"taa:{taa_id}",
        doc_type=DocumentType.TAA,
        title=metadata.get("title", "") or f"TAA {taa_number}",
        full_citation=metadata.get("full_citation", f"Fla. DOR {taa_number}"),
        text=data.get("text", ""),
        effective_date=parse_date(metadata.get("issue_date")),
        source_url=data.get("source_url", ""),
        parent_id=None,
        children_ids=[],
        cites_statutes=metadata.get("statutes_cited", []),
        cites_rules=metadata.get("rules_cited", []),
        cites_cases=[],
        scraped_at=parse_datetime(data.get("scraped_at")),
        metadata=extra_metadata,
    )


def consolidate_case(data: dict, file_path: Path) -> LegalDocument | None:
    """Convert a raw case to a LegalDocument."""
    metadata = data.get("metadata", {})
    cluster_id = metadata.get("cluster_id")

    if not cluster_id:
        return None

    # Convert case citations (int IDs) to case:{id} format
    cases_cited = [f"case:{cid}" for cid in metadata.get("cases_cited", [])]

    extra_metadata = {
        "case_name_full": metadata.get("case_name_full", ""),
        "citations": metadata.get("citations", []),
        "court": metadata.get("court", ""),
        "court_id": metadata.get("court_id", ""),
        "docket_number": metadata.get("docket_number", ""),
        "judges": metadata.get("judges", ""),
        "pdf_url": data.get("pdf_url"),
        "has_opinion_html": data.get("opinion_html") is not None,
    }

    return LegalDocument(
        id=f"case:{cluster_id}",
        doc_type=DocumentType.CASE,
        title=metadata.get("case_name", "") or f"Case {cluster_id}",
        full_citation=metadata.get("full_citation", metadata.get("case_name", "")),
        text=data.get("opinion_text", ""),
        effective_date=parse_date(metadata.get("date_filed")),
        source_url=data.get("source_url", ""),
        parent_id=None,
        children_ids=[],
        cites_statutes=metadata.get("statutes_cited", []),
        cites_rules=[],
        cites_cases=cases_cited,
        scraped_at=parse_datetime(data.get("scraped_at")),
        metadata=extra_metadata,
    )


def consolidate_statutes(data_dir: Path) -> list[LegalDocument]:
    """Load and consolidate all statute files."""
    documents = []
    statutes_dir = data_dir / "statutes"

    if not statutes_dir.exists():
        return documents

    for chapter_dir in sorted(statutes_dir.glob("chapter_*")):
        for json_file in sorted(chapter_dir.glob("*.json")):
            data = load_json_file(json_file)
            if data:
                doc = consolidate_statute(data, json_file)
                if doc:
                    documents.append(doc)

    return documents


def consolidate_rules(data_dir: Path) -> list[LegalDocument]:
    """Load and consolidate all rule files."""
    documents = []
    rules_dir = data_dir / "admin_code"

    if not rules_dir.exists():
        return documents

    for chapter_dir in sorted(rules_dir.glob("chapter_*")):
        for json_file in sorted(chapter_dir.glob("*.json")):
            data = load_json_file(json_file)
            if data:
                doc = consolidate_rule(data, json_file)
                if doc:
                    documents.append(doc)

    return documents


def consolidate_taas(data_dir: Path) -> list[LegalDocument]:
    """Load and consolidate all TAA files."""
    documents = []
    taa_dir = data_dir / "taa"

    if not taa_dir.exists():
        return documents

    for json_file in sorted(taa_dir.glob("TAA_*.json")):
        data = load_json_file(json_file)
        if data:
            doc = consolidate_taa(data, json_file)
            if doc:
                documents.append(doc)

    return documents


def consolidate_cases(data_dir: Path) -> list[LegalDocument]:
    """Load and consolidate all case files."""
    documents = []
    case_dir = data_dir / "case_law"

    if not case_dir.exists():
        return documents

    for json_file in sorted(case_dir.glob("case_*.json")):
        data = load_json_file(json_file)
        if data:
            doc = consolidate_case(data, json_file)
            if doc:
                documents.append(doc)

    return documents


def consolidate_all(data_dir: Path) -> Corpus:
    """Load all raw data and consolidate into a unified corpus.

    Args:
        data_dir: Path to the raw data directory

    Returns:
        A Corpus object containing all consolidated documents
    """
    all_documents = []

    # Consolidate each document type
    print("Consolidating statutes...")
    statutes = consolidate_statutes(data_dir)
    all_documents.extend(statutes)
    print(f"  {len(statutes)} statutes consolidated")

    print("Consolidating rules...")
    rules = consolidate_rules(data_dir)
    all_documents.extend(rules)
    print(f"  {len(rules)} rules consolidated")

    print("Consolidating TAAs...")
    taas = consolidate_taas(data_dir)
    all_documents.extend(taas)
    print(f"  {len(taas)} TAAs consolidated")

    print("Consolidating cases...")
    cases = consolidate_cases(data_dir)
    all_documents.extend(cases)
    print(f"  {len(cases)} cases consolidated")

    # Build corpus metadata
    by_type = {
        "statute": len(statutes),
        "rule": len(rules),
        "taa": len(taas),
        "case": len(cases),
    }

    corpus_metadata = CorpusMetadata(
        created_at=datetime.now(),
        total_documents=len(all_documents),
        by_type=by_type,
        version="1.0",
    )

    return Corpus(
        metadata=corpus_metadata,
        documents=all_documents,
    )
