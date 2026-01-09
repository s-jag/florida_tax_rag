"""Tests for src/ingestion/consolidate.py."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pytest

from src.ingestion.consolidate import (
    consolidate_all,
    consolidate_case,
    consolidate_cases,
    consolidate_rule,
    consolidate_rules,
    consolidate_statute,
    consolidate_statutes,
    consolidate_taa,
    consolidate_taas,
    load_json_file,
    parse_date,
    parse_datetime,
)
from src.ingestion.models import DocumentType

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_statute_data() -> dict:
    """Sample raw statute data."""
    return {
        "metadata": {
            "section": "212.05",
            "section_name": "Sales Tax Rate",
            "chapter": "212",
            "chapter_name": "Tax on Sales, Use, and Other Transactions",
            "title_number": "14",
            "title": "Taxation and Finance",
            "full_citation": "Fla. Stat. § 212.05",
            "effective_date": "2024-01-01",
            "history": ["New 1-1-90", "Amended 7-1-2024"],
        },
        "text": "The sales tax rate is 6 percent.",
        "source_url": "https://leg.state.fl.us/statutes/212.05",
        "scraped_at": "2024-01-15T10:30:00Z",
    }


@pytest.fixture
def sample_rule_data() -> dict:
    """Sample raw rule data."""
    return {
        "metadata": {
            "rule_number": "12A-1.001",
            "chapter": "12A-1",
            "title": "Specific Exemptions",
            "effective_date": "2024-01-15",
            "rulemaking_authority": ["212.17(6)", "212.18(2)"],
            "law_implemented": ["212.05", "212.08"],
            "references_statutes": ["212.05", "212.08", "213.06"],
        },
        "text": "This rule implements sales tax exemptions.",
        "source_url": "https://flrules.org/12A-1.001",
        "scraped_at": "2024-01-15T12:00:00+00:00",
    }


@pytest.fixture
def sample_taa_data() -> dict:
    """Sample raw TAA data."""
    return {
        "metadata": {
            "taa_number": "TAA 24A-001",
            "title": "Manufacturing Equipment Exemption",
            "issue_date": "2024-01-10",
            "tax_type": "Sales and Use Tax",
            "tax_type_code": "A",
            "topics": ["Exemption", "Manufacturing"],
            "question": "Is manufacturing equipment exempt?",
            "answer": "Yes, under Section 212.08(5).",
            "statutes_cited": ["212.08(5)", "212.05"],
            "rules_cited": ["12A-1.096"],
        },
        "text": "Full TAA text content here.",
        "pdf_path": "/path/to/taa.pdf",
        "source_url": "https://floridarevenue.com/taa",
        "scraped_at": "2024-01-15T14:00:00",
    }


@pytest.fixture
def sample_case_data() -> dict:
    """Sample raw case data."""
    return {
        "metadata": {
            "cluster_id": 1234567,
            "case_name": "DOR v. Taxpayer",
            "case_name_full": "Florida Department of Revenue v. John Taxpayer",
            "citations": ["123 So.3d 456"],
            "court": "Supreme Court of Florida",
            "court_id": "fla",
            "date_filed": "2023-06-15",
            "docket_number": "SC23-1234",
            "judges": "Justice Smith",
            "statutes_cited": ["212.05", "212.08"],
            "cases_cited": [111111, 222222],
        },
        "opinion_text": "The court holds that...",
        "opinion_html": "<p>The court holds that...</p>",
        "pdf_url": "https://storage.courtlistener.com/opinion.pdf",
        "source_url": "https://courtlistener.com/opinion/1234567/",
        "scraped_at": "2024-01-15T16:00:00Z",
    }


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create temp directory structure for testing."""
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    return data_dir


# =============================================================================
# Parse Date Tests
# =============================================================================


class TestParseDate:
    """Test parse_date function."""

    def test_parses_iso_date(self) -> None:
        """Should parse YYYY-MM-DD format."""
        result = parse_date("2024-01-15")
        assert result == date(2024, 1, 15)

    def test_parses_datetime_string(self) -> None:
        """Should parse date from datetime string."""
        result = parse_date("2024-01-15T10:30:00Z")
        assert result == date(2024, 1, 15)

    def test_returns_none_for_none(self) -> None:
        """Should return None for None input."""
        result = parse_date(None)
        assert result is None

    def test_returns_none_for_invalid(self) -> None:
        """Should return None for invalid format."""
        result = parse_date("invalid-date")
        assert result is None

    def test_handles_date_object(self) -> None:
        """Should handle date-like objects via str conversion."""
        result = parse_date("2023-06-15")
        assert result == date(2023, 6, 15)


# =============================================================================
# Parse Datetime Tests
# =============================================================================


class TestParseDatetime:
    """Test parse_datetime function."""

    def test_parses_iso_format(self) -> None:
        """Should parse ISO datetime format."""
        result = parse_datetime("2024-01-15T10:30:00")
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 10

    def test_handles_timezone(self) -> None:
        """Should handle datetime with timezone offset."""
        result = parse_datetime("2024-01-15T12:00:00+00:00")
        assert result.year == 2024
        assert result.month == 1

    def test_handles_z_suffix(self) -> None:
        """Should handle Z timezone suffix."""
        result = parse_datetime("2024-01-15T10:30:00Z")
        assert result.year == 2024

    def test_returns_now_for_none(self) -> None:
        """Should return current time for None."""
        result = parse_datetime(None)
        assert isinstance(result, datetime)

    def test_returns_now_for_invalid(self) -> None:
        """Should return current time for invalid format."""
        result = parse_datetime("not-a-datetime")
        assert isinstance(result, datetime)


# =============================================================================
# Load JSON File Tests
# =============================================================================


class TestLoadJsonFile:
    """Test load_json_file function."""

    def test_loads_valid_json(self, tmp_path: Path) -> None:
        """Should load valid JSON file."""
        json_file = tmp_path / "test.json"
        json_file.write_text('{"key": "value"}')

        result = load_json_file(json_file)

        assert result == {"key": "value"}

    def test_returns_none_for_invalid_json(self, tmp_path: Path) -> None:
        """Should return None for invalid JSON."""
        json_file = tmp_path / "invalid.json"
        json_file.write_text("not valid json {")

        result = load_json_file(json_file)

        assert result is None

    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        """Should return None for missing file."""
        result = load_json_file(tmp_path / "nonexistent.json")
        assert result is None


# =============================================================================
# Consolidate Statute Tests
# =============================================================================


class TestConsolidateStatute:
    """Test consolidate_statute function."""

    def test_creates_legal_document(self, sample_statute_data: dict, tmp_path: Path) -> None:
        """Should create a LegalDocument from statute data."""
        result = consolidate_statute(sample_statute_data, tmp_path / "test.json")

        assert result is not None
        assert result.id == "statute:212.05"
        assert result.doc_type == DocumentType.STATUTE

    def test_sets_title_from_section_name(self, sample_statute_data: dict, tmp_path: Path) -> None:
        """Should use section_name as title."""
        result = consolidate_statute(sample_statute_data, tmp_path / "test.json")

        assert result is not None
        assert result.title == "Sales Tax Rate"

    def test_sets_parent_id_from_chapter(self, sample_statute_data: dict, tmp_path: Path) -> None:
        """Should set parent_id from chapter."""
        result = consolidate_statute(sample_statute_data, tmp_path / "test.json")

        assert result is not None
        assert result.parent_id == "statute:chapter_212"

    def test_parses_effective_date(self, sample_statute_data: dict, tmp_path: Path) -> None:
        """Should parse effective date."""
        result = consolidate_statute(sample_statute_data, tmp_path / "test.json")

        assert result is not None
        assert result.effective_date == date(2024, 1, 1)

    def test_returns_none_for_missing_section(self, tmp_path: Path) -> None:
        """Should return None if section is missing."""
        data = {"metadata": {}, "text": "test"}
        result = consolidate_statute(data, tmp_path / "test.json")
        assert result is None


# =============================================================================
# Consolidate Rule Tests
# =============================================================================


class TestConsolidateRule:
    """Test consolidate_rule function."""

    def test_creates_legal_document(self, sample_rule_data: dict, tmp_path: Path) -> None:
        """Should create a LegalDocument from rule data."""
        result = consolidate_rule(sample_rule_data, tmp_path / "test.json")

        assert result is not None
        assert result.id == "rule:12A-1.001"
        assert result.doc_type == DocumentType.RULE

    def test_merges_statute_citations(self, sample_rule_data: dict, tmp_path: Path) -> None:
        """Should merge all statute citation lists."""
        result = consolidate_rule(sample_rule_data, tmp_path / "test.json")

        assert result is not None
        # Should include citations from rulemaking_authority, law_implemented, references_statutes
        assert "212.05" in result.cites_statutes
        assert "212.08" in result.cites_statutes
        assert "212.17(6)" in result.cites_statutes

    def test_returns_none_for_missing_rule_number(self, tmp_path: Path) -> None:
        """Should return None if rule_number is missing."""
        data = {"metadata": {}, "text": "test"}
        result = consolidate_rule(data, tmp_path / "test.json")
        assert result is None


# =============================================================================
# Consolidate TAA Tests
# =============================================================================


class TestConsolidateTAA:
    """Test consolidate_taa function."""

    def test_creates_legal_document(self, sample_taa_data: dict, tmp_path: Path) -> None:
        """Should create a LegalDocument from TAA data."""
        result = consolidate_taa(sample_taa_data, tmp_path / "test.json")

        assert result is not None
        assert result.id == "taa:TAA_24A-001"
        assert result.doc_type == DocumentType.TAA

    def test_includes_statutes_cited(self, sample_taa_data: dict, tmp_path: Path) -> None:
        """Should include statute citations."""
        result = consolidate_taa(sample_taa_data, tmp_path / "test.json")

        assert result is not None
        assert "212.08(5)" in result.cites_statutes
        assert "212.05" in result.cites_statutes

    def test_includes_rules_cited(self, sample_taa_data: dict, tmp_path: Path) -> None:
        """Should include rule citations."""
        result = consolidate_taa(sample_taa_data, tmp_path / "test.json")

        assert result is not None
        assert "12A-1.096" in result.cites_rules

    def test_normalizes_taa_number(self, sample_taa_data: dict, tmp_path: Path) -> None:
        """Should normalize TAA number (replace spaces with underscores)."""
        result = consolidate_taa(sample_taa_data, tmp_path / "test.json")

        assert result is not None
        assert " " not in result.id

    def test_returns_none_for_missing_taa_number(self, tmp_path: Path) -> None:
        """Should return None if taa_number is missing."""
        data = {"metadata": {}, "text": "test"}
        result = consolidate_taa(data, tmp_path / "test.json")
        assert result is None


# =============================================================================
# Consolidate Case Tests
# =============================================================================


class TestConsolidateCase:
    """Test consolidate_case function."""

    def test_creates_legal_document(self, sample_case_data: dict, tmp_path: Path) -> None:
        """Should create a LegalDocument from case data."""
        result = consolidate_case(sample_case_data, tmp_path / "test.json")

        assert result is not None
        assert result.id == "case:1234567"
        assert result.doc_type == DocumentType.CASE

    def test_uses_opinion_text(self, sample_case_data: dict, tmp_path: Path) -> None:
        """Should use opinion_text as text."""
        result = consolidate_case(sample_case_data, tmp_path / "test.json")

        assert result is not None
        assert "The court holds that" in result.text

    def test_converts_case_citations(self, sample_case_data: dict, tmp_path: Path) -> None:
        """Should convert case IDs to case:ID format."""
        result = consolidate_case(sample_case_data, tmp_path / "test.json")

        assert result is not None
        assert "case:111111" in result.cites_cases
        assert "case:222222" in result.cites_cases

    def test_returns_none_for_missing_cluster_id(self, tmp_path: Path) -> None:
        """Should return None if cluster_id is missing."""
        data = {"metadata": {}, "opinion_text": "test"}
        result = consolidate_case(data, tmp_path / "test.json")
        assert result is None


# =============================================================================
# Batch Consolidation Tests
# =============================================================================


class TestConsolidateStatutes:
    """Test consolidate_statutes function."""

    def test_loads_all_statute_files(self, temp_data_dir: Path, sample_statute_data: dict) -> None:
        """Should load all statute files from chapter directories."""
        statutes_dir = temp_data_dir / "statutes" / "chapter_212"
        statutes_dir.mkdir(parents=True)

        (statutes_dir / "212_05.json").write_text(json.dumps(sample_statute_data))
        sample_statute_data["metadata"]["section"] = "212.08"
        (statutes_dir / "212_08.json").write_text(json.dumps(sample_statute_data))

        result = consolidate_statutes(temp_data_dir)

        assert len(result) == 2

    def test_returns_empty_for_missing_dir(self, temp_data_dir: Path) -> None:
        """Should return empty list if statutes directory doesn't exist."""
        result = consolidate_statutes(temp_data_dir)
        assert result == []


class TestConsolidateRules:
    """Test consolidate_rules function."""

    def test_loads_all_rule_files(self, temp_data_dir: Path, sample_rule_data: dict) -> None:
        """Should load all rule files from chapter directories."""
        rules_dir = temp_data_dir / "admin_code" / "chapter_12A_1"
        rules_dir.mkdir(parents=True)

        (rules_dir / "rule_12A-1_001.json").write_text(json.dumps(sample_rule_data))

        result = consolidate_rules(temp_data_dir)

        assert len(result) == 1

    def test_returns_empty_for_missing_dir(self, temp_data_dir: Path) -> None:
        """Should return empty list if admin_code directory doesn't exist."""
        result = consolidate_rules(temp_data_dir)
        assert result == []


class TestConsolidateTAAs:
    """Test consolidate_taas function."""

    def test_loads_all_taa_files(self, temp_data_dir: Path, sample_taa_data: dict) -> None:
        """Should load all TAA files matching pattern."""
        taa_dir = temp_data_dir / "taa"
        taa_dir.mkdir(parents=True)

        (taa_dir / "TAA_24A-001.json").write_text(json.dumps(sample_taa_data))
        (taa_dir / "TAA_24A-002.json").write_text(json.dumps(sample_taa_data))

        result = consolidate_taas(temp_data_dir)

        # Both files should be loaded (same TAA number but that's ok for test)
        assert len(result) == 2


class TestConsolidateCases:
    """Test consolidate_cases function."""

    def test_loads_all_case_files(self, temp_data_dir: Path, sample_case_data: dict) -> None:
        """Should load all case files matching pattern."""
        case_dir = temp_data_dir / "case_law"
        case_dir.mkdir(parents=True)

        (case_dir / "case_1234567.json").write_text(json.dumps(sample_case_data))

        result = consolidate_cases(temp_data_dir)

        assert len(result) == 1


# =============================================================================
# Consolidate All Tests
# =============================================================================


class TestConsolidateAll:
    """Test consolidate_all function."""

    def test_creates_corpus(
        self,
        temp_data_dir: Path,
        sample_statute_data: dict,
        sample_rule_data: dict,
    ) -> None:
        """Should create a Corpus with all document types."""
        # Create statute
        statutes_dir = temp_data_dir / "statutes" / "chapter_212"
        statutes_dir.mkdir(parents=True)
        (statutes_dir / "212_05.json").write_text(json.dumps(sample_statute_data))

        # Create rule
        rules_dir = temp_data_dir / "admin_code" / "chapter_12A_1"
        rules_dir.mkdir(parents=True)
        (rules_dir / "rule_12A-1_001.json").write_text(json.dumps(sample_rule_data))

        result = consolidate_all(temp_data_dir)

        assert result.metadata.total_documents == 2
        assert result.metadata.by_type["statute"] == 1
        assert result.metadata.by_type["rule"] == 1
        assert len(result.documents) == 2

    def test_handles_empty_data_dir(self, temp_data_dir: Path) -> None:
        """Should handle empty data directory."""
        result = consolidate_all(temp_data_dir)

        assert result.metadata.total_documents == 0
        assert len(result.documents) == 0

    def test_handles_malformed_json(self, temp_data_dir: Path, sample_statute_data: dict) -> None:
        """Should skip malformed JSON files."""
        statutes_dir = temp_data_dir / "statutes" / "chapter_212"
        statutes_dir.mkdir(parents=True)

        # Valid file
        (statutes_dir / "212_05.json").write_text(json.dumps(sample_statute_data))
        # Invalid file
        (statutes_dir / "212_08.json").write_text("not valid json {")

        result = consolidate_all(temp_data_dir)

        # Should only include the valid file
        assert result.metadata.total_documents == 1
