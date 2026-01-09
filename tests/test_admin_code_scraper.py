"""Tests for src/scrapers/admin_code.py."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from bs4 import BeautifulSoup

from src.scrapers.admin_code import FloridaAdminCodeScraper
from src.scrapers.models import RawRule, RuleMetadata

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Create temporary directories for testing."""
    cache_dir = tmp_path / "cache"
    raw_data_dir = tmp_path / "raw"
    cache_dir.mkdir()
    raw_data_dir.mkdir()
    return cache_dir, raw_data_dir


@pytest.fixture
def scraper(temp_dirs: tuple[Path, Path]) -> FloridaAdminCodeScraper:
    """Create a scraper instance with temp directories."""
    cache_dir, raw_data_dir = temp_dirs
    return FloridaAdminCodeScraper(
        rate_limit_delay=0.01,
        timeout=5.0,
        use_cache=True,
        cache_dir=cache_dir,
        raw_data_dir=raw_data_dir,
    )


@pytest.fixture
def sample_division_html() -> str:
    """Sample HTML for division page with chapter links."""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Division 12A</title></head>
    <body>
        <table>
            <tr>
                <td>
                    <a href="ChapterHome.asp?Chapter=12A-1">SALES AND USE TAX</a>
                </td>
            </tr>
            <tr>
                <td>
                    <a href="ChapterHome.asp?Chapter=12A-15">TAX SURTAX AND FEES</a>
                </td>
            </tr>
            <tr>
                <td>
                    <a href="ChapterHome.asp?Chapter=12A-19">COMMUNICATIONS SERVICES TAX</a>
                </td>
            </tr>
        </table>
    </body>
    </html>
    """


@pytest.fixture
def sample_chapter_html() -> str:
    """Sample HTML for chapter page with rule links."""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>Chapter 12A-1</title></head>
    <body>
        <table>
            <tr>
                <td><a href="RuleNo.asp?ID=12A-1.001">12A-1.001 Specific Exemptions</a></td>
            </tr>
            <tr>
                <td><a href="RuleNo.asp?ID=12A-1.002">12A-1.002 Sales of Food Products</a></td>
            </tr>
            <tr>
                <td><a href="RuleNo.asp?ID=12A-1.001">12A-1.001 Specific Exemptions</a></td>
            </tr>
            <tr>
                <td><a href="RuleNo.asp?ID=12A-1.003">12A-1.003 Industrial Machinery</a></td>
            </tr>
        </table>
    </body>
    </html>
    """


@pytest.fixture
def sample_rule_html() -> str:
    """Sample HTML for a rule page."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>12A-1.001 : Specific Exemptions - Florida Administrative Rules</title>
    </head>
    <body>
        <table width="100%">
            <tr>
                <td>Rule Title: Specific Exemptions</td>
            </tr>
            <tr>
                <td>
                    <p>(1) The following sales are exempt from sales tax:</p>
                    <p>(a) Sales of food products as defined in Section 212.08(1), F.S.</p>
                    <p>(b) Sales of prescription drugs pursuant to Section 212.08(2), F.S.</p>
                    <p>(2) To claim an exemption, the purchaser must provide a valid exemption certificate.</p>
                </td>
            </tr>
        </table>
        <p>Effective Date: 01/15/2024</p>
        <p>Rulemaking Authority 212.17(6), 212.18(2), 213.06(1) FS. Law Implemented 212.05, 212.08, 212.085 FS. History–New 1-1-90, Amended 6-19-01, 1-15-24.</p>
        <a href="statute.asp?id=212.05">212.05</a>
        <a href="statute.asp?id=212.08">212.08</a>
    </body>
    </html>
    """


# =============================================================================
# Initialization Tests
# =============================================================================


class TestScraperInit:
    """Test scraper initialization."""

    def test_creates_output_dir(self, temp_dirs: tuple[Path, Path]) -> None:
        """Scraper should create admin_code output directory."""
        cache_dir, raw_data_dir = temp_dirs
        FloridaAdminCodeScraper(
            cache_dir=cache_dir,
            raw_data_dir=raw_data_dir,
        )
        assert (raw_data_dir / "admin_code").exists()

    def test_tax_divisions_defined(self, scraper: FloridaAdminCodeScraper) -> None:
        """TAX_DIVISIONS should contain expected divisions."""
        assert "12A" in scraper.TAX_DIVISIONS
        assert "12B" in scraper.TAX_DIVISIONS
        assert "12C" in scraper.TAX_DIVISIONS
        assert "12D" in scraper.TAX_DIVISIONS


# =============================================================================
# Division/Chapter Parsing Tests
# =============================================================================


class TestGetDivisionChapters:
    """Test get_division_chapters method."""

    async def test_parses_chapter_links(
        self, scraper: FloridaAdminCodeScraper, sample_division_html: str
    ) -> None:
        """Should parse chapter links from division page."""
        with patch.object(scraper, "fetch_page", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = sample_division_html

            chapters = await scraper.get_division_chapters("12A")

        assert len(chapters) == 3
        assert chapters[0]["chapter"] == "12A-1"
        assert chapters[0]["title"] == "SALES AND USE TAX"
        assert chapters[1]["chapter"] == "12A-15"
        assert chapters[2]["chapter"] == "12A-19"

    async def test_handles_empty_division(self, scraper: FloridaAdminCodeScraper) -> None:
        """Should return empty list for division with no chapters."""
        empty_html = "<html><body><p>No chapters found</p></body></html>"
        with patch.object(scraper, "fetch_page", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = empty_html

            chapters = await scraper.get_division_chapters("12X")

        assert chapters == []


class TestGetChapterRules:
    """Test get_chapter_rules method."""

    async def test_parses_rule_links(
        self, scraper: FloridaAdminCodeScraper, sample_chapter_html: str
    ) -> None:
        """Should parse rule links from chapter page."""
        with patch.object(scraper, "fetch_page", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = sample_chapter_html

            rules = await scraper.get_chapter_rules("12A-1")

        # Should deduplicate (12A-1.001 appears twice)
        assert len(rules) == 3
        rule_numbers = [r["rule_number"] for r in rules]
        assert "12A-1.001" in rule_numbers
        assert "12A-1.002" in rule_numbers
        assert "12A-1.003" in rule_numbers

    async def test_deduplicates_rules(
        self, scraper: FloridaAdminCodeScraper, sample_chapter_html: str
    ) -> None:
        """Should deduplicate rule links."""
        with patch.object(scraper, "fetch_page", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = sample_chapter_html

            rules = await scraper.get_chapter_rules("12A-1")

        # Count occurrences of 12A-1.001
        count = sum(1 for r in rules if r["rule_number"] == "12A-1.001")
        assert count == 1


# =============================================================================
# Title Extraction Tests
# =============================================================================


class TestExtractRuleTitle:
    """Test _extract_rule_title method."""

    def test_extracts_from_table_cell(self, scraper: FloridaAdminCodeScraper) -> None:
        """Should extract title from 'Rule Title:' table cell."""
        html = """
        <html><body>
            <table>
                <tr><td>Rule Title: Sales Tax Exemptions</td></tr>
            </table>
        </body></html>
        """
        soup = BeautifulSoup(html, "lxml")
        title = scraper._extract_rule_title(soup, "12A-1.001")
        assert title == "Sales Tax Exemptions"

    def test_extracts_from_page_title(self, scraper: FloridaAdminCodeScraper) -> None:
        """Should extract title from page title element."""
        html = """
        <html>
            <head><title>12A-1.001 : Specific Exemptions - Florida Administrative Rules</title></head>
            <body></body>
        </html>
        """
        soup = BeautifulSoup(html, "lxml")
        title = scraper._extract_rule_title(soup, "12A-1.001")
        assert title == "Specific Exemptions"

    def test_returns_empty_for_no_title(self, scraper: FloridaAdminCodeScraper) -> None:
        """Should return empty string when no title found."""
        html = "<html><body><p>No title here</p></body></html>"
        soup = BeautifulSoup(html, "lxml")
        title = scraper._extract_rule_title(soup, "12A-1.001")
        assert title == ""


# =============================================================================
# Rule Text Extraction Tests
# =============================================================================


class TestExtractRuleText:
    """Test _extract_rule_text method."""

    def test_extracts_from_main_table(self, scraper: FloridaAdminCodeScraper) -> None:
        """Should extract text from main content table."""
        html = """
        <html><body>
            <table width="100%">
                <tr>
                    <td>
                        (1) This is the first paragraph of the rule that has
                        enough content to be detected as the main text block.
                        It contains important information about tax exemptions.
                        More text here to make it long enough.
                    </td>
                </tr>
            </table>
        </body></html>
        """
        soup = BeautifulSoup(html, "lxml")
        text = scraper._extract_rule_text(soup)
        assert "(1) This is the first paragraph" in text

    def test_removes_script_and_style(self, scraper: FloridaAdminCodeScraper) -> None:
        """Should remove script and style elements."""
        html = """
        <html><body>
            <script>alert('bad');</script>
            <style>.hidden { display: none; }</style>
            <p>Rule content here with enough text to be considered valid content
            for extraction purposes in the test.</p>
        </body></html>
        """
        soup = BeautifulSoup(html, "lxml")
        text = scraper._extract_rule_text(soup)
        assert "alert" not in text
        assert "hidden" not in text


# =============================================================================
# Date Extraction Tests
# =============================================================================


class TestExtractEffectiveDate:
    """Test _extract_effective_date method."""

    def test_extracts_date_slash_format(self, scraper: FloridaAdminCodeScraper) -> None:
        """Should extract date in MM/DD/YYYY format."""
        html = "<html><body><p>Effective Date: 01/15/2024</p></body></html>"
        soup = BeautifulSoup(html, "lxml")
        result = scraper._extract_effective_date(soup)
        assert result == date(2024, 1, 15)

    def test_extracts_date_dash_format(self, scraper: FloridaAdminCodeScraper) -> None:
        """Should extract date in MM-DD-YYYY format."""
        html = "<html><body><p>Effective Date: 06-19-2023</p></body></html>"
        soup = BeautifulSoup(html, "lxml")
        result = scraper._extract_effective_date(soup)
        assert result == date(2023, 6, 19)

    def test_returns_none_for_no_date(self, scraper: FloridaAdminCodeScraper) -> None:
        """Should return None when no date found."""
        html = "<html><body><p>No date here</p></body></html>"
        soup = BeautifulSoup(html, "lxml")
        result = scraper._extract_effective_date(soup)
        assert result is None

    def test_handles_various_formats(self, scraper: FloridaAdminCodeScraper) -> None:
        """Should handle different text formats."""
        test_cases = [
            ("Effective: 12/25/2023", date(2023, 12, 25)),
            ("Effective Date 03/01/2022", date(2022, 3, 1)),
        ]
        for text, expected in test_cases:
            html = f"<html><body>{text}</body></html>"
            soup = BeautifulSoup(html, "lxml")
            result = scraper._extract_effective_date(soup)
            assert result == expected, f"Failed for: {text}"


# =============================================================================
# Statute Citation Extraction Tests
# =============================================================================


class TestExtractRulemakingAuthority:
    """Test _extract_rulemaking_authority method."""

    def test_extracts_citations(self, scraper: FloridaAdminCodeScraper) -> None:
        """Should extract statute citations from Rulemaking Authority section."""
        html = """
        <html><body>
            <p>Rulemaking Authority 212.17(6), 212.18(2), 213.06(1) FS.
            Law Implemented 212.05, 212.08 FS.</p>
        </body></html>
        """
        soup = BeautifulSoup(html, "lxml")
        citations = scraper._extract_rulemaking_authority(soup)
        assert "212.17(6)" in citations
        assert "212.18(2)" in citations


class TestExtractLawImplemented:
    """Test _extract_law_implemented method."""

    def test_extracts_citations(self, scraper: FloridaAdminCodeScraper) -> None:
        """Should extract statute citations from Law Implemented section."""
        html = """
        <html><body>
            <p>Rulemaking Authority 212.17(6) FS.
            Law Implemented 212.05, 212.08, 212.085 FS.
            History–New 1-1-90.</p>
        </body></html>
        """
        soup = BeautifulSoup(html, "lxml")
        citations = scraper._extract_law_implemented(soup)
        assert "212.05" in citations
        assert "212.08" in citations


class TestExtractStatuteCitations:
    """Test _extract_statute_citations method."""

    def test_extracts_from_links(
        self, scraper: FloridaAdminCodeScraper, sample_rule_html: str
    ) -> None:
        """Should extract citations from statute links."""
        soup = BeautifulSoup(sample_rule_html, "lxml")
        citations = scraper._extract_statute_citations(soup, "Law Implemented")
        assert "212.05" in citations
        assert "212.08" in citations


class TestParseStatuteCitations:
    """Test _parse_statute_citations method."""

    def test_parses_simple_citations(self, scraper: FloridaAdminCodeScraper) -> None:
        """Should parse simple statute citations."""
        text = "212.05, 212.08, 213.06"
        citations = scraper._parse_statute_citations(text)
        assert "212.05" in citations
        assert "212.08" in citations
        assert "213.06" in citations

    def test_parses_citations_with_subsections(self, scraper: FloridaAdminCodeScraper) -> None:
        """Should parse citations with subsections."""
        text = "212.08(7)(f), 212.17(6), 213.06(1)"
        citations = scraper._parse_statute_citations(text)
        assert "212.08(7)(f)" in citations
        assert "212.17(6)" in citations

    def test_parses_complex_subsections(self, scraper: FloridaAdminCodeScraper) -> None:
        """Should parse complex subsection formats."""
        text = "212.08(7)(h)2."
        citations = scraper._parse_statute_citations(text)
        assert len(citations) >= 1


# =============================================================================
# History Extraction Tests
# =============================================================================


class TestExtractHistory:
    """Test _extract_history method."""

    def test_extracts_history(self, scraper: FloridaAdminCodeScraper) -> None:
        """Should extract history section."""
        html = """
        <html><body>
            <p>History– New 1-1-90, Amended 6-19-01, 1-15-24.</p>
        </body></html>
        """
        soup = BeautifulSoup(html, "lxml")
        history = scraper._extract_history(soup)
        assert "New 1-1-90" in history
        assert "Amended" in history

    def test_returns_empty_for_no_history(self, scraper: FloridaAdminCodeScraper) -> None:
        """Should return empty string when no history found."""
        html = "<html><body><p>No history</p></body></html>"
        soup = BeautifulSoup(html, "lxml")
        history = scraper._extract_history(soup)
        assert history == ""


# =============================================================================
# Text Cleaning Tests
# =============================================================================


class TestCleanText:
    """Test _clean_text method."""

    def test_removes_excessive_whitespace(self, scraper: FloridaAdminCodeScraper) -> None:
        """Should remove excessive whitespace."""
        text = "Line 1\n\n\n\nLine 2"
        result = scraper._clean_text(text)
        # The clean_text method collapses 4+ newlines to 2
        assert "\n\n\n\n" not in result
        assert "Line 1" in result
        assert "Line 2" in result

    def test_removes_navigation_elements(self, scraper: FloridaAdminCodeScraper) -> None:
        """Should remove navigation elements."""
        text = "Content here\nPrevious Up Next\nMore content"
        result = scraper._clean_text(text)
        assert "Previous" not in result
        assert "Content here" in result

    def test_removes_multiple_spaces(self, scraper: FloridaAdminCodeScraper) -> None:
        """Should collapse multiple spaces."""
        text = "Word1    Word2     Word3"
        result = scraper._clean_text(text)
        assert result == "Word1 Word2 Word3"


# =============================================================================
# Rule Scraping Tests
# =============================================================================


class TestScrapeRule:
    """Test scrape_rule method."""

    async def test_returns_raw_rule(
        self, scraper: FloridaAdminCodeScraper, sample_rule_html: str
    ) -> None:
        """Should return a RawRule object."""
        with patch.object(scraper, "fetch_page", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = sample_rule_html

            rule = await scraper.scrape_rule("12A-1.001", "Specific Exemptions")

        assert isinstance(rule, RawRule)
        assert rule.metadata.rule_number == "12A-1.001"
        assert rule.source_url is not None

    async def test_extracts_metadata(
        self, scraper: FloridaAdminCodeScraper, sample_rule_html: str
    ) -> None:
        """Should extract rule metadata."""
        with patch.object(scraper, "fetch_page", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = sample_rule_html

            rule = await scraper.scrape_rule("12A-1.001")

        assert rule.metadata.title == "Specific Exemptions"
        assert rule.metadata.effective_date == date(2024, 1, 15)

    async def test_extracts_chapter(
        self, scraper: FloridaAdminCodeScraper, sample_rule_html: str
    ) -> None:
        """Should extract chapter from rule number."""
        with patch.object(scraper, "fetch_page", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = sample_rule_html

            rule = await scraper.scrape_rule("12A-1.001")

        assert rule.metadata.chapter == "12A-1"


# =============================================================================
# Chapter Scraping Tests
# =============================================================================


class TestScrapeChapter:
    """Test scrape_chapter method."""

    async def test_scrapes_all_rules(self, scraper: FloridaAdminCodeScraper) -> None:
        """Should scrape all rules in a chapter."""
        rules_list = [
            {"rule_number": "12A-1.001", "title": "Exemptions"},
            {"rule_number": "12A-1.002", "title": "Food Products"},
        ]

        mock_rule = MagicMock(spec=RawRule)
        mock_rule.metadata = MagicMock(spec=RuleMetadata)
        mock_rule.metadata.rule_number = "12A-1.001"
        mock_rule.metadata.chapter = "12A-1"
        mock_rule.model_dump_json.return_value = "{}"

        with patch.object(scraper, "get_chapter_rules", new_callable=AsyncMock) as mock_get_rules:
            mock_get_rules.return_value = rules_list

            with patch.object(scraper, "scrape_rule", new_callable=AsyncMock) as mock_scrape:
                mock_scrape.return_value = mock_rule

                rules = await scraper.scrape_chapter("12A-1", delay=0.001)

        assert len(rules) == 2
        assert mock_scrape.call_count == 2


# =============================================================================
# Full Scrape Tests
# =============================================================================


class TestScrape:
    """Test main scrape method."""

    async def test_scrapes_specified_divisions(self, scraper: FloridaAdminCodeScraper) -> None:
        """Should scrape only specified divisions."""
        mock_rule = MagicMock(spec=RawRule)
        mock_rule.metadata = MagicMock(spec=RuleMetadata)
        mock_rule.metadata.rule_number = "12A-1.001"
        mock_rule.metadata.chapter = "12A-1"
        mock_rule.model_dump.return_value = {"test": "data"}
        mock_rule.model_dump_json.return_value = "{}"

        with patch.object(scraper, "scrape_division", new_callable=AsyncMock) as mock_scrape_div:
            mock_scrape_div.return_value = [mock_rule]

            with patch.object(scraper, "_save_combined_output"):
                results = await scraper.scrape(divisions=["12A"], delay=0.001)

        assert len(results) == 1
        mock_scrape_div.assert_called_once()
        call_args = mock_scrape_div.call_args
        assert call_args[0][0] == "12A"

    async def test_warns_unknown_division(self, scraper: FloridaAdminCodeScraper) -> None:
        """Should warn for unknown divisions."""
        with patch.object(scraper, "scrape_division", new_callable=AsyncMock) as mock_scrape_div:
            mock_scrape_div.return_value = []

            with patch.object(scraper, "_save_combined_output"):
                await scraper.scrape(divisions=["12Z"], delay=0.001)

        # Should not call scrape_division for unknown division
        mock_scrape_div.assert_not_called()


# =============================================================================
# Save Tests
# =============================================================================


class TestSaveRule:
    """Test _save_rule method."""

    def test_creates_chapter_directory(self, scraper: FloridaAdminCodeScraper) -> None:
        """Should create chapter subdirectory."""
        mock_rule = MagicMock(spec=RawRule)
        mock_rule.metadata = MagicMock()
        mock_rule.metadata.rule_number = "12A-1.001"
        mock_rule.model_dump_json.return_value = '{"test": "data"}'

        path = scraper._save_rule(mock_rule, "12A-1")

        assert path.parent.name == "chapter_12A_1"
        assert path.exists()

    def test_saves_rule_json(self, scraper: FloridaAdminCodeScraper) -> None:
        """Should save rule as JSON."""
        mock_rule = MagicMock(spec=RawRule)
        mock_rule.metadata = MagicMock()
        mock_rule.metadata.rule_number = "12A-1.001"
        mock_rule.model_dump_json.return_value = '{"test": "data"}'

        path = scraper._save_rule(mock_rule, "12A-1")

        content = path.read_text()
        assert content == '{"test": "data"}'


class TestSaveCombinedOutput:
    """Test _save_combined_output method."""

    def test_saves_all_rules_json(self, scraper: FloridaAdminCodeScraper) -> None:
        """Should save all rules as JSON array."""
        mock_rule = MagicMock(spec=RawRule)
        mock_rule.metadata = MagicMock()
        mock_rule.metadata.chapter = "12A-1"
        mock_rule.metadata.rule_number = "12A-1.001"
        mock_rule.model_dump_json.return_value = '{"rule": "data"}'

        scraper._save_combined_output([mock_rule])

        output_path = scraper.output_dir / "all_rules.json"
        assert output_path.exists()
        content = output_path.read_text()
        assert '{"rule": "data"}' in content
