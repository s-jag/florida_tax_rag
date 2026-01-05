"""Tests for the Florida Statutes scraper with mocked HTTP responses."""

from __future__ import annotations

import asyncio
import pytest

# Configure pytest-asyncio
pytest_plugins = ("pytest_asyncio",)
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import date

from src.scrapers.statutes import (
    FloridaStatutesScraper,
    ChapterInfo,
    SectionInfo,
    TITLES,
    BASE_URL,
)


# Sample HTML fixtures
SAMPLE_TITLE_INDEX_HTML = """
<!DOCTYPE html>
<html>
<head><title>Florida Statutes</title></head>
<body>
<h1>Title XIV - TAXATION AND FINANCE</h1>
<p>
<a href="index.cfm?App_mode=Display_Statute&URL=0100-0199/0192/0192ContentsIndex.html">
Chapter 192
</a>
TAXATION: GENERAL PROVISIONS
</p>
<p>
<a href="index.cfm?App_mode=Display_Statute&URL=0200-0299/0212/0212ContentsIndex.html">
Chapter 212
</a>
TAX ON SALES, USE, AND OTHER TRANSACTIONS
</p>
<p>
<a href="index.cfm?App_mode=Display_Statute&URL=0200-0299/0220/0220ContentsIndex.html">
Chapter 220
</a>
INCOME TAX CODE
</p>
</body>
</html>
"""

SAMPLE_CHAPTER_CONTENTS_HTML = """
<!DOCTYPE html>
<html>
<head><title>Chapter 212</title></head>
<body>
<h1>Chapter 212 - TAX ON SALES, USE, AND OTHER TRANSACTIONS</h1>
<ul>
<li>
<a href="index.cfm?App_mode=Display_Statute&URL=0200-0299/0212/Sections/0212.01.html">
212.01
</a>
Short title.
</li>
<li>
<a href="index.cfm?App_mode=Display_Statute&URL=0200-0299/0212/Sections/0212.02.html">
212.02
</a>
Definitions.
</li>
<li>
<a href="index.cfm?App_mode=Display_Statute&URL=0200-0299/0212/Sections/0212.05.html">
212.05
</a>
Sales, storage, use tax.
</li>
<li>
<a href="index.cfm?App_mode=Display_Statute&URL=0200-0299/0212/Sections/0212.08.html">
212.08
</a>
Exemptions.
</li>
</ul>
</body>
</html>
"""

SAMPLE_SECTION_HTML = """
<!DOCTYPE html>
<html>
<head><title>F.S. 212.05</title></head>
<body>
<div class="Statute">
<h1>212.05 Sales, storage, use tax.—</h1>

<p>It is hereby declared to be the legislative intent that every person
is exercising a taxable privilege who engages in the business of selling
tangible personal property at retail in this state.</p>

<p>(1) For the exercise of such privilege, a tax is levied on each
taxable transaction or incident, which tax is due and payable as follows:</p>

<p>(a) At the rate of 6 percent of the sales price of each item or
article of tangible personal property when sold at retail in this state.</p>

<p>(b) At the rate of 6 percent of the cost price of each item or
article of tangible personal property when the same is not sold but
is used, consumed, distributed, or stored for use or consumption in
this state.</p>

<p>(2) The tax imposed by this chapter shall be collected by the
dealer from the purchaser or consumer.</p>

<p>History.—s. 5, ch. 26319, 1949; s. 1, ch. 57-398; s. 1, ch. 63-253;
s. 1, ch. 67-180; s. 1, ch. 69-222; s. 1, ch. 71-360; s. 2, ch. 72-277;
s. 1, ch. 77-460; s. 6, ch. 81-259; s. 39, ch. 83-217; s. 2, ch. 84-356;
s. 5, ch. 85-342; s. 54, ch. 87-6; s. 26, ch. 87-101; s. 62, ch. 87-548;
s. 9, ch. 88-119; s. 6, ch. 92-319; s. 5, ch. 93-233; s. 1, ch. 95-416;
s. 6, ch. 96-397; s. 5, ch. 97-99; s. 1, ch. 2000-170; s. 1, ch. 2000-355;
s. 40, ch. 2001-254; s. 2, ch. 2003-254; s. 1, ch. 2005-280; s. 30, ch. 2010-147.</p>

<p>Note.—Former s. 212.02.</p>
</div>
</body>
</html>
"""

SAMPLE_SECTION_WITH_CROSS_REFS_HTML = """
<!DOCTYPE html>
<html>
<head><title>F.S. 212.08</title></head>
<body>
<div class="Statute">
<h1>212.08 Exemptions.—</h1>

<p>(1) The sale of the following is exempt from the tax imposed by this chapter:</p>

<p>(a) Items specifically exempt under s. 212.05, F.S., or s. 212.031.</p>

<p>(b) Any transaction exempt under § 212.06 or Rule 12A-1.005, F.A.C.</p>

<p>(7)(a) Agricultural exemptions.—There are exempt from the tax imposed
by this chapter, if used exclusively on a farm or in a forest:</p>

<p>History.—s. 6, ch. 26319, 1949; s. 3, ch. 57-398; s. 2, ch. 2024-99.</p>
</div>
</body>
</html>
"""


class TestFloridaStatutesScraperURLBuilding:
    """Tests for URL building methods."""

    def test_get_chapter_range_path_100s(self):
        """Test range path for chapters in the 100s."""
        scraper = FloridaStatutesScraper(use_cache=False)
        assert scraper._get_chapter_range_path(192) == "0100-0199/0192"
        assert scraper._get_chapter_range_path(199) == "0100-0199/0199"

    def test_get_chapter_range_path_200s(self):
        """Test range path for chapters in the 200s."""
        scraper = FloridaStatutesScraper(use_cache=False)
        assert scraper._get_chapter_range_path(212) == "0200-0299/0212"
        assert scraper._get_chapter_range_path(220) == "0200-0299/0220"

    def test_int_to_roman(self):
        """Test Roman numeral conversion."""
        scraper = FloridaStatutesScraper(use_cache=False)
        assert scraper._int_to_roman(14) == "XIV"
        assert scraper._int_to_roman(1) == "I"
        assert scraper._int_to_roman(4) == "IV"
        assert scraper._int_to_roman(9) == "IX"

    def test_build_title_index_url(self):
        """Test title index URL building."""
        scraper = FloridaStatutesScraper(use_cache=False)
        url = scraper._build_title_index_url(14)
        assert "App_mode=Display_Index" in url
        assert "Title_Request=XIV" in url

    def test_build_chapter_contents_url(self):
        """Test chapter contents URL building."""
        scraper = FloridaStatutesScraper(use_cache=False)
        url = scraper._build_chapter_contents_url(212)
        assert "0200-0299/0212" in url
        assert "0212ContentsIndex.html" in url


class TestFloridaStatutesScraperParsing:
    """Tests for HTML parsing methods."""

    @pytest.fixture
    def scraper(self, tmp_path):
        """Create a scraper instance with temp output dir."""
        return FloridaStatutesScraper(
            use_cache=False,
            output_dir=tmp_path / "statutes",
        )

    @pytest.mark.asyncio
    async def test_get_title_chapters(self, scraper):
        """Test parsing chapters from title index page."""
        with patch.object(scraper, "fetch_page", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = SAMPLE_TITLE_INDEX_HTML

            chapters = await scraper.get_title_chapters(14)

            assert len(chapters) == 3
            assert chapters[0].chapter_number == 192
            assert chapters[1].chapter_number == 212
            assert chapters[2].chapter_number == 220

            assert "TAXATION: GENERAL PROVISIONS" in chapters[0].chapter_name
            assert "TAX ON SALES" in chapters[1].chapter_name

    @pytest.mark.asyncio
    async def test_get_chapter_sections(self, scraper):
        """Test parsing sections from chapter contents page."""
        with patch.object(scraper, "fetch_page", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = SAMPLE_CHAPTER_CONTENTS_HTML

            chapter_info = ChapterInfo(
                chapter_number=212,
                chapter_name="TAX ON SALES, USE, AND OTHER TRANSACTIONS",
                contents_url="http://example.com/chapter212",
            )

            sections = await scraper.get_chapter_sections(chapter_info)

            assert len(sections) == 4
            assert sections[0].section_number == "212.01"
            assert sections[1].section_number == "212.02"
            assert sections[2].section_number == "212.05"
            assert sections[3].section_number == "212.08"

            assert "Short title" in sections[0].section_name
            assert "Definitions" in sections[1].section_name

    @pytest.mark.asyncio
    async def test_scrape_section(self, scraper):
        """Test scraping a single section."""
        with patch.object(scraper, "fetch_page", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = SAMPLE_SECTION_HTML

            section_info = SectionInfo(
                section_number="212.05",
                section_name="Sales, storage, use tax.",
                section_url="http://example.com/212.05",
                chapter_number=212,
            )

            statute = await scraper.scrape_section(section_info)

            assert statute.metadata.section == "212.05"
            assert statute.metadata.chapter == 212
            assert statute.metadata.title_number == 14
            assert "taxable privilege" in statute.text
            assert "6 percent" in statute.text

    def test_extract_history_years(self, scraper):
        """Test extracting amendment years from history text."""
        history = (
            "History.—s. 5, ch. 26319, 1949; s. 1, ch. 57-398; "
            "s. 2, ch. 2024-99."
        )
        years = scraper._extract_history_years(history)

        assert "1949" in years
        assert "2024" in years

    def test_parse_section_text_with_history(self, scraper):
        """Test parsing section text and separating history."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(SAMPLE_SECTION_HTML, "lxml")
        main_text, history_text = scraper._parse_section_text(soup)

        assert "taxable privilege" in main_text
        assert "6 percent" in main_text
        assert "History.—" in history_text
        assert "ch. 26319, 1949" in history_text


class TestFloridaStatutesScraperSaving:
    """Tests for data saving methods."""

    @pytest.fixture
    def scraper(self, tmp_path):
        """Create a scraper instance with temp output dir."""
        return FloridaStatutesScraper(
            use_cache=False,
            output_dir=tmp_path / "statutes",
        )

    @pytest.mark.asyncio
    async def test_save_section_creates_file(self, scraper):
        """Test that _save_section creates the expected file."""
        with patch.object(scraper, "fetch_page", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = SAMPLE_SECTION_HTML

            section_info = SectionInfo(
                section_number="212.05",
                section_name="Sales, storage, use tax.",
                section_url="http://example.com/212.05",
                chapter_number=212,
            )

            statute = await scraper.scrape_section(section_info)
            scraper._save_section(statute, 212)

            expected_file = scraper.output_dir / "chapter_212" / "section_212.05.json"
            assert expected_file.exists()


class TestFloridaStatutesScraperIntegration:
    """Integration tests with mocked HTTP."""

    @pytest.fixture
    def scraper(self, tmp_path):
        """Create a scraper instance with temp output dir."""
        return FloridaStatutesScraper(
            use_cache=False,
            output_dir=tmp_path / "statutes",
            rate_limit_delay=0.01,  # Fast for tests
        )

    @pytest.mark.asyncio
    async def test_scrape_chapter_workflow(self, scraper):
        """Test the full chapter scraping workflow."""
        call_urls = []

        async def mock_fetch(url):
            call_urls.append(url)

            if "ContentsIndex" in url or url == "http://example.com/212":
                return SAMPLE_CHAPTER_CONTENTS_HTML
            elif "212.05" in url:
                return SAMPLE_SECTION_HTML
            elif "212.08" in url:
                return SAMPLE_SECTION_WITH_CROSS_REFS_HTML
            else:
                # Return a minimal valid section
                return """
                <html><body>
                <h1>212.XX Section Title.—</h1>
                <p>Section text here.</p>
                <p>History.—s. 1, ch. 2020-1.</p>
                </body></html>
                """

        # Create a proper AsyncMock that calls our function
        mock = AsyncMock(side_effect=mock_fetch)

        with patch.object(scraper, "fetch_page", mock):
            chapter_info = ChapterInfo(
                chapter_number=212,
                chapter_name="TAX ON SALES, USE, AND OTHER TRANSACTIONS",
                contents_url="http://example.com/212",
            )

            statutes = await scraper.scrape_chapter(chapter_info)

            # Should have scraped all 4 sections from SAMPLE_CHAPTER_CONTENTS_HTML
            assert len(statutes) == 4, f"Expected 4 statutes, got {len(statutes)}. URLs called: {call_urls}"
            assert all(s.metadata.chapter == 212 for s in statutes)
            assert len(call_urls) >= 5  # 1 contents + 4 sections


class TestTitleMetadata:
    """Tests for title metadata constants."""

    def test_title_xiv_exists(self):
        """Test that Title XIV is defined."""
        assert 14 in TITLES
        assert TITLES[14]["name"] == "TAXATION AND FINANCE"
        assert TITLES[14]["roman"] == "XIV"

    def test_title_xiv_chapters(self):
        """Test that Title XIV has correct chapter range."""
        chapters = TITLES[14]["chapters"]
        assert 192 in chapters
        assert 212 in chapters
        assert 220 in chapters
        assert len(chapters) == 29  # 192-220 inclusive
