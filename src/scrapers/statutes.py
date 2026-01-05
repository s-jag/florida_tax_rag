"""
Florida Statutes Scraper for Online Sunshine (leg.state.fl.us)

URL PATTERNS (documented from manual inspection):
===============================================

1. TITLE INDEX PAGE:
   http://www.leg.state.fl.us/statutes/index.cfm?App_mode=Display_Index&Title_Request=XIV
   - Lists all chapters in a title
   - Chapter links format: <a href="index.cfm?App_mode=Display_Statute&URL=XXXX-XXXX/XXXX/XXXXContentsIndex.html">

2. CHAPTER CONTENTS PAGE:
   http://www.leg.state.fl.us/statutes/index.cfm?App_mode=Display_Statute&URL=0200-0299/0212/0212ContentsIndex.html
   - Lists all sections in a chapter
   - Section links format: index.cfm?App_mode=Display_Statute&URL=XXXX-XXXX/XXXX/Sections/XXXX.XX.html

3. SECTION PAGE:
   http://www.leg.state.fl.us/statutes/index.cfm?App_mode=Display_Statute&URL=0200-0299/0212/Sections/0212.05.html
   - Contains the actual statute text
   - Structure: Title, body text with subsections, History notes at bottom

URL PATH STRUCTURE:
- Chapters 100-199: 0100-0199/0XXX/
- Chapters 200-299: 0200-0299/0XXX/
- Format: {RANGE}/{CHAPTER_PADDED}/

TITLE XIV (Taxation and Finance): Chapters 192-220
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse, parse_qs

from bs4 import BeautifulSoup, NavigableString
import structlog

from src.scrapers.base import BaseScraper
from src.scrapers.models import RawStatute, StatuteMetadata
from src.scrapers.utils import (
    clean_html_text,
    extract_dates,
    normalize_whitespace,
    parse_statute_citation,
)

logger = structlog.get_logger(__name__)

# Base URL for Florida Legislature
BASE_URL = "http://www.leg.state.fl.us/statutes/"

# Title XIV chapters (Taxation and Finance)
TITLE_XIV_CHAPTERS = list(range(192, 221))  # 192-220 inclusive

# Title metadata
TITLES = {
    14: {
        "name": "TAXATION AND FINANCE",
        "roman": "XIV",
        "chapters": TITLE_XIV_CHAPTERS,
    }
}


@dataclass
class ChapterInfo:
    """Information about a chapter from the title index."""

    chapter_number: int
    chapter_name: str
    contents_url: str


@dataclass
class SectionInfo:
    """Information about a section from the chapter contents."""

    section_number: str  # e.g., "212.05"
    section_name: str
    section_url: str
    chapter_number: int


class FloridaStatutesScraper(BaseScraper):
    """Scraper for Florida Statutes from leg.state.fl.us (Online Sunshine)."""

    def __init__(
        self,
        rate_limit_delay: float = 1.0,
        output_dir: Optional[Path] = None,
        **kwargs,
    ):
        """Initialize the Florida Statutes scraper.

        Args:
            rate_limit_delay: Seconds between requests (default 1.0 for politeness).
            output_dir: Directory for saving scraped data.
            **kwargs: Additional arguments passed to BaseScraper.
        """
        super().__init__(rate_limit_delay=rate_limit_delay, **kwargs)
        self.output_dir = output_dir or Path("data/raw/statutes")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_chapter_range_path(self, chapter: int) -> str:
        """Get the URL path range for a chapter number.

        Args:
            chapter: Chapter number (e.g., 212).

        Returns:
            Path component like "0200-0299/0212".
        """
        # Determine the hundreds range
        lower = (chapter // 100) * 100
        upper = lower + 99
        return f"{lower:04d}-{upper:04d}/{chapter:04d}"

    def _build_title_index_url(self, title_num: int) -> str:
        """Build URL for a title's index page.

        Args:
            title_num: Title number (e.g., 14 for XIV).

        Returns:
            Full URL to the title index page.
        """
        roman = self._int_to_roman(title_num)
        return f"{BASE_URL}index.cfm?App_mode=Display_Index&Title_Request={roman}"

    def _int_to_roman(self, num: int) -> str:
        """Convert integer to Roman numeral."""
        val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
        syms = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
        result = ""
        for i, v in enumerate(val):
            while num >= v:
                result += syms[i]
                num -= v
        return result

    def _build_chapter_contents_url(self, chapter: int) -> str:
        """Build URL for a chapter's contents/index page.

        Args:
            chapter: Chapter number.

        Returns:
            Full URL to the chapter contents page.
        """
        range_path = self._get_chapter_range_path(chapter)
        return f"{BASE_URL}index.cfm?App_mode=Display_Statute&URL={range_path}/{chapter:04d}ContentsIndex.html"

    def _build_section_url(self, chapter: int, section: str) -> str:
        """Build URL for a specific section.

        Args:
            chapter: Chapter number.
            section: Section number string (e.g., "212.05").

        Returns:
            Full URL to the section page.
        """
        range_path = self._get_chapter_range_path(chapter)
        # Section files use the full section number with leading zeros for chapter
        section_file = f"{section.zfill(7) if '.' in section else section}.html"
        return f"{BASE_URL}index.cfm?App_mode=Display_Statute&URL={range_path}/Sections/{section_file}"

    async def get_title_chapters(self, title_num: int) -> list[ChapterInfo]:
        """Get all chapters in a title.

        Args:
            title_num: Title number (e.g., 14).

        Returns:
            List of ChapterInfo objects for chapters in this title.
        """
        self.log.info("fetching_title_index", title=title_num)

        url = self._build_title_index_url(title_num)
        html = await self.fetch_page(url)
        soup = BeautifulSoup(html, "lxml")

        chapters = []
        title_info = TITLES.get(title_num, {})
        expected_chapters = title_info.get("chapters", [])

        # Find all chapter links - they contain "ContentsIndex.html" in the URL
        for link in soup.find_all("a", href=True):
            href = link.get("href", "")
            if "ContentsIndex.html" not in href:
                continue

            # Extract chapter number from URL or link text
            chapter_match = re.search(r"/(\d{4})ContentsIndex\.html", href)
            if not chapter_match:
                # Try extracting from link text
                text_match = re.search(r"Chapter\s+(\d+)", link.get_text())
                if text_match:
                    chapter_num = int(text_match.group(1))
                else:
                    continue
            else:
                chapter_num = int(chapter_match.group(1))

            # Filter to only chapters in this title
            if expected_chapters and chapter_num not in expected_chapters:
                continue

            # Get chapter name from sibling text
            chapter_name = ""
            next_sibling = link.next_sibling
            if isinstance(next_sibling, NavigableString):
                chapter_name = normalize_whitespace(str(next_sibling).strip())

            # Build the full URL
            full_url = urljoin(BASE_URL, href)

            chapters.append(
                ChapterInfo(
                    chapter_number=chapter_num,
                    chapter_name=chapter_name,
                    contents_url=full_url,
                )
            )

        # Sort by chapter number
        chapters.sort(key=lambda c: c.chapter_number)
        self.log.info("found_chapters", title=title_num, count=len(chapters))

        return chapters

    async def get_chapter_sections(self, chapter_info: ChapterInfo) -> list[SectionInfo]:
        """Get all sections in a chapter.

        Args:
            chapter_info: ChapterInfo object from get_title_chapters.

        Returns:
            List of SectionInfo objects for sections in this chapter.
        """
        self.log.info(
            "fetching_chapter_contents",
            chapter=chapter_info.chapter_number,
            name=chapter_info.chapter_name,
        )

        html = await self.fetch_page(chapter_info.contents_url)
        soup = BeautifulSoup(html, "lxml")

        sections = []

        # Find all section links - they contain "/Sections/" in the URL
        for link in soup.find_all("a", href=True):
            href = link.get("href", "")
            if "/Sections/" not in href or ".html" not in href:
                continue

            # Extract section number from link text or URL
            link_text = link.get_text(strip=True)
            section_match = re.match(r"(\d+\.\d+)", link_text)
            if not section_match:
                # Try extracting from URL
                url_match = re.search(r"/(\d+\.\d+)\.html", href)
                if url_match:
                    section_number = url_match.group(1)
                else:
                    continue
            else:
                section_number = section_match.group(1)

            # Get section name from sibling text
            section_name = ""
            next_sibling = link.next_sibling
            if isinstance(next_sibling, NavigableString):
                section_name = normalize_whitespace(str(next_sibling).strip())
                # Clean up leading dashes or spaces
                section_name = re.sub(r"^[\s\-—]+", "", section_name)

            full_url = urljoin(BASE_URL, href)

            sections.append(
                SectionInfo(
                    section_number=section_number,
                    section_name=section_name,
                    section_url=full_url,
                    chapter_number=chapter_info.chapter_number,
                )
            )

        # Sort by section number
        sections.sort(key=lambda s: [int(x) for x in s.section_number.split(".")])
        self.log.info(
            "found_sections",
            chapter=chapter_info.chapter_number,
            count=len(sections),
        )

        return sections

    def _parse_section_text(self, soup: BeautifulSoup) -> tuple[str, str]:
        """Extract the main text and history from a section page.

        The page structure has the statute content inside <div id="statutes">
        which contains an embedded HTML document with the actual content.

        Args:
            soup: BeautifulSoup object of the section page.

        Returns:
            Tuple of (main_text, history_text).
        """
        main_text = ""
        history_text = ""

        # First, try to find the statutes div which contains the actual content
        statutes_div = soup.find("div", {"id": "statutes"})

        if statutes_div:
            # The div contains embedded HTML - parse just that portion
            # Look for the Section div within the embedded content
            section_div = statutes_div.find("div", {"class": "Section"})

            if section_div:
                # Extract the section body (main statute text)
                section_body = section_div.find("span", {"class": "SectionBody"})
                if section_body:
                    main_text = section_body.get_text(separator="\n", strip=True)

                # If no SectionBody, try getting all text except history
                if not main_text:
                    # Clone and remove history to get just the main text
                    history_div = section_div.find("div", {"class": "History"})
                    if history_div:
                        history_div_copy = history_div.extract()
                        main_text = section_div.get_text(separator="\n", strip=True)
                        # Re-add for history extraction
                        section_div.append(history_div_copy)
                    else:
                        main_text = section_div.get_text(separator="\n", strip=True)

                # Extract history
                history_div = section_div.find("div", {"class": "History"})
                if history_div:
                    history_text = history_div.get_text(separator=" ", strip=True)
            else:
                # Fallback: get all text from statutes div
                full_text = statutes_div.get_text(separator="\n", strip=True)
                # Split at History marker
                if "History.—" in full_text:
                    parts = full_text.split("History.—", 1)
                    main_text = parts[0].strip()
                    history_text = "History.—" + parts[1].strip()
                else:
                    main_text = full_text
        else:
            # Fallback to old method if no statutes div found
            body = soup.find("body")
            if body:
                full_text = body.get_text(separator="\n", strip=True)
                if "History.—" in full_text:
                    parts = full_text.split("History.—", 1)
                    main_text = parts[0].strip()
                    history_text = "History.—" + parts[1].strip()
                else:
                    main_text = full_text

        # Clean up the text
        main_text = normalize_whitespace(main_text)
        history_text = normalize_whitespace(history_text)

        return main_text, history_text

    def _extract_history_years(self, history_text: str) -> list[str]:
        """Extract amendment years from history text.

        Args:
            history_text: The history section text.

        Returns:
            List of years mentioned in the history (e.g., ["1949", "1961", "2024"]).
        """
        years = set()

        # Pattern 1: "ch. 26319, 1949" - old format with chapter number followed by year
        old_format = re.findall(r"ch\.\s*\d+,\s*(\d{4})", history_text)
        years.update(old_format)

        # Pattern 2: "ch. 2024-99" - modern Laws of Florida format (4-digit year-chapter)
        modern_format = re.findall(r"ch\.\s*(\d{4})-\d+", history_text)
        years.update(modern_format)

        # Pattern 3: "ch. 95-290" - older Laws of Florida format (2-digit year-chapter)
        # Convert 2-digit years: 00-29 -> 2000-2029, 30-99 -> 1930-1999
        two_digit_format = re.findall(r"ch\.\s*(\d{2})-\d+", history_text)
        for y in two_digit_format:
            year_int = int(y)
            if year_int <= 29:
                years.add(str(2000 + year_int))
            else:
                years.add(str(1900 + year_int))

        # Pattern 4: standalone years like ", 2024." at end of history
        standalone_years = re.findall(r",\s*(\d{4})\.", history_text)
        years.update(standalone_years)

        # Filter to reasonable year range (1800-2100)
        valid_years = [y for y in years if 1800 <= int(y) <= 2100]

        return sorted(valid_years)

    def _extract_section_title(self, soup: BeautifulSoup, section_number: str) -> str:
        """Extract the section title from the page.

        Args:
            soup: BeautifulSoup object.
            section_number: The section number to look for.

        Returns:
            The section title.
        """
        # First, try to find the title in the structured HTML
        statutes_div = soup.find("div", {"id": "statutes"})
        if statutes_div:
            # Look for Catchline span which contains the title
            catchline = statutes_div.find("span", {"class": "Catchline"})
            if catchline:
                catchline_text = catchline.find("span", {"class": "CatchlineText"})
                if catchline_text:
                    return normalize_whitespace(catchline_text.get_text(strip=True))
                # Fallback to full catchline text
                title = catchline.get_text(strip=True)
                # Remove the em-dash if present
                title = re.sub(r"[—\-]+$", "", title).strip()
                return normalize_whitespace(title)

        # Fallback: Look for the section number followed by its title in body text
        body_text = soup.get_text()

        # Pattern: "212.05 Section Title.—" or similar
        pattern = re.compile(
            rf"{re.escape(section_number)}\s+(.+?)(?:\.—|—|\.\s*\(1\)|\.\s*$)",
            re.DOTALL,
        )
        match = pattern.search(body_text)
        if match:
            title = match.group(1).strip()
            # Clean up extra whitespace
            title = normalize_whitespace(title)
            # Remove any trailing punctuation
            title = re.sub(r"[\.\-—]+$", "", title).strip()
            return title

        return ""

    async def scrape_section(
        self,
        section_info: SectionInfo,
        title_num: int = 14,
    ) -> RawStatute:
        """Scrape a single statute section.

        Args:
            section_info: SectionInfo object with section details.
            title_num: Title number (default 14).

        Returns:
            RawStatute object with the scraped content.
        """
        self.log.info(
            "scraping_section",
            section=section_info.section_number,
            chapter=section_info.chapter_number,
        )

        html = await self.fetch_page(section_info.section_url)
        soup = BeautifulSoup(html, "lxml")

        # Extract text content
        main_text, history_text = self._parse_section_text(soup)

        # Extract section title
        section_title = section_info.section_name
        if not section_title:
            section_title = self._extract_section_title(soup, section_info.section_number)

        # Extract history/amendment years
        history_years = self._extract_history_years(history_text)

        # Extract effective dates
        effective_dates = extract_dates(history_text)
        effective_date = effective_dates[-1] if effective_dates else None

        # Get title info
        title_info = TITLES.get(title_num, {})

        # Build metadata
        metadata = StatuteMetadata(
            title=title_info.get("name", ""),
            title_number=title_num,
            chapter=section_info.chapter_number,
            chapter_name="",  # Will be filled in by caller
            section=section_info.section_number,
            section_name=section_title,
            subsection=None,
            effective_date=effective_date,
            history=history_years,
        )

        # Create RawStatute
        return RawStatute(
            metadata=metadata,
            text=main_text,
            html=html,
            source_url=section_info.section_url,
            scraped_at=datetime.now(timezone.utc),
        )

    async def scrape_chapter(
        self,
        chapter_info: ChapterInfo,
        title_num: int = 14,
    ) -> list[RawStatute]:
        """Scrape all sections in a chapter.

        Args:
            chapter_info: ChapterInfo object.
            title_num: Title number.

        Returns:
            List of RawStatute objects for all sections.
        """
        self.log.info(
            "scraping_chapter",
            chapter=chapter_info.chapter_number,
            name=chapter_info.chapter_name,
        )

        # Get all sections in the chapter
        sections = await self.get_chapter_sections(chapter_info)

        statutes = []
        for i, section_info in enumerate(sections):
            try:
                statute = await self.scrape_section(section_info, title_num)
                # Update chapter name
                statute.metadata.chapter_name = chapter_info.chapter_name
                statutes.append(statute)

                # Save progress incrementally
                self._save_section(statute, chapter_info.chapter_number)

            except Exception as e:
                self.log.error(
                    "section_scrape_failed",
                    section=section_info.section_number,
                    error=str(e),
                )

            # Rate limit between sections
            if i < len(sections) - 1:
                await asyncio.sleep(self.rate_limit_delay)

        self.log.info(
            "chapter_complete",
            chapter=chapter_info.chapter_number,
            sections_scraped=len(statutes),
        )

        return statutes

    def _save_section(self, statute: RawStatute, chapter: int) -> None:
        """Save a single section to disk incrementally.

        Args:
            statute: RawStatute to save.
            chapter: Chapter number for directory organization.
        """
        chapter_dir = self.output_dir / f"chapter_{chapter}"
        chapter_dir.mkdir(parents=True, exist_ok=True)

        filename = f"section_{statute.metadata.section}.json"
        filepath = chapter_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(statute.model_dump(mode="json"), f, indent=2, default=str)

    async def scrape_title(self, title_num: int = 14) -> list[RawStatute]:
        """Scrape all sections in a title.

        Args:
            title_num: Title number (default 14 for Taxation and Finance).

        Returns:
            List of all RawStatute objects scraped.
        """
        self.log.info("starting_title_scrape", title=title_num)

        # Get all chapters in the title
        chapters = await self.get_title_chapters(title_num)

        all_statutes = []
        for i, chapter_info in enumerate(chapters):
            try:
                statutes = await self.scrape_chapter(chapter_info, title_num)
                all_statutes.extend(statutes)
            except Exception as e:
                self.log.error(
                    "chapter_scrape_failed",
                    chapter=chapter_info.chapter_number,
                    error=str(e),
                )

            # Rate limit between chapters
            if i < len(chapters) - 1:
                await asyncio.sleep(self.rate_limit_delay)

        self.log.info(
            "title_scrape_complete",
            title=title_num,
            total_statutes=len(all_statutes),
        )

        # Save combined output
        self._save_title_summary(title_num, all_statutes)

        return all_statutes

    def _save_title_summary(self, title_num: int, statutes: list[RawStatute]) -> None:
        """Save a summary of all scraped statutes for a title.

        Args:
            title_num: Title number.
            statutes: List of all scraped statutes.
        """
        summary = {
            "title_number": title_num,
            "title_name": TITLES.get(title_num, {}).get("name", ""),
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "total_sections": len(statutes),
            "chapters": {},
        }

        # Group by chapter
        for statute in statutes:
            chapter = statute.metadata.chapter
            if chapter not in summary["chapters"]:
                summary["chapters"][chapter] = {
                    "chapter_name": statute.metadata.chapter_name,
                    "sections": [],
                }
            summary["chapters"][chapter]["sections"].append(
                {
                    "section": statute.metadata.section,
                    "section_name": statute.metadata.section_name,
                    "effective_date": (
                        statute.metadata.effective_date.isoformat()
                        if statute.metadata.effective_date
                        else None
                    ),
                }
            )

        # Save summary
        filepath = self.output_dir / f"title_{title_num}_summary.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        self.log.info("saved_title_summary", filepath=str(filepath))

    async def scrape(self) -> list[dict]:
        """Main scrape method - scrapes Title XIV by default.

        Returns:
            List of scraped statute dictionaries.
        """
        statutes = await self.scrape_title(14)
        return [s.model_dump(mode="json") for s in statutes]
