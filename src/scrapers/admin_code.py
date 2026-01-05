"""Florida Administrative Code scraper for tax-related rules."""

from __future__ import annotations

import asyncio
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

from bs4 import BeautifulSoup

from .base import BaseScraper, FetchError
from .models import RawRule, RuleMetadata


class FloridaAdminCodeScraper(BaseScraper):
    """Scraper for Florida Administrative Code (flrules.org).

    Focuses on Department of Revenue rules (12A, 12B, 12C, 12D).
    Extracts rulemaking authority and law implemented for graph links.
    """

    BASE_URL = "https://www.flrules.org"
    DEPARTMENT_URL = f"{BASE_URL}/gateway/Department.asp?DeptID=12"

    # Tax-related divisions in Department 12 (Revenue)
    TAX_DIVISIONS = {
        "12A": "Sales and Use Tax, Surtax, Surcharge, and Fees; Communications Services Tax",
        "12B": "Miscellaneous Tax",
        "12C": "Corporate, Estate and Intangible Tax",
        "12D": "Property Tax Oversight Program",
    }

    def __init__(
        self,
        rate_limit_delay: float = 1.0,
        timeout: float = 30.0,
        max_retries: int = 3,
        use_cache: bool = True,
        cache_dir: Path | None = None,
        raw_data_dir: Path | None = None,
    ):
        super().__init__(
            rate_limit_delay=rate_limit_delay,
            timeout=timeout,
            max_retries=max_retries,
            use_cache=use_cache,
            cache_dir=cache_dir,
            raw_data_dir=raw_data_dir,
        )
        self.output_dir = self.raw_data_dir / "admin_code"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def get_division_chapters(self, division: str) -> list[dict[str, str]]:
        """Get all chapters in a division.

        Args:
            division: Division code (e.g., '12A')

        Returns:
            List of dicts with 'chapter' and 'title' keys.
        """
        url = f"{self.BASE_URL}/gateway/Division.asp?DivID={division}"
        html = await self.fetch_page(url)
        soup = BeautifulSoup(html, "lxml")

        chapters = []
        # Look for chapter links in the table
        for link in soup.find_all("a", href=True):
            href = link.get("href", "")
            if "ChapterHome.asp" in href and "Chapter=" in href:
                # Extract chapter from URL
                match = re.search(r"Chapter=([^&\"]+)", href)
                if match:
                    chapter = match.group(1)
                    title = link.get_text(strip=True)
                    chapters.append({
                        "chapter": chapter,
                        "title": title,
                    })

        self.log.info("found_chapters", division=division, count=len(chapters))
        return chapters

    async def get_chapter_rules(self, chapter: str) -> list[dict[str, Any]]:
        """Get all rules in a chapter.

        Args:
            chapter: Chapter code (e.g., '12A-1')

        Returns:
            List of dicts with rule metadata.
        """
        url = f"{self.BASE_URL}/gateway/ChapterHome.asp?Chapter={chapter}"
        html = await self.fetch_page(url)
        soup = BeautifulSoup(html, "lxml")

        rules = []
        seen = set()

        # Look for rule links
        for link in soup.find_all("a", href=True):
            href = link.get("href", "")
            if "RuleNo.asp" in href and "ID=" in href:
                # Extract rule ID
                match = re.search(r"ID=([^&\"]+)", href)
                if match:
                    rule_id = match.group(1)
                    if rule_id not in seen:
                        seen.add(rule_id)
                        title = link.get_text(strip=True)
                        rules.append({
                            "rule_number": rule_id,
                            "title": title,
                        })

        self.log.info("found_rules", chapter=chapter, count=len(rules))
        return rules

    async def scrape_rule(self, rule_number: str, chapter_title: str = "") -> RawRule:
        """Scrape a single rule.

        Args:
            rule_number: Full rule number (e.g., '12A-1.001')
            chapter_title: Optional chapter title for URL construction

        Returns:
            RawRule object with full metadata.
        """
        # URL encode the title for the request
        title_encoded = chapter_title.replace(" ", "%20") if chapter_title else ""
        url = f"{self.BASE_URL}/gateway/RuleNo.asp?title={title_encoded}&ID={rule_number}"

        html = await self.fetch_page(url)
        soup = BeautifulSoup(html, "lxml")

        # Parse the rule page
        rule_title = self._extract_rule_title(soup, rule_number)
        text = self._extract_rule_text(soup)
        effective_date = self._extract_effective_date(soup)
        rulemaking_authority = self._extract_rulemaking_authority(soup)
        law_implemented = self._extract_law_implemented(soup)
        history = self._extract_history(soup)

        # Extract chapter from rule number (e.g., '12A-1' from '12A-1.001')
        chapter = "-".join(rule_number.split("-")[:-1]) + "-" + rule_number.split("-")[-1].split(".")[0]
        if "-" in rule_number:
            parts = rule_number.split(".")
            if len(parts) >= 1:
                chapter = parts[0].rsplit(".", 1)[0] if "." in parts[0] else parts[0]
                # For format like "12A-1.001", chapter should be "12A-1"
                chapter = rule_number.rsplit(".", 1)[0]
                if "." not in chapter:
                    chapter = rule_number.rsplit(".", 1)[0]

        # Actually extract chapter correctly: 12A-1.001 -> 12A-1
        chapter = re.match(r"^(\d+[A-Z]?-\d+)", rule_number)
        chapter = chapter.group(1) if chapter else rule_number.split(".")[0]

        metadata = RuleMetadata(
            chapter=chapter,
            rule_number=rule_number,
            title=rule_title,
            effective_date=effective_date,
            rulemaking_authority=rulemaking_authority,
            law_implemented=law_implemented,
            references_statutes=list(set(rulemaking_authority + law_implemented)),
        )

        return RawRule(
            metadata=metadata,
            text=text,
            html=html,
            source_url=url,
            scraped_at=datetime.now(timezone.utc),
        )

    def _extract_rule_title(self, soup: BeautifulSoup, rule_number: str) -> str:
        """Extract the rule title from the page."""
        # Method 1: Look for "Rule Title: XXX" in a table cell
        for td in soup.find_all("td"):
            text = td.get_text(strip=True)
            if text.startswith("Rule Title:"):
                title = text.replace("Rule Title:", "").strip()
                if title:
                    return title

        # Method 2: Look in the page title
        # Format: "12A-1.001 : Specific Exemptions - Florida Administrative Rules..."
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)
            # Extract between rule number and " - Florida"
            match = re.search(
                rf"{re.escape(rule_number)}\s*:\s*([^-]+?)(?:\s*-\s*Florida|\s*$)",
                title
            )
            if match:
                return match.group(1).strip()

        # Method 3: Look in page text for pattern
        text = soup.get_text()
        patterns = [
            rf"{re.escape(rule_number)}\s*:\s*([^\n-]+)",
            rf"Rule Title:\s*([^\n]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                title = match.group(1).strip()
                # Clean up common suffixes
                title = re.sub(r"\s*(Prev|Up|Next|Previous|Florida).*$", "", title)
                if title and len(title) > 3:
                    return title

        return ""

    def _extract_rule_text(self, soup: BeautifulSoup) -> str:
        """Extract the main rule text content."""
        # The rule text is typically in a specific div or the Word document
        # For now, extract from the page content

        # Remove script and style elements
        for element in soup(["script", "style", "nav", "header", "footer"]):
            element.decompose()

        # Look for the main content area
        content = ""

        # Try to find rule content in tables or specific divs
        main_table = soup.find("table", {"width": "100%"})
        if main_table:
            # Get text from table cells that contain rule text
            for td in main_table.find_all("td"):
                td_text = td.get_text(separator="\n", strip=True)
                if len(td_text) > 100:  # Likely rule content
                    content = td_text
                    break

        if not content:
            # Fallback to body text
            body = soup.find("body")
            if body:
                content = body.get_text(separator="\n", strip=True)

        # Clean up the text
        content = self._clean_text(content)
        return content

    def _extract_effective_date(self, soup: BeautifulSoup) -> Optional[date]:
        """Extract the effective date from the page."""
        # Look for "Effective Date" text
        text = soup.get_text()

        # Pattern: "Effective Date: MM/DD/YYYY" or similar
        patterns = [
            r"Effective\s*(?:Date)?[:\s]+(\d{1,2}/\d{1,2}/\d{4})",
            r"Effective\s*(?:Date)?[:\s]+(\d{1,2}-\d{1,2}-\d{4})",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                try:
                    if "/" in date_str:
                        return datetime.strptime(date_str, "%m/%d/%Y").date()
                    else:
                        return datetime.strptime(date_str, "%m-%d-%Y").date()
                except ValueError:
                    pass

        return None

    def _extract_rulemaking_authority(self, soup: BeautifulSoup) -> list[str]:
        """Extract rulemaking authority statute citations.

        This identifies which statutes authorize the agency to create this rule.
        """
        return self._extract_statute_citations(soup, "Rulemaking Authority")

    def _extract_law_implemented(self, soup: BeautifulSoup) -> list[str]:
        """Extract law implemented statute citations.

        This identifies which statutes this rule implements.
        """
        return self._extract_statute_citations(soup, "Law Implemented")

    def _extract_statute_citations(self, soup: BeautifulSoup, section_name: str) -> list[str]:
        """Extract statute citations from a specific section.

        Args:
            soup: BeautifulSoup object
            section_name: Either "Rulemaking Authority" or "Law Implemented"

        Returns:
            List of statute citations (e.g., ['212.05', '212.08(7)(f)'])
        """
        citations = []

        # Get the full text to find the section
        text = soup.get_text()

        # Find the section in the text
        # Pattern: "Rulemaking Authority ... Law Implemented" or "Law Implemented ... History"
        if section_name == "Rulemaking Authority":
            # Extract text between "Rulemaking Authority" and "Law Implemented"
            match = re.search(
                r"Rulemaking Authority\s+(.*?)\s+Law Implemented",
                text,
                re.IGNORECASE | re.DOTALL
            )
        else:
            # Extract text between "Law Implemented" and "History"
            match = re.search(
                r"Law Implemented\s+(.*?)\s+History",
                text,
                re.IGNORECASE | re.DOTALL
            )

        if match:
            section_text = match.group(1)
            # Extract citation patterns
            citations = self._parse_statute_citations(section_text)

        # Also look for statute links in the HTML
        for link in soup.find_all("a", href=True):
            href = link.get("href", "")
            if "statute.asp" in href:
                # Extract statute ID from URL
                id_match = re.search(r"id=([^&\"]+)", href)
                if id_match:
                    stat_id = id_match.group(1).strip()
                    # Clean up the citation
                    stat_id = stat_id.replace(" FS.", "").replace("FS.", "").strip()
                    if stat_id and stat_id not in citations:
                        citations.append(stat_id)

        return list(set(citations))

    def _parse_statute_citations(self, text: str) -> list[str]:
        """Parse statute citations from text.

        Handles formats like:
        - 212.05
        - 212.08(7)(f)
        - 212.08(7)(h)2.
        - ss. 212.05, 212.06
        """
        citations = []

        # Pattern for Florida statute citations
        # Matches: NNN.NN, NNN.NN(N)(a), NNN.NN(N)(a)N.
        pattern = r"\b(\d{3}\.\d+(?:\([^)]+\))*(?:\d+\.)?)"

        matches = re.findall(pattern, text)
        for match in matches:
            # Clean up the citation
            citation = match.strip().rstrip(".,")
            if citation:
                citations.append(citation)

        return list(set(citations))

    def _extract_history(self, soup: BeautifulSoup) -> str:
        """Extract the rule history section."""
        text = soup.get_text()

        # Find history section
        match = re.search(r"History[–-]\s*(.*?)(?:\s*$|\s*Previous|\s*Up|\s*Next)", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        return ""

    def _clean_text(self, text: str) -> str:
        """Clean up extracted text."""
        # Remove excessive whitespace
        text = re.sub(r"\n\s*\n", "\n\n", text)
        text = re.sub(r" +", " ", text)

        # Remove navigation elements that might be captured
        lines = text.split("\n")
        filtered_lines = []
        skip_patterns = ["Previous", "Up", "Next", "MyFLRules", "Add to Favorites"]

        for line in lines:
            line = line.strip()
            if line and not any(p in line for p in skip_patterns):
                filtered_lines.append(line)

        return "\n".join(filtered_lines).strip()

    async def scrape_chapter(
        self,
        chapter: str,
        delay: float | None = None,
    ) -> list[RawRule]:
        """Scrape all rules in a chapter.

        Args:
            chapter: Chapter code (e.g., '12A-1')
            delay: Delay between requests (uses instance default if None)

        Returns:
            List of RawRule objects.
        """
        delay = delay if delay is not None else self.rate_limit_delay

        # Get list of rules in this chapter
        rules_list = await self.get_chapter_rules(chapter)
        self.log.info("scraping_chapter", chapter=chapter, total_rules=len(rules_list))

        rules = []
        for i, rule_info in enumerate(rules_list):
            try:
                rule = await self.scrape_rule(
                    rule_info["rule_number"],
                    rule_info.get("title", ""),
                )
                rules.append(rule)

                # Save individual rule
                self._save_rule(rule, chapter)

            except FetchError as e:
                self.log.error(
                    "rule_fetch_failed",
                    rule=rule_info["rule_number"],
                    error=str(e),
                )

            # Rate limit
            if i < len(rules_list) - 1:
                await asyncio.sleep(delay)

        return rules

    def _save_rule(self, rule: RawRule, chapter: str) -> Path:
        """Save a rule to its JSON file."""
        chapter_dir = self.output_dir / f"chapter_{chapter.replace('-', '_')}"
        chapter_dir.mkdir(parents=True, exist_ok=True)

        filename = f"rule_{rule.metadata.rule_number.replace('.', '_')}.json"
        filepath = chapter_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(rule.model_dump_json(indent=2))

        self.log.debug("saved_rule", rule=rule.metadata.rule_number, path=str(filepath))
        return filepath

    async def scrape_division(
        self,
        division: str,
        delay: float | None = None,
    ) -> list[RawRule]:
        """Scrape all rules in a division.

        Args:
            division: Division code (e.g., '12A')
            delay: Delay between requests

        Returns:
            List of RawRule objects.
        """
        chapters = await self.get_division_chapters(division)
        self.log.info("scraping_division", division=division, total_chapters=len(chapters))

        all_rules = []
        for chapter_info in chapters:
            chapter_rules = await self.scrape_chapter(
                chapter_info["chapter"],
                delay=delay,
            )
            all_rules.extend(chapter_rules)

        return all_rules

    async def scrape(
        self,
        divisions: list[str] | None = None,
        delay: float | None = None,
    ) -> list[dict[str, Any]]:
        """Scrape tax-related administrative rules.

        Args:
            divisions: List of division codes to scrape. Defaults to all tax divisions.
            delay: Delay between requests.

        Returns:
            List of scraped rule dictionaries.
        """
        divisions = divisions or list(self.TAX_DIVISIONS.keys())
        delay = delay if delay is not None else self.rate_limit_delay

        all_rules = []
        for division in divisions:
            if division not in self.TAX_DIVISIONS:
                self.log.warning("unknown_division", division=division)
                continue

            rules = await self.scrape_division(division, delay=delay)
            all_rules.extend(rules)

        # Save combined output
        self._save_combined_output(all_rules)

        return [r.model_dump() for r in all_rules]

    def _save_combined_output(self, rules: list[RawRule]) -> None:
        """Save combined output files."""
        # Save as JSON array
        output_path = self.output_dir / "all_rules.json"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("[\n")
            for i, rule in enumerate(rules):
                f.write(rule.model_dump_json(indent=2))
                if i < len(rules) - 1:
                    f.write(",\n")
            f.write("\n]")

        self.log.info("saved_combined", path=str(output_path), count=len(rules))

        # Save summary
        summary = {
            "total_rules": len(rules),
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "chapters": {},
        }

        for rule in rules:
            chapter = rule.metadata.chapter
            if chapter not in summary["chapters"]:
                summary["chapters"][chapter] = {
                    "count": 0,
                    "rules": [],
                }
            summary["chapters"][chapter]["count"] += 1
            summary["chapters"][chapter]["rules"].append(rule.metadata.rule_number)

        summary_path = self.output_dir / "summary.json"
        self.save_raw(summary, "admin_code/summary.json")
