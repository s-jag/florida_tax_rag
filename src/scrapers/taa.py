"""Florida Department of Revenue Technical Assistance Advisement (TAA) scraper."""

from __future__ import annotations

import asyncio
import re
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

from bs4 import BeautifulSoup

from .base import BaseScraper, FetchError
from .models import RawTAA, TAAMetadata
from .utils import parse_rule_citation, parse_statute_citation

# Try to import PDF libraries
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False


class FloridaTAAScraper(BaseScraper):
    """Scraper for Florida DOR Technical Assistance Advisements.

    TAAs are available as PDFs at floridarevenue.com/TaxLaw/Documents/
    """

    BASE_URL = "https://floridarevenue.com/TaxLaw"
    DOCUMENTS_URL = f"{BASE_URL}/Documents"
    SEARCH_URL = f"{BASE_URL}/Pages/results.aspx"

    # Tax type code to name mapping
    TAX_TYPE_CODES = {
        "A": "Sales and Use Tax",
        "B": "Corporate Income Tax",
        "C": "Documentary Stamp Tax",
        "D": "Property Tax",
        "E": "Estate Tax",
        "F": "Fuel Tax",
        "G": "Gross Receipts Tax",
        "H": "Hazardous Waste Tax",
        "I": "Insurance Premium Tax",
        "L": "Local Option Tax",
        "M": "Motor Vehicle Tax",
        "N": "Communications Services Tax",
        "P": "Pollutant Tax",
        "R": "Reemployment Tax",
        "S": "Solid Waste Tax",
        "T": "Tobacco Tax",
        "U": "Unemployment Tax",
        "V": "Severance Tax",
        "W": "Withholding Tax",
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
        self.output_dir = self.raw_data_dir / "taa"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir = self.output_dir / "pdfs"
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

    async def _fetch_search_page_with_curl(self) -> str:
        """Fetch the search page using curl (SharePoint sometimes blocks httpx)."""
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["curl", "-s", self.SEARCH_URL],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return result.stdout
            self.log.warning("curl_failed", returncode=result.returncode)
            return ""
        except Exception as e:
            self.log.warning("curl_error", error=str(e))
            return ""

    async def get_taa_index(self) -> list[dict[str, Any]]:
        """Scrape the TAA index/search page to find available TAAs.

        Returns:
            List of dicts with 'filename', 'title', 'url', 'date' keys.
        """
        # Try httpx first
        html = await self.fetch_page(self.SEARCH_URL)

        # Check if we got the expected content (SharePoint can serve different content)
        if "TaxLaw/Documents" not in html:
            self.log.info("httpx_missing_content", message="Falling back to curl")
            html = await self._fetch_search_page_with_curl()

        taas = []
        seen = set()

        # Use regex to parse the table structure directly (SharePoint HTML is tricky)
        # Pattern: <tr><td><a href="URL">TITLE</a></td><td>TYPE</td><td>DATE</td>
        row_pattern = re.compile(
            r'<tr><td><a href="([^"]+)"[^>]*>([^<]+)</a></td><td>([^<]*)</td><td>([^<]*)</td>',
            re.IGNORECASE
        )

        for match in row_pattern.finditer(html):
            href = match.group(1)
            title = match.group(2).strip()
            doc_type = match.group(3).strip()
            date_str = match.group(4).strip()

            # Skip if not a PDF
            if not href.lower().endswith(".pdf"):
                continue

            # Skip TIPs (Tax Information Publications) - they're in /taxes/tips/
            if "/tips/" in href.lower():
                continue

            # Skip court case documents (vs. in title)
            if " vs. " in title.lower() or " vs " in title.lower():
                continue

            # Skip PTO BUL (Property Tax Oversight Bulletins)
            if "pto bul" in title.lower() or "PTO%20BUL" in href:
                continue

            filename = href.split("/")[-1]
            if filename in seen:
                continue

            # Check if this looks like a TAA based on filename pattern
            # TAAs have patterns like: 25A-009.pdf, 24A-001.pdf, 23B-005.pdf
            # Decode URL encoding first
            decoded_filename = filename.replace("%20", " ")
            taa_pattern = re.match(r"^(\d{2}[A-Z]\d*-\d+)\.pdf$", decoded_filename, re.IGNORECASE)
            if not taa_pattern:
                continue

            seen.add(filename)

            # Build full URL
            if href.startswith("/"):
                url = f"https://floridarevenue.com{href}"
            elif href.startswith("http"):
                url = href
            else:
                url = f"{self.DOCUMENTS_URL}/{filename}"

            # Extract TAA number from filename
            taa_number = self._extract_taa_number(decoded_filename)

            taas.append({
                "filename": filename,
                "taa_number": taa_number,
                "url": url,
                "title": title or filename,
                "date": date_str,
            })

        self.log.info("found_taas", count=len(taas))
        return taas

    def _extract_taa_number(self, filename: str) -> str:
        """Extract TAA number from filename.

        Examples:
            - "TAA 25A01-19.pdf" -> "TAA 25A01-19"
            - "TAA 12C1-006.pdf" -> "TAA 12C1-006"
            - "10A-049.pdf" -> "TAA 10A-049"
        """
        # Remove .pdf extension
        name = filename.replace(".pdf", "").replace(".PDF", "")

        # If already has TAA prefix, clean it up
        if name.upper().startswith("TAA"):
            name = name[3:].strip()

        # Clean up URL encoding
        name = name.replace("%20", " ")

        return f"TAA {name}"

    def _extract_tax_type(self, taa_number: str) -> tuple[str, str]:
        """Extract tax type code and name from TAA number.

        Returns:
            Tuple of (tax_type_code, tax_type_name)
        """
        # TAA format: "TAA YYX##-###" where X is the tax type letter
        match = re.search(r"TAA\s*\d{2}([A-Z])", taa_number, re.IGNORECASE)
        if match:
            code = match.group(1).upper()
            name = self.TAX_TYPE_CODES.get(code, f"Unknown ({code})")
            return code, name

        return "", "Unknown"

    async def download_pdf(self, url: str, filename: str) -> Path:
        """Download a TAA PDF file.

        Args:
            url: URL to the PDF
            filename: Filename to save as

        Returns:
            Path to the downloaded PDF
        """
        pdf_path = self.pdf_dir / filename

        # Check if already downloaded
        if pdf_path.exists():
            self.log.debug("pdf_cached", filename=filename)
            return pdf_path

        # Download the PDF
        client = await self._get_client()
        headers = {"User-Agent": self._get_next_user_agent()}

        self.log.info("downloading_pdf", url=url)
        response = await client.get(url, headers=headers)
        response.raise_for_status()

        # Save to file
        pdf_path.write_bytes(response.content)
        self.log.debug("pdf_saved", path=str(pdf_path))

        return pdf_path

    def extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text from a PDF file.

        Uses pdfplumber first, falls back to pypdf.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text content
        """
        text = ""

        # Try pdfplumber first (better text extraction)
        if HAS_PDFPLUMBER:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    pages_text = []
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            pages_text.append(page_text)
                    text = "\n\n".join(pages_text)
                    if text.strip():
                        return text
            except Exception as e:
                self.log.warning("pdfplumber_failed", error=str(e), path=str(pdf_path))

        # Fallback to pypdf
        if HAS_PYPDF:
            try:
                reader = PdfReader(pdf_path)
                pages_text = []
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pages_text.append(page_text)
                text = "\n\n".join(pages_text)
            except Exception as e:
                self.log.warning("pypdf_failed", error=str(e), path=str(pdf_path))

        return text

    def parse_taa_content(self, text: str, taa_number: str) -> TAAMetadata:
        """Parse TAA content to extract metadata.

        Args:
            text: Full text content of the TAA
            taa_number: TAA number

        Returns:
            TAAMetadata object
        """
        # Extract tax type
        tax_type_code, tax_type = self._extract_tax_type(taa_number)

        # Extract issue date
        issue_date = self._extract_issue_date(text)

        # Extract title/subject from "Re:" line
        title = self._extract_subject(text)

        # Extract question and answer sections
        question = self._extract_question(text)
        answer = self._extract_answer(text)

        # Extract citations
        statutes_cited = parse_statute_citation(text)
        rules_cited = parse_rule_citation(text)

        # Extract topics (from subject line keywords)
        topics = self._extract_topics(title)

        return TAAMetadata(
            taa_number=taa_number,
            title=title,
            issue_date=issue_date,
            tax_type=tax_type,
            tax_type_code=tax_type_code,
            topics=topics,
            question=question,
            answer=answer,
            statutes_cited=statutes_cited,
            rules_cited=rules_cited,
        )

    def _extract_issue_date(self, text: str) -> Optional[date]:
        """Extract the issue date from TAA text."""
        # Common date patterns in TAAs
        patterns = [
            r"(?:Date|Dated|Issued)[:\s]+(\w+\s+\d{1,2},?\s+\d{4})",
            r"(\w+\s+\d{1,2},?\s+\d{4})",  # "January 15, 2024"
            r"(\d{1,2}/\d{1,2}/\d{4})",  # "1/15/2024"
        ]

        for pattern in patterns:
            match = re.search(pattern, text[:2000], re.IGNORECASE)  # Look in first 2000 chars
            if match:
                date_str = match.group(1)
                try:
                    # Try various formats
                    for fmt in ["%B %d, %Y", "%B %d %Y", "%m/%d/%Y", "%m/%d/%y"]:
                        try:
                            return datetime.strptime(date_str.replace(",", ""), fmt).date()
                        except ValueError:
                            continue
                except Exception:
                    pass

        return None

    def _extract_subject(self, text: str) -> str:
        """Extract the subject/title from the TAA text."""
        # Look for "Re:" or "RE:" or "Subject:" line
        patterns = [
            r"(?:Re|RE|Subject)[:\s]+([^\n]+)",
            r"(?:SUBJECT|MATTER)[:\s]+([^\n]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text[:3000])
            if match:
                subject = match.group(1).strip()
                # Clean up common suffixes
                subject = re.sub(r"\s*[-–]\s*TAA.*$", "", subject)
                if len(subject) > 10:  # Reasonable subject length
                    return subject

        return ""

    def _extract_question(self, text: str) -> str:
        """Extract the question/issue section from TAA text."""
        # Common section headers
        patterns = [
            r"(?:QUESTION|ISSUE|FACTS)[:\s]*\n([\s\S]+?)(?=\n(?:ANSWER|RESPONSE|CONCLUSION|RULING))",
            r"(?:Your (?:question|inquiry|request))[:\s]*([\s\S]+?)(?=\n(?:ANSWER|RESPONSE|CONCLUSION|Based on))",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                question = match.group(1).strip()
                # Limit length
                if len(question) > 100:
                    return question[:5000]

        return ""

    def _extract_answer(self, text: str) -> str:
        """Extract the answer/response section from TAA text."""
        # Common section headers
        patterns = [
            r"(?:ANSWER|RESPONSE|CONCLUSION|RULING)[:\s]*\n([\s\S]+?)(?=\n(?:TAXPAYER|DEPARTMENT|Sincerely|This (?:letter|TAA)))",
            r"(?:Based on the (?:facts|information))[:\s]*([\s\S]+?)(?=\n(?:TAXPAYER|Sincerely|This (?:letter|TAA)))",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                # Limit length
                if len(answer) > 100:
                    return answer[:10000]

        return ""

    def _extract_topics(self, subject: str) -> list[str]:
        """Extract topic keywords from the subject line."""
        topics = []

        # Common tax topic keywords
        topic_keywords = [
            "exemption", "exempt", "taxable", "sale", "purchase", "resale",
            "service", "lease", "rental", "repair", "installation",
            "manufacturing", "agriculture", "construction", "real property",
            "tangible personal property", "software", "digital", "electronic",
            "food", "medical", "educational", "nonprofit", "government",
            "import", "export", "interstate", "intrastate",
        ]

        subject_lower = subject.lower()
        for keyword in topic_keywords:
            if keyword in subject_lower:
                topics.append(keyword.title())

        return topics[:5]  # Limit to top 5 topics

    async def scrape_taa(self, url: str, taa_number: str, filename: str) -> RawTAA:
        """Scrape a single TAA.

        Args:
            url: URL to the TAA PDF
            taa_number: TAA number
            filename: PDF filename

        Returns:
            RawTAA object
        """
        # Download the PDF
        pdf_path = await self.download_pdf(url, filename)

        # Extract text
        text = self.extract_pdf_text(pdf_path)

        # Parse metadata
        metadata = self.parse_taa_content(text, taa_number)

        return RawTAA(
            metadata=metadata,
            text=text,
            pdf_path=str(pdf_path),
            source_url=url,
            scraped_at=datetime.now(timezone.utc),
        )

    def _save_taa(self, taa: RawTAA) -> Path:
        """Save a TAA to JSON file."""
        # Create filename from TAA number
        safe_name = taa.metadata.taa_number.replace(" ", "_").replace("/", "-")
        filename = f"{safe_name}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(taa.model_dump_json(indent=2))

        self.log.debug("saved_taa", taa=taa.metadata.taa_number, path=str(filepath))
        return filepath

    async def scrape_from_index(
        self,
        max_count: int | None = None,
        delay: float | None = None,
    ) -> list[RawTAA]:
        """Scrape TAAs from the index page.

        Args:
            max_count: Maximum number of TAAs to scrape (None for all)
            delay: Delay between requests

        Returns:
            List of RawTAA objects
        """
        delay = delay if delay is not None else self.rate_limit_delay

        # Get TAA index
        taa_index = await self.get_taa_index()
        self.log.info("scraping_taas", total=len(taa_index), max_count=max_count)

        if max_count:
            taa_index = taa_index[:max_count]

        taas = []
        for i, taa_info in enumerate(taa_index):
            try:
                taa = await self.scrape_taa(
                    taa_info["url"],
                    taa_info["taa_number"],
                    taa_info["filename"],
                )
                taas.append(taa)
                self._save_taa(taa)

            except FetchError as e:
                self.log.error("taa_fetch_failed", taa=taa_info["taa_number"], error=str(e))
            except Exception as e:
                self.log.error("taa_parse_failed", taa=taa_info["taa_number"], error=str(e))

            # Rate limit
            if i < len(taa_index) - 1:
                import asyncio
                await asyncio.sleep(delay)

        return taas

    async def scrape_by_url(self, url: str) -> RawTAA:
        """Scrape a single TAA by URL.

        Args:
            url: URL to the TAA PDF

        Returns:
            RawTAA object
        """
        filename = url.split("/")[-1]
        taa_number = self._extract_taa_number(filename)

        taa = await self.scrape_taa(url, taa_number, filename)
        self._save_taa(taa)

        return taa

    async def scrape(
        self,
        max_count: int | None = None,
        delay: float | None = None,
    ) -> list[dict[str, Any]]:
        """Main scrape method.

        Args:
            max_count: Maximum number of TAAs to scrape
            delay: Delay between requests

        Returns:
            List of scraped TAA dictionaries
        """
        taas = await self.scrape_from_index(max_count=max_count, delay=delay)

        # Save summary
        summary = {
            "total_taas": len(taas),
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "taas": [taa.metadata.taa_number for taa in taas],
        }
        self.save_raw(summary, "taa/summary.json")

        return [taa.model_dump() for taa in taas]
