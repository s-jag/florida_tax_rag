"""Florida Case Law scraper using CourtListener API."""

from __future__ import annotations

import asyncio
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote_plus

from .base import BaseScraper, FetchError
from .models import CaseMetadata, RawCase
from .utils import parse_statute_citation


class FloridaCaseLawScraper(BaseScraper):
    """Scraper for Florida tax-related case law from CourtListener.

    Uses the free CourtListener REST API to search and retrieve
    Florida court cases related to tax law.
    """

    BASE_URL = "https://www.courtlistener.com/api/rest/v4"
    SEARCH_URL = f"{BASE_URL}/search/"

    # Florida court IDs in CourtListener
    FLORIDA_COURTS = {
        "fla": "Supreme Court of Florida",
        "flaapp": "Florida District Courts of Appeal",
        "fladistctapp": "Florida District Court of Appeal",
    }

    # Default search queries for tax cases
    TAX_QUERIES = [
        '"department of revenue"',
        '"sales tax"',
        '"use tax"',
        '"corporate income tax"',
        '"documentary stamp tax"',
        '"property tax" assessment',
    ]

    def __init__(
        self,
        rate_limit_delay: float = 0.5,
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
        self.output_dir = self.raw_data_dir / "case_law"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def search_cases(
        self,
        query: str,
        court: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        """Search for cases using the CourtListener API.

        Args:
            query: Search query string
            court: Court ID filter (e.g., 'fla', 'flaapp')
            page: Page number (1-indexed)
            page_size: Results per page (max 20)

        Returns:
            API response dict with 'count', 'next', 'previous', 'results'
        """
        import json
        import subprocess

        # Build URL with query parameters - encode properly for shell
        encoded_query = quote_plus(query)
        url = f"{self.SEARCH_URL}?q={encoded_query}&type=o&order_by=dateFiled%20desc&page={page}"

        if court:
            url += f"&court={court}"

        self.log.debug("searching_courtlistener", url=url)

        # Use curl to fetch (CourtListener may block httpx)
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["curl", "-s", url, "-H", "Accept: application/json"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0 and result.stdout.strip():
                try:
                    data = json.loads(result.stdout)
                    self.log.debug("search_results", count=data.get("count", 0))
                    return data
                except json.JSONDecodeError as e:
                    self.log.error("json_decode_error", error=str(e), output=result.stdout[:200])
                    return {"count": 0, "results": []}
            self.log.warning("curl_failed", returncode=result.returncode, stderr=result.stderr[:200] if result.stderr else "")
            return {"count": 0, "results": []}
        except Exception as e:
            self.log.error("search_error", error=str(e))
            return {"count": 0, "results": []}

    async def get_cluster_details(self, cluster_id: int) -> dict[str, Any]:
        """Get detailed information about a case cluster.

        Args:
            cluster_id: CourtListener cluster ID

        Returns:
            Cluster details dict
        """
        import json

        url = f"{self.BASE_URL}/clusters/{cluster_id}/"

        import subprocess
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["curl", "-s", url, "-H", "Accept: application/json"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                return json.loads(result.stdout)
            return {}
        except Exception as e:
            self.log.error("cluster_fetch_error", cluster_id=cluster_id, error=str(e))
            return {}

    async def get_opinion_text(self, opinion_id: int) -> str:
        """Get the full text of an opinion.

        Args:
            opinion_id: CourtListener opinion ID

        Returns:
            Opinion text or empty string
        """
        import json

        url = f"{self.BASE_URL}/opinions/{opinion_id}/"

        import subprocess
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                ["curl", "-s", url, "-H", "Accept: application/json"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                # Try different text fields
                return (
                    data.get("plain_text", "") or
                    data.get("html", "") or
                    data.get("html_lawbox", "") or
                    data.get("html_columbia", "") or
                    ""
                )
            return ""
        except Exception as e:
            self.log.error("opinion_fetch_error", opinion_id=opinion_id, error=str(e))
            return ""

    def _parse_search_result(self, result: dict[str, Any]) -> Optional[RawCase]:
        """Parse a search result into a RawCase.

        Args:
            result: CourtListener search result dict

        Returns:
            RawCase or None if parsing fails
        """
        try:
            # Extract cluster ID from URLs
            cluster_id = result.get("cluster_id", 0)
            if not cluster_id and result.get("cluster"):
                # Extract from URL like "/api/rest/v4/clusters/12345/"
                match = re.search(r"/clusters/(\d+)/", result.get("cluster", ""))
                if match:
                    cluster_id = int(match.group(1))

            if not cluster_id:
                return None

            # Parse date
            date_filed = None
            date_str = result.get("dateFiled") or result.get("date_filed")
            if date_str:
                try:
                    date_filed = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
                except ValueError:
                    pass

            # Build citations list
            citations = []
            if result.get("citation"):
                citations = result["citation"] if isinstance(result["citation"], list) else [result["citation"]]
            elif result.get("citations"):
                citations = [c.get("cite", "") for c in result.get("citations", []) if c.get("cite")]

            # Extract case names
            case_name = result.get("caseName", "") or result.get("case_name", "")
            case_name_full = result.get("case_name_full", "") or case_name

            # Extract court info
            court = result.get("court", "")
            court_id = result.get("court_id", "")
            if not court and court_id:
                court = self.FLORIDA_COURTS.get(court_id, court_id)

            # Extract snippet/text and PDF URL from opinions array
            snippet = ""
            pdf_url = None
            cases_cited = []
            opinions = result.get("opinions", [])
            if opinions and isinstance(opinions, list):
                first_opinion = opinions[0]
                snippet = first_opinion.get("snippet", "") or ""
                pdf_url = first_opinion.get("download_url")
                # Get cases cited from opinion
                opinion_cites = first_opinion.get("cites", [])
                if opinion_cites:
                    cases_cited.extend([int(c) for c in opinion_cites if str(c).isdigit()])

            # Fallback to top-level fields
            if not snippet:
                snippet = result.get("snippet", "") or result.get("text", "") or ""

            # Extract statute citations from snippet
            statutes_cited = parse_statute_citation(snippet) if snippet else []

            # Also get cases cited from top-level
            if result.get("cites"):
                for c in result.get("cites", []):
                    if str(c).isdigit() and int(c) not in cases_cited:
                        cases_cited.append(int(c))

            metadata = CaseMetadata(
                case_name=case_name,
                case_name_full=case_name_full,
                citations=citations,
                court=court,
                court_id=court_id,
                date_filed=date_filed,
                docket_number=result.get("docketNumber", "") or result.get("docket_number", ""),
                judges=result.get("judge", "") or result.get("judges", ""),
                statutes_cited=statutes_cited,
                cases_cited=cases_cited,
                cluster_id=cluster_id,
            )

            # Build source URL
            absolute_url = result.get("absolute_url", "")
            if absolute_url:
                source_url = f"https://www.courtlistener.com{absolute_url}"
            else:
                source_url = f"https://www.courtlistener.com/opinion/{cluster_id}/"

            return RawCase(
                metadata=metadata,
                opinion_text=snippet,
                opinion_html=None,
                source_url=source_url,
                pdf_url=pdf_url,
                scraped_at=datetime.now(timezone.utc),
            )
        except Exception as e:
            self.log.warning("parse_result_failed", error=str(e), result=result)
            return None

    def _save_case(self, case: RawCase) -> Path:
        """Save a case to JSON file."""
        # Create filename from cluster ID
        filename = f"case_{case.metadata.cluster_id}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(case.model_dump_json(indent=2))

        self.log.debug("saved_case", case=case.metadata.case_name, path=str(filepath))
        return filepath

    async def scrape_search_results(
        self,
        query: str,
        court: str | None = None,
        max_results: int | None = None,
        delay: float | None = None,
    ) -> list[RawCase]:
        """Scrape cases from search results using cursor-based pagination.

        Args:
            query: Search query
            court: Court filter
            max_results: Maximum number of results
            delay: Delay between requests

        Returns:
            List of RawCase objects
        """
        import json
        import subprocess

        delay = delay if delay is not None else self.rate_limit_delay
        cases = []
        total_fetched = 0

        # Initial search
        self.log.info("searching", query=query, page=1)
        results = await self.search_cases(query, court=court, page=1)

        if not results.get("results"):
            return cases

        # Process first page
        for result in results["results"]:
            case = self._parse_search_result(result)
            if case:
                cases.append(case)
                self._save_case(case)
                total_fetched += 1

                if max_results and total_fetched >= max_results:
                    self.log.info("scraped_cases", query=query, count=len(cases))
                    return cases

        # Follow cursor-based pagination
        page_num = 2
        while results.get("next"):
            await asyncio.sleep(delay)

            next_url = results["next"]
            self.log.info("searching", query=query, page=page_num, next_url=next_url[:80] + "...")

            try:
                curl_result = await asyncio.to_thread(
                    subprocess.run,
                    ["curl", "-s", next_url, "-H", "Accept: application/json"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if curl_result.returncode == 0 and curl_result.stdout.strip():
                    results = json.loads(curl_result.stdout)
                else:
                    self.log.warning("pagination_failed", page=page_num)
                    break
            except Exception as e:
                self.log.error("pagination_error", error=str(e))
                break

            if not results.get("results"):
                break

            for result in results["results"]:
                case = self._parse_search_result(result)
                if case:
                    cases.append(case)
                    self._save_case(case)
                    total_fetched += 1

                    if max_results and total_fetched >= max_results:
                        self.log.info("scraped_cases", query=query, count=len(cases))
                        return cases

            page_num += 1

        self.log.info("scraped_cases", query=query, count=len(cases))
        return cases

    async def scrape_all_tax_cases(
        self,
        courts: list[str] | None = None,
        max_per_query: int | None = None,
        delay: float | None = None,
    ) -> list[RawCase]:
        """Scrape all Florida tax-related cases.

        Args:
            courts: List of court IDs to search (default: all Florida courts)
            max_per_query: Maximum results per query
            delay: Delay between requests

        Returns:
            List of RawCase objects
        """
        courts = courts or list(self.FLORIDA_COURTS.keys())
        delay = delay if delay is not None else self.rate_limit_delay
        all_cases: dict[int, RawCase] = {}  # Deduplicate by cluster_id

        # Search with main query
        main_query = '"department of revenue"'
        for court in courts:
            self.log.info("scraping_court", court=court, query=main_query)
            cases = await self.scrape_search_results(
                main_query,
                court=court,
                max_results=max_per_query,
                delay=delay,
            )
            for case in cases:
                all_cases[case.metadata.cluster_id] = case
            await asyncio.sleep(delay)

        self.log.info("total_unique_cases", count=len(all_cases))
        return list(all_cases.values())

    async def scrape(
        self,
        query: str = '"department of revenue"',
        courts: list[str] | None = None,
        max_results: int | None = None,
        delay: float | None = None,
    ) -> list[dict[str, Any]]:
        """Main scrape method.

        Args:
            query: Search query
            courts: Court IDs to search
            max_results: Maximum results
            delay: Delay between requests

        Returns:
            List of scraped case dictionaries
        """
        courts = courts or ["fla"]
        cases = []

        for court in courts:
            court_cases = await self.scrape_search_results(
                query, court=court, max_results=max_results, delay=delay
            )
            cases.extend(court_cases)

        # Save summary
        summary = {
            "total_cases": len(cases),
            "query": query,
            "courts": courts,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "cases": [
                {
                    "cluster_id": c.metadata.cluster_id,
                    "case_name": c.metadata.case_name,
                    "date_filed": str(c.metadata.date_filed) if c.metadata.date_filed else None,
                }
                for c in cases
            ],
        }
        self.save_raw(summary, "case_law/summary.json")

        return [case.model_dump() for case in cases]
