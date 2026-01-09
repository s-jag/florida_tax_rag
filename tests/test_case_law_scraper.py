"""Tests for src/scrapers/case_law.py."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.scrapers.case_law import FloridaCaseLawScraper
from src.scrapers.models import CaseMetadata, RawCase

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
def scraper(temp_dirs: tuple[Path, Path]) -> FloridaCaseLawScraper:
    """Create a scraper instance with temp directories."""
    cache_dir, raw_data_dir = temp_dirs
    return FloridaCaseLawScraper(
        rate_limit_delay=0.01,
        timeout=5.0,
        use_cache=True,
        cache_dir=cache_dir,
        raw_data_dir=raw_data_dir,
    )


@pytest.fixture
def sample_search_response() -> dict:
    """Sample CourtListener search API response."""
    return {
        "count": 2,
        "next": None,
        "previous": None,
        "results": [
            {
                "cluster_id": 12345,
                "caseName": "Department of Revenue v. Taxpayer",
                "case_name_full": "Florida Department of Revenue v. John Taxpayer",
                "dateFiled": "2023-06-15",
                "court": "Supreme Court of Florida",
                "court_id": "fla",
                "docketNumber": "SC23-1234",
                "judge": "Justice Smith",
                "absolute_url": "/opinion/12345/department-of-revenue-v-taxpayer/",
                "citation": ["123 So.3d 456"],
                "opinions": [
                    {
                        "snippet": "The issue is whether 212.05 applies to this transaction...",
                        "download_url": "https://storage.courtlistener.com/opinion.pdf",
                        "cites": [111, 222],
                    }
                ],
            },
            {
                "cluster_id": 12346,
                "caseName": "DOR v. Corporation",
                "dateFiled": "2023-05-10",
                "court_id": "flaapp",
                "citations": [{"cite": "321 So.3d 789"}],
                "snippet": "Sales tax exemption under 212.08 was properly claimed...",
            },
        ],
    }


@pytest.fixture
def sample_search_result() -> dict:
    """Single search result for parsing tests."""
    return {
        "cluster_id": 12345,
        "caseName": "Department of Revenue v. Taxpayer",
        "case_name_full": "Florida Department of Revenue v. John Taxpayer",
        "dateFiled": "2023-06-15",
        "court": "Supreme Court of Florida",
        "court_id": "fla",
        "docketNumber": "SC23-1234",
        "judge": "Justice Smith",
        "absolute_url": "/opinion/12345/dept-revenue-taxpayer/",
        "citation": ["123 So.3d 456"],
        "opinions": [
            {
                "snippet": "The issue is whether Section 212.05 applies...",
                "download_url": "https://storage.courtlistener.com/opinion.pdf",
                "cites": [111, 222],
            }
        ],
    }


@pytest.fixture
def sample_cluster_response() -> dict:
    """Sample cluster details response."""
    return {
        "id": 12345,
        "case_name": "Department of Revenue v. Taxpayer",
        "date_filed": "2023-06-15",
        "docket_number": "SC23-1234",
        "sub_opinions": [
            {
                "id": 67890,
                "type": "lead",
                "plain_text": "This is the full opinion text...",
            }
        ],
    }


@pytest.fixture
def sample_opinion_response() -> dict:
    """Sample opinion details response."""
    return {
        "id": 67890,
        "plain_text": "This is the plain text of the opinion...",
        "html": "<p>This is the HTML version...</p>",
        "html_lawbox": "",
        "html_columbia": "",
    }


# =============================================================================
# Initialization Tests
# =============================================================================


class TestScraperInit:
    """Test scraper initialization."""

    def test_creates_output_dir(self, temp_dirs: tuple[Path, Path]) -> None:
        """Should create case_law output directory."""
        cache_dir, raw_data_dir = temp_dirs
        FloridaCaseLawScraper(
            cache_dir=cache_dir,
            raw_data_dir=raw_data_dir,
        )
        assert (raw_data_dir / "case_law").exists()

    def test_florida_courts_defined(self, scraper: FloridaCaseLawScraper) -> None:
        """FLORIDA_COURTS should contain expected courts."""
        assert "fla" in scraper.FLORIDA_COURTS
        assert "flaapp" in scraper.FLORIDA_COURTS

    def test_tax_queries_defined(self, scraper: FloridaCaseLawScraper) -> None:
        """TAX_QUERIES should contain search queries."""
        assert len(scraper.TAX_QUERIES) > 0
        assert '"department of revenue"' in scraper.TAX_QUERIES


# =============================================================================
# Search Tests
# =============================================================================


class TestSearchCases:
    """Test search_cases method."""

    async def test_returns_results(
        self, scraper: FloridaCaseLawScraper, sample_search_response: dict
    ) -> None:
        """Should return search results."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(sample_search_response)

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = mock_result

            results = await scraper.search_cases('"department of revenue"')

        assert results["count"] == 2
        assert len(results["results"]) == 2

    async def test_handles_empty_results(self, scraper: FloridaCaseLawScraper) -> None:
        """Should handle empty results."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"count": 0, "results": []})

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = mock_result

            results = await scraper.search_cases("nonexistent query")

        assert results["count"] == 0
        assert results["results"] == []

    async def test_handles_curl_failure(self, scraper: FloridaCaseLawScraper) -> None:
        """Should handle curl command failure."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Connection refused"

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = mock_result

            results = await scraper.search_cases("test query")

        assert results == {"count": 0, "results": []}

    async def test_handles_json_decode_error(self, scraper: FloridaCaseLawScraper) -> None:
        """Should handle invalid JSON response."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Invalid JSON {"

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = mock_result

            results = await scraper.search_cases("test query")

        assert results == {"count": 0, "results": []}


# =============================================================================
# Cluster/Opinion Tests
# =============================================================================


class TestGetClusterDetails:
    """Test get_cluster_details method."""

    async def test_returns_cluster_data(
        self, scraper: FloridaCaseLawScraper, sample_cluster_response: dict
    ) -> None:
        """Should return cluster details."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(sample_cluster_response)

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = mock_result

            cluster = await scraper.get_cluster_details(12345)

        assert cluster["id"] == 12345
        assert cluster["case_name"] == "Department of Revenue v. Taxpayer"

    async def test_handles_fetch_error(self, scraper: FloridaCaseLawScraper) -> None:
        """Should return empty dict on fetch error."""
        mock_result = MagicMock()
        mock_result.returncode = 1

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = mock_result

            cluster = await scraper.get_cluster_details(12345)

        assert cluster == {}


class TestGetOpinionText:
    """Test get_opinion_text method."""

    async def test_returns_plain_text(
        self, scraper: FloridaCaseLawScraper, sample_opinion_response: dict
    ) -> None:
        """Should return plain_text first."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(sample_opinion_response)

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = mock_result

            text = await scraper.get_opinion_text(67890)

        assert text == "This is the plain text of the opinion..."

    async def test_falls_back_to_html(self, scraper: FloridaCaseLawScraper) -> None:
        """Should fall back to html if plain_text empty."""
        response = {"plain_text": "", "html": "<p>HTML content</p>"}
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(response)

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = mock_result

            text = await scraper.get_opinion_text(67890)

        assert text == "<p>HTML content</p>"

    async def test_returns_empty_on_error(self, scraper: FloridaCaseLawScraper) -> None:
        """Should return empty string on error."""
        mock_result = MagicMock()
        mock_result.returncode = 1

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = mock_result

            text = await scraper.get_opinion_text(67890)

        assert text == ""


# =============================================================================
# Parse Search Result Tests
# =============================================================================


class TestParseSearchResult:
    """Test _parse_search_result method."""

    def test_parses_basic_fields(
        self, scraper: FloridaCaseLawScraper, sample_search_result: dict
    ) -> None:
        """Should parse basic case fields."""
        case = scraper._parse_search_result(sample_search_result)

        assert case is not None
        assert case.metadata.cluster_id == 12345
        assert case.metadata.case_name == "Department of Revenue v. Taxpayer"
        assert case.metadata.docket_number == "SC23-1234"

    def test_parses_date_filed(
        self, scraper: FloridaCaseLawScraper, sample_search_result: dict
    ) -> None:
        """Should parse date_filed correctly."""
        case = scraper._parse_search_result(sample_search_result)

        assert case is not None
        assert case.metadata.date_filed == date(2023, 6, 15)

    def test_parses_citations_from_array(
        self, scraper: FloridaCaseLawScraper, sample_search_result: dict
    ) -> None:
        """Should parse citations from citation array."""
        case = scraper._parse_search_result(sample_search_result)

        assert case is not None
        assert "123 So.3d 456" in case.metadata.citations

    def test_parses_citations_from_citations_field(self, scraper: FloridaCaseLawScraper) -> None:
        """Should parse citations from citations field with dicts."""
        result = {
            "cluster_id": 123,
            "caseName": "Test Case",
            "citations": [{"cite": "100 So.3d 200"}, {"cite": "101 So.3d 201"}],
        }
        case = scraper._parse_search_result(result)

        assert case is not None
        assert "100 So.3d 200" in case.metadata.citations
        assert "101 So.3d 201" in case.metadata.citations

    def test_extracts_cluster_id_from_url(self, scraper: FloridaCaseLawScraper) -> None:
        """Should extract cluster_id from cluster URL if not present."""
        result = {
            "cluster": "/api/rest/v4/clusters/99999/",
            "caseName": "Test Case",
        }
        case = scraper._parse_search_result(result)

        assert case is not None
        assert case.metadata.cluster_id == 99999

    def test_parses_opinion_snippet(
        self, scraper: FloridaCaseLawScraper, sample_search_result: dict
    ) -> None:
        """Should parse opinion snippet."""
        case = scraper._parse_search_result(sample_search_result)

        assert case is not None
        assert "Section 212.05" in case.opinion_text

    def test_parses_pdf_url(
        self, scraper: FloridaCaseLawScraper, sample_search_result: dict
    ) -> None:
        """Should parse PDF download URL."""
        case = scraper._parse_search_result(sample_search_result)

        assert case is not None
        assert case.pdf_url == "https://storage.courtlistener.com/opinion.pdf"

    def test_parses_cases_cited(
        self, scraper: FloridaCaseLawScraper, sample_search_result: dict
    ) -> None:
        """Should parse cases cited."""
        case = scraper._parse_search_result(sample_search_result)

        assert case is not None
        assert 111 in case.metadata.cases_cited
        assert 222 in case.metadata.cases_cited

    def test_builds_source_url(
        self, scraper: FloridaCaseLawScraper, sample_search_result: dict
    ) -> None:
        """Should build correct source URL."""
        case = scraper._parse_search_result(sample_search_result)

        assert case is not None
        assert "courtlistener.com" in case.source_url
        assert "/opinion/12345/" in case.source_url

    def test_returns_none_for_missing_cluster_id(self, scraper: FloridaCaseLawScraper) -> None:
        """Should return None if cluster_id cannot be determined."""
        result = {"caseName": "Test Case"}
        case = scraper._parse_search_result(result)

        assert case is None

    def test_handles_invalid_date(self, scraper: FloridaCaseLawScraper) -> None:
        """Should handle invalid date format."""
        result = {
            "cluster_id": 123,
            "caseName": "Test Case",
            "dateFiled": "invalid-date",
        }
        case = scraper._parse_search_result(result)

        assert case is not None
        assert case.metadata.date_filed is None

    def test_maps_court_id_to_name(self, scraper: FloridaCaseLawScraper) -> None:
        """Should map court_id to court name."""
        result = {
            "cluster_id": 123,
            "caseName": "Test Case",
            "court_id": "fla",
        }
        case = scraper._parse_search_result(result)

        assert case is not None
        assert case.metadata.court == "Supreme Court of Florida"


# =============================================================================
# Save Case Tests
# =============================================================================


class TestSaveCase:
    """Test _save_case method."""

    def test_saves_case_json(self, scraper: FloridaCaseLawScraper) -> None:
        """Should save case to JSON file."""
        metadata = CaseMetadata(
            case_name="Test Case",
            cluster_id=12345,
            court="Supreme Court of Florida",
            court_id="fla",
        )
        case = RawCase(
            metadata=metadata,
            opinion_text="Test opinion text",
            source_url="https://example.com",
            scraped_at=datetime.now(UTC),
        )

        path = scraper._save_case(case)

        assert path.exists()
        assert path.name == "case_12345.json"

        # Verify content
        content = json.loads(path.read_text())
        assert content["metadata"]["case_name"] == "Test Case"
        assert content["metadata"]["cluster_id"] == 12345


# =============================================================================
# Scrape Search Results Tests
# =============================================================================


class TestScrapeSearchResults:
    """Test scrape_search_results method."""

    async def test_scrapes_all_results(
        self, scraper: FloridaCaseLawScraper, sample_search_response: dict
    ) -> None:
        """Should scrape all search results."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(sample_search_response)

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = mock_result

            cases = await scraper.scrape_search_results('"department of revenue"', delay=0.001)

        assert len(cases) == 2
        assert all(isinstance(c, RawCase) for c in cases)

    async def test_respects_max_results(
        self, scraper: FloridaCaseLawScraper, sample_search_response: dict
    ) -> None:
        """Should stop at max_results."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(sample_search_response)

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = mock_result

            cases = await scraper.scrape_search_results(
                '"department of revenue"', max_results=1, delay=0.001
            )

        assert len(cases) == 1

    async def test_handles_empty_results(self, scraper: FloridaCaseLawScraper) -> None:
        """Should handle empty results gracefully."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"count": 0, "results": []})

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = mock_result

            cases = await scraper.scrape_search_results("no results", delay=0.001)

        assert cases == []


# =============================================================================
# Scrape All Tax Cases Tests
# =============================================================================


class TestScrapeAllTaxCases:
    """Test scrape_all_tax_cases method."""

    async def test_deduplicates_by_cluster_id(self, scraper: FloridaCaseLawScraper) -> None:
        """Should deduplicate cases by cluster_id."""
        # Create two responses with overlapping cluster_id
        response1 = {
            "count": 1,
            "next": None,
            "results": [
                {
                    "cluster_id": 12345,
                    "caseName": "Case 1",
                }
            ],
        }
        response2 = {
            "count": 1,
            "next": None,
            "results": [
                {
                    "cluster_id": 12345,  # Same ID
                    "caseName": "Case 1 (duplicate)",
                }
            ],
        }

        call_count = 0

        async def mock_scrape_results(query, court=None, max_results=None, delay=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [scraper._parse_search_result(response1["results"][0])]
            return [scraper._parse_search_result(response2["results"][0])]

        with patch.object(scraper, "scrape_search_results", side_effect=mock_scrape_results):
            cases = await scraper.scrape_all_tax_cases(courts=["fla", "flaapp"], delay=0.001)

        # Should only have one case due to deduplication
        assert len(cases) == 1
        assert cases[0].metadata.cluster_id == 12345


# =============================================================================
# Main Scrape Tests
# =============================================================================


class TestScrape:
    """Test main scrape method."""

    async def test_scrapes_and_saves_summary(
        self, scraper: FloridaCaseLawScraper, sample_search_response: dict
    ) -> None:
        """Should scrape cases and save summary."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(sample_search_response)

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = mock_result

            results = await scraper.scrape(
                query='"test"', courts=["fla"], max_results=10, delay=0.001
            )

        assert len(results) == 2
        assert all(isinstance(r, dict) for r in results)

        # Check summary was saved
        summary_path = scraper.raw_data_dir / "case_law" / "summary.json"
        assert summary_path.exists()

    async def test_returns_dict_results(
        self, scraper: FloridaCaseLawScraper, sample_search_response: dict
    ) -> None:
        """Should return dictionaries, not RawCase objects."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(sample_search_response)

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            mock_thread.return_value = mock_result

            results = await scraper.scrape(query='"test"', courts=["fla"], delay=0.001)

        assert all(isinstance(r, dict) for r in results)
        assert "metadata" in results[0]
