"""Tests for src/scrapers/base.py."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.scrapers.base import (
    USER_AGENTS,
    BaseScraper,
    FetchError,
    ScraperError,
)


# =============================================================================
# Concrete Implementation for Testing
# =============================================================================


class ConcreteScraper(BaseScraper):
    """Concrete implementation for testing abstract BaseScraper."""

    async def scrape(self) -> list[dict[str, Any]]:
        """Minimal scrape implementation."""
        return [{"test": "data"}]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Create temporary cache and raw data directories."""
    cache_dir = tmp_path / "cache"
    raw_data_dir = tmp_path / "raw"
    cache_dir.mkdir()
    raw_data_dir.mkdir()
    return cache_dir, raw_data_dir


@pytest.fixture
def scraper(temp_dirs: tuple[Path, Path]) -> ConcreteScraper:
    """Create a scraper with temp directories."""
    cache_dir, raw_data_dir = temp_dirs
    return ConcreteScraper(
        rate_limit_delay=0.01,  # Fast for tests
        timeout=5.0,
        max_retries=3,
        use_cache=True,
        cache_dir=cache_dir,
        raw_data_dir=raw_data_dir,
    )


@pytest.fixture
def scraper_no_cache(temp_dirs: tuple[Path, Path]) -> ConcreteScraper:
    """Create a scraper with caching disabled."""
    cache_dir, raw_data_dir = temp_dirs
    return ConcreteScraper(
        rate_limit_delay=0.01,
        timeout=5.0,
        use_cache=False,
        cache_dir=cache_dir,
        raw_data_dir=raw_data_dir,
    )


# =============================================================================
# Exception Tests
# =============================================================================


class TestExceptions:
    """Test scraper exceptions."""

    def test_scraper_error_is_exception(self) -> None:
        """ScraperError should be an Exception."""
        assert issubclass(ScraperError, Exception)

    def test_fetch_error_is_scraper_error(self) -> None:
        """FetchError should be a ScraperError."""
        assert issubclass(FetchError, ScraperError)

    def test_fetch_error_message(self) -> None:
        """FetchError should preserve message."""
        error = FetchError("test message")
        assert str(error) == "test message"


# =============================================================================
# Initialization Tests
# =============================================================================


class TestScraperInit:
    """Test scraper initialization."""

    def test_creates_directories(self, tmp_path: Path) -> None:
        """Scraper should create cache and raw data directories."""
        cache_dir = tmp_path / "new_cache"
        raw_dir = tmp_path / "new_raw"

        scraper = ConcreteScraper(cache_dir=cache_dir, raw_data_dir=raw_dir)

        assert cache_dir.exists()
        assert raw_dir.exists()

    def test_default_parameters(self, temp_dirs: tuple[Path, Path]) -> None:
        """Scraper should use default parameters."""
        cache_dir, raw_data_dir = temp_dirs
        scraper = ConcreteScraper(cache_dir=cache_dir, raw_data_dir=raw_data_dir)

        assert scraper.rate_limit_delay == 1.0
        assert scraper.timeout == 30.0
        assert scraper.max_retries == 3
        assert scraper.use_cache is True

    def test_custom_parameters(self, temp_dirs: tuple[Path, Path]) -> None:
        """Scraper should accept custom parameters."""
        cache_dir, raw_data_dir = temp_dirs
        scraper = ConcreteScraper(
            rate_limit_delay=2.5,
            timeout=60.0,
            max_retries=5,
            use_cache=False,
            cache_dir=cache_dir,
            raw_data_dir=raw_data_dir,
        )

        assert scraper.rate_limit_delay == 2.5
        assert scraper.timeout == 60.0
        assert scraper.max_retries == 5
        assert scraper.use_cache is False


# =============================================================================
# User Agent Tests
# =============================================================================


class TestUserAgentRotation:
    """Test user agent rotation."""

    def test_get_next_user_agent_returns_string(
        self, scraper: ConcreteScraper
    ) -> None:
        """_get_next_user_agent should return a user agent string."""
        ua = scraper._get_next_user_agent()
        assert isinstance(ua, str)
        assert "Mozilla" in ua

    def test_rotates_through_agents(self, scraper: ConcreteScraper) -> None:
        """Should rotate through all user agents."""
        agents = [scraper._get_next_user_agent() for _ in range(len(USER_AGENTS))]
        assert agents == USER_AGENTS

    def test_wraps_around(self, scraper: ConcreteScraper) -> None:
        """Should wrap around to first agent after all are used."""
        for _ in range(len(USER_AGENTS)):
            scraper._get_next_user_agent()

        first_agent = scraper._get_next_user_agent()
        assert first_agent == USER_AGENTS[0]


# =============================================================================
# Cache Key Tests
# =============================================================================


class TestCacheKey:
    """Test cache key generation."""

    def test_cache_key_is_sha256(self, scraper: ConcreteScraper) -> None:
        """Cache key should be a SHA256 hash."""
        key = scraper._get_cache_key("https://example.com")
        assert len(key) == 64  # SHA256 produces 64 hex chars
        assert all(c in "0123456789abcdef" for c in key)

    def test_same_url_same_key(self, scraper: ConcreteScraper) -> None:
        """Same URL should produce same cache key."""
        url = "https://example.com/page"
        key1 = scraper._get_cache_key(url)
        key2 = scraper._get_cache_key(url)
        assert key1 == key2

    def test_different_urls_different_keys(self, scraper: ConcreteScraper) -> None:
        """Different URLs should produce different cache keys."""
        key1 = scraper._get_cache_key("https://example.com/page1")
        key2 = scraper._get_cache_key("https://example.com/page2")
        assert key1 != key2


# =============================================================================
# Cache Path Tests
# =============================================================================


class TestCachePath:
    """Test cache path generation."""

    def test_cache_path_in_cache_dir(self, scraper: ConcreteScraper) -> None:
        """Cache path should be in cache directory."""
        path = scraper._get_cache_path("https://example.com")
        assert path.parent == scraper.cache_dir

    def test_cache_path_has_html_extension(self, scraper: ConcreteScraper) -> None:
        """Cache path should have .html extension."""
        path = scraper._get_cache_path("https://example.com")
        assert path.suffix == ".html"


# =============================================================================
# Cache Read/Write Tests
# =============================================================================


class TestCacheReadWrite:
    """Test cache read/write operations."""

    def test_load_from_cache_returns_none_when_empty(
        self, scraper: ConcreteScraper
    ) -> None:
        """_load_from_cache should return None when cache is empty."""
        result = scraper._load_from_cache("https://nonexistent.com")
        assert result is None

    def test_save_and_load_cache(self, scraper: ConcreteScraper) -> None:
        """Should save and load content from cache."""
        url = "https://example.com/test"
        content = "<html><body>Test content</body></html>"

        scraper._save_to_cache(url, content)
        loaded = scraper._load_from_cache(url)

        assert loaded == content

    def test_cache_disabled_does_not_save(
        self, scraper_no_cache: ConcreteScraper
    ) -> None:
        """Should not save to cache when caching is disabled."""
        url = "https://example.com/test"
        content = "<html>Test</html>"

        scraper_no_cache._save_to_cache(url, content)
        cache_path = scraper_no_cache._get_cache_path(url)

        assert not cache_path.exists()

    def test_cache_disabled_returns_none(
        self, scraper_no_cache: ConcreteScraper
    ) -> None:
        """Should return None when caching is disabled."""
        # Manually create a cache file
        url = "https://example.com/test"
        cache_path = scraper_no_cache._get_cache_path(url)
        cache_path.write_text("<html>Cached</html>")

        result = scraper_no_cache._load_from_cache(url)
        assert result is None


# =============================================================================
# HTTP Client Tests
# =============================================================================


class TestHttpClient:
    """Test HTTP client management."""

    async def test_get_client_creates_client(self, scraper: ConcreteScraper) -> None:
        """_get_client should create an httpx.AsyncClient."""
        client = await scraper._get_client()
        assert isinstance(client, httpx.AsyncClient)
        await scraper.close()

    async def test_get_client_reuses_client(self, scraper: ConcreteScraper) -> None:
        """_get_client should reuse existing client."""
        client1 = await scraper._get_client()
        client2 = await scraper._get_client()
        assert client1 is client2
        await scraper.close()

    async def test_close_closes_client(self, scraper: ConcreteScraper) -> None:
        """close() should close the client."""
        client = await scraper._get_client()
        await scraper.close()
        assert client.is_closed

    async def test_close_when_no_client(self, scraper: ConcreteScraper) -> None:
        """close() should not error when no client exists."""
        await scraper.close()  # Should not raise


# =============================================================================
# Fetch Tests
# =============================================================================


class TestFetchWithRetry:
    """Test fetch with retry functionality."""

    async def test_fetch_success(self, scraper: ConcreteScraper) -> None:
        """Successful fetch should return content."""
        mock_response = MagicMock()
        mock_response.text = "<html>Success</html>"
        mock_response.raise_for_status = MagicMock()

        with patch.object(scraper, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await scraper._fetch_with_retry("https://example.com")

        assert result == "<html>Success</html>"

    async def test_fetch_raises_on_http_error(self, scraper: ConcreteScraper) -> None:
        """Fetch should raise on HTTP error."""
        with patch.object(scraper, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "Error",
                    request=MagicMock(),
                    response=MagicMock(status_code=500),
                )
            )
            mock_get_client.return_value = mock_client

            with pytest.raises(httpx.HTTPStatusError):
                await scraper._fetch_with_retry("https://example.com")


class TestFetchPage:
    """Test fetch_page functionality."""

    async def test_uses_cache_hit(self, scraper: ConcreteScraper) -> None:
        """fetch_page should return cached content on cache hit."""
        url = "https://example.com/cached"
        cached_content = "<html>Cached</html>"
        scraper._save_to_cache(url, cached_content)

        result = await scraper.fetch_page(url)

        assert result == cached_content

    async def test_fetches_on_cache_miss(self, scraper: ConcreteScraper) -> None:
        """fetch_page should fetch and cache on cache miss."""
        url = "https://example.com/new"
        content = "<html>Fresh</html>"

        with patch.object(
            scraper, "_fetch_with_retry", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.return_value = content

            result = await scraper.fetch_page(url)

        assert result == content
        # Verify it was cached
        cached = scraper._load_from_cache(url)
        assert cached == content

    async def test_raises_fetch_error(self, scraper: ConcreteScraper) -> None:
        """fetch_page should raise FetchError on failure."""
        url = "https://example.com/error"

        with patch.object(
            scraper, "_fetch_with_retry", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.side_effect = Exception("Network error")

            with pytest.raises(FetchError) as exc_info:
                await scraper.fetch_page(url)

        assert "Failed to fetch" in str(exc_info.value)
        assert url in str(exc_info.value)


class TestFetchWithRateLimit:
    """Test rate-limited batch fetching."""

    async def test_fetches_all_urls(self, scraper: ConcreteScraper) -> None:
        """fetch_with_rate_limit should fetch all URLs."""
        urls = [
            "https://example.com/1",
            "https://example.com/2",
            "https://example.com/3",
        ]

        with patch.object(
            scraper, "fetch_page", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.side_effect = ["<html>1</html>", "<html>2</html>", "<html>3</html>"]

            results = await scraper.fetch_with_rate_limit(urls)

        assert len(results) == 3
        assert results[0] == ("https://example.com/1", "<html>1</html>")
        assert results[1] == ("https://example.com/2", "<html>2</html>")
        assert results[2] == ("https://example.com/3", "<html>3</html>")

    async def test_handles_failed_fetches(self, scraper: ConcreteScraper) -> None:
        """fetch_with_rate_limit should return None for failed fetches."""
        urls = ["https://example.com/1", "https://example.com/2"]

        with patch.object(
            scraper, "fetch_page", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.side_effect = ["<html>1</html>", FetchError("Failed")]

            results = await scraper.fetch_with_rate_limit(urls)

        assert len(results) == 2
        assert results[0] == ("https://example.com/1", "<html>1</html>")
        assert results[1] == ("https://example.com/2", None)

    async def test_respects_custom_delay(self, scraper: ConcreteScraper) -> None:
        """fetch_with_rate_limit should use custom delay."""
        urls = ["https://example.com/1", "https://example.com/2"]

        with patch.object(
            scraper, "fetch_page", new_callable=AsyncMock
        ) as mock_fetch:
            mock_fetch.return_value = "<html>Test</html>"

            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                await scraper.fetch_with_rate_limit(urls, delay=0.5)

        # Should only sleep between requests (not after last one)
        mock_sleep.assert_called_once_with(0.5)


# =============================================================================
# Raw Data Persistence Tests
# =============================================================================


class TestRawDataPersistence:
    """Test raw data save/load operations."""

    def test_save_raw_creates_file(self, scraper: ConcreteScraper) -> None:
        """save_raw should create a JSON file."""
        data = {"key": "value"}
        path = scraper.save_raw(data, "test.json")

        assert path.exists()
        assert path.name == "test.json"

    def test_save_raw_adds_metadata(self, scraper: ConcreteScraper) -> None:
        """save_raw should add scraper metadata."""
        data = {"key": "value"}
        path = scraper.save_raw(data, "test.json")

        with open(path) as f:
            saved_data = json.load(f)

        assert "_scraped_at" in saved_data
        assert "_scraper" in saved_data
        assert saved_data["_scraper"] == "ConcreteScraper"
        assert saved_data["key"] == "value"

    def test_save_raw_creates_subdirectories(self, scraper: ConcreteScraper) -> None:
        """save_raw should create parent directories."""
        data = {"key": "value"}
        path = scraper.save_raw(data, "subdir/nested/test.json")

        assert path.exists()
        assert "subdir/nested" in str(path)

    def test_load_raw_returns_data(self, scraper: ConcreteScraper) -> None:
        """load_raw should return saved data."""
        data = {"key": "value", "nested": {"inner": 123}}
        scraper.save_raw(data, "test.json")

        loaded = scraper.load_raw("test.json")

        assert loaded["key"] == "value"
        assert loaded["nested"]["inner"] == 123

    def test_load_raw_raises_file_not_found(self, scraper: ConcreteScraper) -> None:
        """load_raw should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            scraper.load_raw("nonexistent.json")


# =============================================================================
# Context Manager Tests
# =============================================================================


class TestContextManager:
    """Test async context manager functionality."""

    async def test_async_enter_returns_scraper(
        self, temp_dirs: tuple[Path, Path]
    ) -> None:
        """__aenter__ should return the scraper instance."""
        cache_dir, raw_data_dir = temp_dirs
        scraper = ConcreteScraper(cache_dir=cache_dir, raw_data_dir=raw_data_dir)

        async with scraper as s:
            assert s is scraper

    async def test_async_exit_closes_client(
        self, temp_dirs: tuple[Path, Path]
    ) -> None:
        """__aexit__ should close the HTTP client."""
        cache_dir, raw_data_dir = temp_dirs
        scraper = ConcreteScraper(cache_dir=cache_dir, raw_data_dir=raw_data_dir)

        async with scraper:
            client = await scraper._get_client()
            assert not client.is_closed

        assert client.is_closed

    async def test_context_manager_handles_exception(
        self, temp_dirs: tuple[Path, Path]
    ) -> None:
        """Context manager should close client even on exception."""
        cache_dir, raw_data_dir = temp_dirs
        scraper = ConcreteScraper(cache_dir=cache_dir, raw_data_dir=raw_data_dir)

        with pytest.raises(ValueError):
            async with scraper:
                client = await scraper._get_client()
                raise ValueError("Test error")

        assert client.is_closed


# =============================================================================
# Abstract Method Tests
# =============================================================================


class TestAbstractMethods:
    """Test abstract method implementation."""

    async def test_scrape_is_implemented(self, scraper: ConcreteScraper) -> None:
        """Concrete scraper should implement scrape()."""
        result = await scraper.scrape()
        assert result == [{"test": "data"}]
