"""Base scraper infrastructure with retry logic, rate limiting, and caching."""

from __future__ import annotations

import asyncio
import hashlib
import json
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = structlog.get_logger(__name__)

# Rotating User-Agent strings to be polite and avoid blocks
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
]

# Default paths
RAW_DATA_DIR = Path("data/raw")
CACHE_DIR = Path("data/raw/.cache")


class ScraperError(Exception):
    """Base exception for scraper errors."""

    pass


class FetchError(ScraperError):
    """Raised when a page fetch fails after retries."""

    pass


class BaseScraper(ABC):
    """Abstract base class for all scrapers with common functionality."""

    def __init__(
        self,
        rate_limit_delay: float = 1.0,
        timeout: float = 30.0,
        max_retries: int = 3,
        use_cache: bool = True,
        cache_dir: Path | None = None,
        raw_data_dir: Path | None = None,
    ):
        """Initialize the scraper.

        Args:
            rate_limit_delay: Seconds to wait between requests.
            timeout: HTTP request timeout in seconds.
            max_retries: Maximum number of retry attempts.
            use_cache: Whether to cache fetched pages.
            cache_dir: Directory for cached pages.
            raw_data_dir: Directory for raw scraped data.
        """
        self.rate_limit_delay = rate_limit_delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.use_cache = use_cache
        self.cache_dir = cache_dir or CACHE_DIR
        self.raw_data_dir = raw_data_dir or RAW_DATA_DIR
        self._user_agent_index = 0
        self._client: httpx.AsyncClient | None = None

        # Ensure directories exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

        self.log = logger.bind(scraper=self.__class__.__name__)

    def _get_next_user_agent(self) -> str:
        """Rotate through user agents."""
        ua = USER_AGENTS[self._user_agent_index % len(USER_AGENTS)]
        self._user_agent_index += 1
        return ua

    def _get_cache_key(self, url: str) -> str:
        """Generate a cache key for a URL."""
        return hashlib.sha256(url.encode()).hexdigest()

    def _get_cache_path(self, url: str) -> Path:
        """Get the cache file path for a URL."""
        return self.cache_dir / f"{self._get_cache_key(url)}.html"

    def _load_from_cache(self, url: str) -> str | None:
        """Load a page from cache if it exists."""
        if not self.use_cache:
            return None

        cache_path = self._get_cache_path(url)
        if cache_path.exists():
            self.log.debug("cache_hit", url=url)
            return cache_path.read_text(encoding="utf-8")
        return None

    def _save_to_cache(self, url: str, content: str) -> None:
        """Save a page to cache."""
        if not self.use_cache:
            return

        cache_path = self._get_cache_path(url)
        cache_path.write_text(content, encoding="utf-8")
        self.log.debug("cache_saved", url=url)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _fetch_with_retry(self, url: str) -> str:
        """Fetch a page with automatic retry on failure."""
        client = await self._get_client()
        headers = {"User-Agent": self._get_next_user_agent()}

        self.log.info("fetching", url=url)
        response = await client.get(url, headers=headers)
        response.raise_for_status()

        return response.text

    async def fetch_page(self, url: str) -> str:
        """Fetch a page, using cache if available.

        Args:
            url: The URL to fetch.

        Returns:
            The page HTML content.

        Raises:
            FetchError: If the fetch fails after all retries.
        """
        # Check cache first
        cached = self._load_from_cache(url)
        if cached is not None:
            return cached

        try:
            content = await self._fetch_with_retry(url)
            self._save_to_cache(url, content)
            return content
        except Exception as e:
            self.log.error("fetch_failed", url=url, error=str(e))
            raise FetchError(f"Failed to fetch {url}: {e}") from e

    async def fetch_with_rate_limit(
        self,
        urls: list[str],
        delay: float | None = None,
    ) -> list[tuple[str, str | None]]:
        """Fetch multiple pages with rate limiting.

        Args:
            urls: List of URLs to fetch.
            delay: Seconds to wait between requests (uses instance default if None).

        Returns:
            List of (url, content) tuples. Content is None if fetch failed.
        """
        delay = delay if delay is not None else self.rate_limit_delay
        results: list[tuple[str, str | None]] = []

        for i, url in enumerate(urls):
            try:
                content = await self.fetch_page(url)
                results.append((url, content))
            except FetchError:
                results.append((url, None))

            # Rate limit (don't wait after the last request)
            if i < len(urls) - 1:
                self.log.debug("rate_limiting", delay=delay)
                await asyncio.sleep(delay)

        return results

    def save_raw(self, data: dict[str, Any], filename: str) -> Path:
        """Save raw scraped data to JSON.

        Args:
            data: The data to save.
            filename: The filename (without directory).

        Returns:
            The path to the saved file.
        """
        filepath = self.raw_data_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Add metadata
        data_with_meta = {
            "_scraped_at": datetime.now(UTC).isoformat(),
            "_scraper": self.__class__.__name__,
            **data,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data_with_meta, f, indent=2, ensure_ascii=False, default=str)

        self.log.info("saved_raw", filepath=str(filepath))
        return filepath

    def load_raw(self, filename: str) -> dict[str, Any]:
        """Load raw scraped data from JSON.

        Args:
            filename: The filename (without directory).

        Returns:
            The loaded data.

        Raises:
            FileNotFoundError: If the file doesn't exist.
        """
        filepath = self.raw_data_dir / filename
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)

    @abstractmethod
    async def scrape(self) -> list[dict[str, Any]]:
        """Scrape the data source.

        Returns:
            List of scraped items as dictionaries.
        """
        pass

    async def __aenter__(self) -> BaseScraper:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
