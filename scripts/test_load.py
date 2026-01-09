#!/usr/bin/env python3
"""Load testing script for the Florida Tax RAG API.

This script runs concurrent requests against the API to test performance
and rate limiting.

Usage:
    python scripts/test_load.py --num-requests 50 --concurrency 5
    python scripts/test_load.py --url http://localhost:8000/api/v1/query
"""

from __future__ import annotations

import argparse
import asyncio
import random
import time
from dataclasses import dataclass, field
from statistics import mean, stdev

import httpx

# Sample queries for testing
TEST_QUERIES = [
    "What is the Florida sales tax rate?",
    "Are groceries exempt from sales tax in Florida?",
    "What is the corporate income tax rate in Florida?",
    "How does Florida tax rental income?",
    "What items are exempt from Florida sales tax?",
    "What is the documentary stamp tax rate in Florida?",
    "How is the Florida intangibles tax calculated?",
    "What are the requirements for sales tax nexus in Florida?",
    "How does Florida tax SaaS subscriptions?",
    "What is the Florida communications services tax?",
]


@dataclass
class RequestResult:
    """Result of a single request."""

    success: bool
    status_code: int
    latency_ms: float
    error: str | None = None
    request_id: str | None = None


@dataclass
class LoadTestResults:
    """Aggregated results from load test."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    latencies: list[float] = field(default_factory=list)
    errors: dict[str, int] = field(default_factory=dict)
    status_codes: dict[int, int] = field(default_factory=dict)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time

    @property
    def requests_per_second(self) -> float:
        if self.duration_seconds > 0:
            return self.total_requests / self.duration_seconds
        return 0.0

    @property
    def success_rate(self) -> float:
        if self.total_requests > 0:
            return (self.successful_requests / self.total_requests) * 100
        return 0.0

    @property
    def avg_latency_ms(self) -> float:
        if self.latencies:
            return mean(self.latencies)
        return 0.0

    @property
    def p50_latency_ms(self) -> float:
        if self.latencies:
            sorted_latencies = sorted(self.latencies)
            idx = len(sorted_latencies) // 2
            return sorted_latencies[idx]
        return 0.0

    @property
    def p95_latency_ms(self) -> float:
        if self.latencies:
            sorted_latencies = sorted(self.latencies)
            idx = int(len(sorted_latencies) * 0.95)
            return sorted_latencies[min(idx, len(sorted_latencies) - 1)]
        return 0.0

    @property
    def p99_latency_ms(self) -> float:
        if self.latencies:
            sorted_latencies = sorted(self.latencies)
            idx = int(len(sorted_latencies) * 0.99)
            return sorted_latencies[min(idx, len(sorted_latencies) - 1)]
        return 0.0


async def make_request(
    client: httpx.AsyncClient,
    url: str,
    query: str,
    timeout: float,
) -> RequestResult:
    """Make a single request to the API."""
    start_time = time.perf_counter()

    try:
        response = await client.post(
            url,
            json={"query": query, "options": {"timeout_seconds": int(timeout)}},
            timeout=timeout,
        )
        latency_ms = (time.perf_counter() - start_time) * 1000
        success = response.status_code == 200

        request_id = response.headers.get("X-Request-ID")

        error = None
        if not success:
            try:
                error_data = response.json()
                error = error_data.get("error", str(response.status_code))
            except Exception:
                error = f"HTTP {response.status_code}"

        return RequestResult(
            success=success,
            status_code=response.status_code,
            latency_ms=latency_ms,
            error=error,
            request_id=request_id,
        )

    except httpx.TimeoutException:
        latency_ms = (time.perf_counter() - start_time) * 1000
        return RequestResult(
            success=False,
            status_code=0,
            latency_ms=latency_ms,
            error="TIMEOUT",
        )
    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        return RequestResult(
            success=False,
            status_code=0,
            latency_ms=latency_ms,
            error=str(e),
        )


async def run_load_test(
    url: str,
    num_requests: int,
    concurrency: int,
    timeout: float,
    think_time: float,
) -> LoadTestResults:
    """Run the load test with specified parameters."""
    results = LoadTestResults()
    results.start_time = time.perf_counter()

    semaphore = asyncio.Semaphore(concurrency)

    async def limited_request(query: str) -> RequestResult:
        async with semaphore:
            if think_time > 0:
                await asyncio.sleep(random.uniform(0, think_time))
            return await make_request(client, url, query, timeout)

    async with httpx.AsyncClient() as client:
        # Create tasks for all requests
        queries = [random.choice(TEST_QUERIES) for _ in range(num_requests)]
        tasks = [limited_request(q) for q in queries]

        # Run with progress indicator
        completed = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            completed += 1

            results.total_requests += 1
            if result.success:
                results.successful_requests += 1
            else:
                results.failed_requests += 1
                error_key = result.error or "UNKNOWN"
                results.errors[error_key] = results.errors.get(error_key, 0) + 1

            results.latencies.append(result.latency_ms)
            results.status_codes[result.status_code] = (
                results.status_codes.get(result.status_code, 0) + 1
            )

            # Progress indicator
            if completed % 10 == 0 or completed == num_requests:
                print(f"Progress: {completed}/{num_requests} requests completed")

    results.end_time = time.perf_counter()
    return results


def print_results(results: LoadTestResults) -> None:
    """Print formatted load test results."""
    print("\n" + "=" * 60)
    print("LOAD TEST RESULTS")
    print("=" * 60)

    print("\nSummary:")
    print(f"  Total Requests:     {results.total_requests}")
    print(f"  Successful:         {results.successful_requests}")
    print(f"  Failed:             {results.failed_requests}")
    print(f"  Success Rate:       {results.success_rate:.1f}%")
    print(f"  Duration:           {results.duration_seconds:.2f}s")
    print(f"  Requests/Second:    {results.requests_per_second:.2f}")

    print("\nLatency (ms):")
    print(f"  Average:            {results.avg_latency_ms:.0f}")
    print(f"  P50:                {results.p50_latency_ms:.0f}")
    print(f"  P95:                {results.p95_latency_ms:.0f}")
    print(f"  P99:                {results.p99_latency_ms:.0f}")
    if results.latencies:
        print(f"  Min:                {min(results.latencies):.0f}")
        print(f"  Max:                {max(results.latencies):.0f}")
        if len(results.latencies) > 1:
            print(f"  Std Dev:            {stdev(results.latencies):.0f}")

    if results.status_codes:
        print("\nStatus Codes:")
        for code, count in sorted(results.status_codes.items()):
            print(f"  {code}: {count}")

    if results.errors:
        print("\nErrors:")
        for error, count in sorted(results.errors.items(), key=lambda x: -x[1]):
            print(f"  {error}: {count}")

    print("=" * 60)


async def check_health(url: str) -> bool:
    """Check if the API is healthy before running load test."""
    health_url = url.rsplit("/", 1)[0] + "/health"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(health_url, timeout=10)
            return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False


async def get_metrics(url: str) -> dict | None:
    """Get metrics from the API."""
    metrics_url = url.rsplit("/", 1)[0] + "/metrics"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(metrics_url, timeout=10)
            if response.status_code == 200:
                return response.json()
    except Exception:
        pass
    return None


async def main():
    parser = argparse.ArgumentParser(
        description="Load test the Florida Tax RAG API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000/api/v1/query",
        help="API endpoint URL",
    )
    parser.add_argument(
        "--num-requests",
        "-n",
        type=int,
        default=50,
        help="Total number of requests to make",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=5,
        help="Number of concurrent requests",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=float,
        default=120.0,
        help="Request timeout in seconds",
    )
    parser.add_argument(
        "--think-time",
        type=float,
        default=0.0,
        help="Random delay (0 to N seconds) between requests",
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Skip initial health check",
    )

    args = parser.parse_args()

    print("Florida Tax RAG API Load Test")
    print(f"URL: {args.url}")
    print(f"Requests: {args.num_requests}, Concurrency: {args.concurrency}")
    print(f"Timeout: {args.timeout}s, Think Time: {args.think_time}s")

    # Health check
    if not args.skip_health_check:
        print("\nChecking API health...")
        if not await check_health(args.url):
            print("API health check failed. Use --skip-health-check to bypass.")
            return

    # Get initial metrics
    print("\nFetching initial metrics...")
    initial_metrics = await get_metrics(args.url)
    if initial_metrics:
        print(f"  Initial queries: {initial_metrics.get('total_queries', 'N/A')}")

    # Run load test
    print("\nStarting load test...")
    results = await run_load_test(
        url=args.url,
        num_requests=args.num_requests,
        concurrency=args.concurrency,
        timeout=args.timeout,
        think_time=args.think_time,
    )

    # Print results
    print_results(results)

    # Get final metrics
    print("\nFetching final metrics...")
    final_metrics = await get_metrics(args.url)
    if final_metrics:
        print(f"  Total queries processed: {final_metrics.get('total_queries', 'N/A')}")
        print(f"  Success rate: {final_metrics.get('success_rate_percent', 'N/A')}%")
        if "latency_ms" in final_metrics:
            print(f"  Avg latency: {final_metrics['latency_ms'].get('avg', 'N/A')}ms")


if __name__ == "__main__":
    asyncio.run(main())
