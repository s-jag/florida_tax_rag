#!/usr/bin/env python3
"""CLI script to scrape Florida tax case law from CourtListener."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scrapers.case_law import FloridaCaseLawScraper


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape Florida tax case law from CourtListener API"
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default='"department of revenue"',
        help='Search query (default: "department of revenue")',
    )
    parser.add_argument(
        "--court",
        "-c",
        type=str,
        action="append",
        help="Court ID to search (e.g., 'fla', 'flaapp'). Can be specified multiple times.",
    )
    parser.add_argument(
        "--max",
        "-m",
        type=int,
        help="Maximum number of cases to scrape (default: all)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between requests in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only count results, don't download",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching (re-fetch all pages)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output directory (default: data/raw/case_law)",
    )
    parser.add_argument(
        "--list-courts",
        action="store_true",
        help="List available Florida court IDs",
    )
    parser.add_argument(
        "--all-tax",
        action="store_true",
        help="Scrape all Florida tax cases (uses multiple queries)",
    )

    args = parser.parse_args()

    # Handle --list-courts
    if args.list_courts:
        print("Florida Court IDs:")
        print("-" * 50)
        for court_id, court_name in FloridaCaseLawScraper.FLORIDA_COURTS.items():
            print(f"  {court_id}: {court_name}")
        return

    # Configure scraper
    raw_data_dir = Path(args.output).parent if args.output else None

    async with FloridaCaseLawScraper(
        rate_limit_delay=args.delay,
        use_cache=not args.no_cache,
        raw_data_dir=raw_data_dir,
    ) as scraper:
        courts = args.court or ["fla"]

        if args.dry_run:
            print("=== DRY RUN - Counting results ===")
            print(f"Query: {args.query}")
            print(f"Courts: {', '.join(courts)}")
            print()

            total_count = 0
            for court in courts:
                results = await scraper.search_cases(args.query, court=court, page=1)
                count = results.get("count", 0)
                court_name = scraper.FLORIDA_COURTS.get(court, court)
                print(f"  {court_name}: {count} cases")
                total_count += count
                await asyncio.sleep(0.5)

            print(f"\nTotal: {total_count} cases")
            if args.max:
                print(
                    f"Would scrape: min({total_count}, {args.max}) = {min(total_count, args.max)} cases"
                )
            est_time = min(total_count, args.max or total_count) * args.delay / 60
            print(f"Estimated time: {est_time:.1f} minutes at {args.delay}s delay")
            return

        # Handle --all-tax
        if args.all_tax:
            print("Scraping ALL Florida tax cases...")
            print(f"Courts: {', '.join(courts)}")
            print(f"Rate limit delay: {args.delay}s")
            print()

            cases = await scraper.scrape_all_tax_cases(
                courts=courts,
                max_per_query=args.max,
                delay=args.delay,
            )

            print("\n=== Scrape Complete ===")
            print(f"Total unique cases: {len(cases)}")
            return

        # Regular scrape
        print("Scraping cases...")
        print(f"Query: {args.query}")
        print(f"Courts: {', '.join(courts)}")
        print(f"Max cases: {args.max if args.max else 'all'}")
        print(f"Rate limit delay: {args.delay}s")
        print()

        all_cases = []
        for court in courts:
            court_name = scraper.FLORIDA_COURTS.get(court, court)
            print(f"Searching {court_name}...")

            cases = await scraper.scrape_search_results(
                args.query,
                court=court,
                max_results=args.max,
                delay=args.delay,
            )
            all_cases.extend(cases)
            print(f"  Found {len(cases)} cases")

        print("\n=== Scrape Complete ===")
        print(f"Total cases scraped: {len(all_cases)}")

        # Print summary by court
        court_counts: dict[str, int] = {}
        total_statutes = 0
        total_cases_cited = 0

        for case in all_cases:
            court_id = case.metadata.court_id
            court_counts[court_id] = court_counts.get(court_id, 0) + 1
            total_statutes += len(case.metadata.statutes_cited)
            total_cases_cited += len(case.metadata.cases_cited)

        print("\nBy Court:")
        for court_id, count in sorted(court_counts.items()):
            court_name = scraper.FLORIDA_COURTS.get(court_id, court_id)
            print(f"  {court_name}: {count}")

        print(f"\nTotal statute citations: {total_statutes}")
        print(f"Total case citations: {total_cases_cited}")


if __name__ == "__main__":
    asyncio.run(main())
