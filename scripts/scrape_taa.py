#!/usr/bin/env python3
"""CLI script to scrape Florida DOR Technical Assistance Advisements (TAAs)."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scrapers.taa import FloridaTAAScraper


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape Florida DOR Technical Assistance Advisements from floridarevenue.com"
    )
    parser.add_argument(
        "--max",
        "-m",
        type=int,
        help="Maximum number of TAAs to scrape (default: all)",
    )
    parser.add_argument(
        "--url",
        "-u",
        type=str,
        help="Scrape a single TAA by URL",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between requests in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list what would be scraped, don't download",
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
        help="Output directory (default: data/raw/taa)",
    )
    parser.add_argument(
        "--list-tax-types",
        action="store_true",
        help="List all tax type codes and their meanings",
    )

    args = parser.parse_args()

    # Handle --list-tax-types
    if args.list_tax_types:
        print("TAA Tax Type Codes:")
        print("-" * 40)
        for code, name in FloridaTAAScraper.TAX_TYPE_CODES.items():
            print(f"  {code}: {name}")
        return

    # Configure scraper
    raw_data_dir = Path(args.output).parent if args.output else None

    async with FloridaTAAScraper(
        rate_limit_delay=args.delay,
        use_cache=not args.no_cache,
        raw_data_dir=raw_data_dir,
    ) as scraper:

        # Handle --url (single TAA)
        if args.url:
            print(f"Scraping single TAA: {args.url}")
            if args.dry_run:
                print(f"  [DRY RUN] Would scrape: {args.url}")
                return

            taa = await scraper.scrape_by_url(args.url)
            print(f"  TAA Number: {taa.metadata.taa_number}")
            print(f"  Title: {taa.metadata.title}")
            print(f"  Issue Date: {taa.metadata.issue_date}")
            print(f"  Tax Type: {taa.metadata.tax_type}")
            print(f"  Topics: {', '.join(taa.metadata.topics) if taa.metadata.topics else 'None'}")
            print(f"  Statutes Cited: {len(taa.metadata.statutes_cited)}")
            print(f"  Rules Cited: {len(taa.metadata.rules_cited)}")
            print(f"  Text length: {len(taa.text)} chars")
            print(f"  PDF saved to: {taa.pdf_path}")
            return

        # Get TAA index first
        print("Fetching TAA index from Florida DOR...")
        taa_index = await scraper.get_taa_index()
        print(f"Found {len(taa_index)} TAAs in index")

        if args.dry_run:
            print("\n=== DRY RUN - Listing what would be scraped ===\n")
            count = args.max or len(taa_index)
            for i, taa_info in enumerate(taa_index[:count]):
                print(f"  [{i+1}] {taa_info['taa_number']}: {taa_info['title'][:60]}...")
                if i >= 19 and count > 20:
                    print(f"  ... and {count - 20} more")
                    break

            print(f"\nTotal: {count} TAAs to scrape")
            est_time = count * args.delay / 60
            print(f"Estimated time: {est_time:.1f} minutes at {args.delay}s delay")
            return

        # Full scrape
        max_count = args.max
        print(f"\nStarting TAA scrape...")
        print(f"Max TAAs: {max_count if max_count else 'all'}")
        print(f"Rate limit delay: {args.delay}s")
        print()

        taas = await scraper.scrape(max_count=max_count, delay=args.delay)

        print(f"\n=== Scrape Complete ===")
        print(f"Total TAAs scraped: {len(taas)}")

        # Print summary by tax type
        tax_type_counts: dict[str, int] = {}
        total_statutes = 0
        total_rules = 0

        for taa in taas:
            tax_type = taa.get("metadata", {}).get("tax_type", "Unknown")
            tax_type_counts[tax_type] = tax_type_counts.get(tax_type, 0) + 1
            total_statutes += len(taa.get("metadata", {}).get("statutes_cited", []))
            total_rules += len(taa.get("metadata", {}).get("rules_cited", []))

        print("\nBy Tax Type:")
        for tax_type, count in sorted(tax_type_counts.items()):
            print(f"  {tax_type}: {count}")

        print(f"\nTotal statute citations: {total_statutes}")
        print(f"Total rule citations: {total_rules}")


if __name__ == "__main__":
    asyncio.run(main())
