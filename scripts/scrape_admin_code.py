#!/usr/bin/env python3
"""CLI script to scrape Florida Administrative Code rules."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scrapers.admin_code import FloridaAdminCodeScraper


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape Florida Administrative Code tax rules from flrules.org"
    )
    parser.add_argument(
        "--division",
        "-d",
        type=str,
        help="Division to scrape (e.g., '12A'). Can be specified multiple times.",
        action="append",
    )
    parser.add_argument(
        "--chapter",
        "-c",
        type=str,
        help="Single chapter to scrape (e.g., '12A-1')",
    )
    parser.add_argument(
        "--rule",
        "-r",
        type=str,
        help="Single rule to scrape (e.g., '12A-1.001')",
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
        help="Output directory (default: data/raw/admin_code)",
    )
    parser.add_argument(
        "--list-chapters",
        action="store_true",
        help="List all chapters in specified division(s)",
    )

    args = parser.parse_args()

    # Configure scraper
    raw_data_dir = Path(args.output).parent if args.output else None

    async with FloridaAdminCodeScraper(
        rate_limit_delay=args.delay,
        use_cache=not args.no_cache,
        raw_data_dir=raw_data_dir,
    ) as scraper:
        # Handle --list-chapters
        if args.list_chapters:
            divisions = args.division or list(scraper.TAX_DIVISIONS.keys())
            for division in divisions:
                print(
                    f"\n=== Division {division}: {scraper.TAX_DIVISIONS.get(division, 'Unknown')} ==="
                )
                chapters = await scraper.get_division_chapters(division)
                for ch in chapters:
                    print(f"  {ch['chapter']}: {ch['title']}")
            return

        # Handle --rule (single rule)
        if args.rule:
            print(f"Scraping single rule: {args.rule}")
            if args.dry_run:
                print(f"  [DRY RUN] Would scrape: {args.rule}")
                return

            rule = await scraper.scrape_rule(args.rule)
            print(f"  Title: {rule.metadata.title}")
            print(f"  Effective: {rule.metadata.effective_date}")
            print(f"  Rulemaking Authority: {rule.metadata.rulemaking_authority}")
            print(f"  Law Implemented: {rule.metadata.law_implemented}")
            print(f"  Text length: {len(rule.text)} chars")
            return

        # Handle --chapter (single chapter)
        if args.chapter:
            print(f"Scraping chapter: {args.chapter}")
            rules_list = await scraper.get_chapter_rules(args.chapter)
            print(f"Found {len(rules_list)} rules")

            if args.dry_run:
                for r in rules_list[:10]:
                    print(f"  [DRY RUN] {r['rule_number']}: {r['title']}")
                if len(rules_list) > 10:
                    print(f"  ... and {len(rules_list) - 10} more")
                return

            rules = await scraper.scrape_chapter(args.chapter, delay=args.delay)
            print(f"\nScraped {len(rules)} rules from chapter {args.chapter}")

            # Print summary
            total_rulemaking = sum(len(r.metadata.rulemaking_authority) for r in rules)
            total_implemented = sum(len(r.metadata.law_implemented) for r in rules)
            print(f"  Total rulemaking authority citations: {total_rulemaking}")
            print(f"  Total law implemented citations: {total_implemented}")
            return

        # Handle --division or full scrape
        divisions = args.division or list(scraper.TAX_DIVISIONS.keys())

        if args.dry_run:
            print("=== DRY RUN - Listing what would be scraped ===\n")
            total_rules = 0

            for division in divisions:
                if division not in scraper.TAX_DIVISIONS:
                    print(f"Warning: Unknown division '{division}'")
                    continue

                print(f"Division {division}: {scraper.TAX_DIVISIONS[division]}")
                chapters = await scraper.get_division_chapters(division)

                for chapter_info in chapters:
                    rules = await scraper.get_chapter_rules(chapter_info["chapter"])
                    print(f"  Chapter {chapter_info['chapter']}: {len(rules)} rules")
                    total_rules += len(rules)

            print(f"\nTotal: {total_rules} rules across {len(divisions)} division(s)")
            est_time = total_rules * args.delay / 60
            print(f"Estimated time: {est_time:.1f} minutes at {args.delay}s delay")
            return

        # Full scrape
        print(f"Starting full scrape of divisions: {', '.join(divisions)}")
        print(f"Rate limit delay: {args.delay}s")
        print()

        results = await scraper.scrape(divisions=divisions, delay=args.delay)
        print("\n=== Scrape Complete ===")
        print(f"Total rules scraped: {len(results)}")


if __name__ == "__main__":
    asyncio.run(main())
