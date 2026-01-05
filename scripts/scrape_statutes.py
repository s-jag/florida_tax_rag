#!/usr/bin/env python3
"""
CLI script to scrape Florida Statutes from leg.state.fl.us.

Usage:
    python scripts/scrape_statutes.py                    # Scrape all of Title XIV
    python scripts/scrape_statutes.py --chapter 212     # Scrape only Chapter 212
    python scripts/scrape_statutes.py --title 14        # Scrape Title XIV (default)
    python scripts/scrape_statutes.py --dry-run         # Show what would be scraped
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import structlog

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scrapers.statutes import (
    FloridaStatutesScraper,
    ChapterInfo,
    TITLES,
)


def setup_logging(verbose: bool = False) -> None:
    """Configure structlog for console output."""
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


async def scrape_single_chapter(
    scraper: FloridaStatutesScraper,
    chapter: int,
    title_num: int = 14,
) -> list:
    """Scrape a single chapter.

    Args:
        scraper: The scraper instance.
        chapter: Chapter number to scrape.
        title_num: Title number.

    Returns:
        List of scraped statutes.
    """
    log = structlog.get_logger()
    log.info("scraping_single_chapter", chapter=chapter, title=title_num)

    # Get chapter info
    chapters = await scraper.get_title_chapters(title_num)
    chapter_info = next((c for c in chapters if c.chapter_number == chapter), None)

    if not chapter_info:
        # Build chapter info manually if not found in index
        title_info = TITLES.get(title_num, {})
        chapter_info = ChapterInfo(
            chapter_number=chapter,
            chapter_name=f"Chapter {chapter}",
            contents_url=scraper._build_chapter_contents_url(chapter),
        )
        log.warning("chapter_not_in_index", chapter=chapter, using_manual_url=True)

    statutes = await scraper.scrape_chapter(chapter_info, title_num)
    return statutes


async def scrape_full_title(
    scraper: FloridaStatutesScraper,
    title_num: int = 14,
) -> list:
    """Scrape an entire title.

    Args:
        scraper: The scraper instance.
        title_num: Title number to scrape.

    Returns:
        List of scraped statutes.
    """
    return await scraper.scrape_title(title_num)


async def dry_run(scraper: FloridaStatutesScraper, title_num: int, chapter: int | None) -> None:
    """Show what would be scraped without actually scraping.

    Args:
        scraper: The scraper instance.
        title_num: Title number.
        chapter: Optional specific chapter.
    """
    log = structlog.get_logger()
    log.info("dry_run_starting", title=title_num, chapter=chapter)

    # Get chapters
    chapters = await scraper.get_title_chapters(title_num)

    if chapter:
        chapters = [c for c in chapters if c.chapter_number == chapter]

    print("\n" + "=" * 60)
    print(f"DRY RUN: Would scrape the following from Title {title_num}")
    print("=" * 60)

    total_sections = 0
    for ch in chapters:
        sections = await scraper.get_chapter_sections(ch)
        total_sections += len(sections)

        print(f"\nChapter {ch.chapter_number}: {ch.chapter_name}")
        print(f"  Sections: {len(sections)}")
        if sections:
            print(f"  First: {sections[0].section_number} - {sections[0].section_name[:50]}...")
            print(f"  Last:  {sections[-1].section_number} - {sections[-1].section_name[:50]}...")

        # Rate limit even in dry run
        await asyncio.sleep(0.5)

    print("\n" + "=" * 60)
    print(f"TOTAL: {len(chapters)} chapters, {total_sections} sections")
    print(f"Estimated time at 1 req/sec: ~{total_sections + len(chapters)} seconds")
    print("=" * 60 + "\n")


def print_summary(statutes: list, output_path: Path) -> None:
    """Print a summary of the scraping results.

    Args:
        statutes: List of scraped statutes.
        output_path: Path where data was saved.
    """
    print("\n" + "=" * 60)
    print("SCRAPING COMPLETE")
    print("=" * 60)

    # Group by chapter
    chapters = {}
    for s in statutes:
        ch = s.metadata.chapter
        if ch not in chapters:
            chapters[ch] = []
        chapters[ch].append(s)

    print(f"\nTotal sections scraped: {len(statutes)}")
    print(f"Chapters covered: {len(chapters)}")
    print(f"\nBreakdown by chapter:")

    for ch in sorted(chapters.keys()):
        sections = chapters[ch]
        print(f"  Chapter {ch}: {len(sections)} sections")

    print(f"\nData saved to: {output_path}")
    print("=" * 60 + "\n")


async def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, non-zero for errors).
    """
    parser = argparse.ArgumentParser(
        description="Scrape Florida Statutes from leg.state.fl.us",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/scrape_statutes.py                    # Scrape all of Title XIV
    python scripts/scrape_statutes.py --chapter 212     # Scrape only Chapter 212
    python scripts/scrape_statutes.py --dry-run         # Show what would be scraped
        """,
    )

    parser.add_argument(
        "--title",
        type=int,
        default=14,
        help="Title number to scrape (default: 14 for Taxation and Finance)",
    )
    parser.add_argument(
        "--chapter",
        type=int,
        default=None,
        help="Specific chapter to scrape (default: all chapters in title)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be scraped without actually scraping",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between requests in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/statutes"),
        help="Output directory for scraped data",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable request caching (re-fetch all pages)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    log = structlog.get_logger()

    # Validate arguments
    if args.title not in TITLES:
        log.error("unsupported_title", title=args.title, supported=list(TITLES.keys()))
        print(f"Error: Title {args.title} is not supported. Supported titles: {list(TITLES.keys())}")
        return 1

    if args.chapter:
        title_chapters = TITLES[args.title]["chapters"]
        if args.chapter not in title_chapters:
            log.warning(
                "chapter_not_in_title",
                chapter=args.chapter,
                title=args.title,
                expected_chapters=f"{title_chapters[0]}-{title_chapters[-1]}",
            )

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Initialize scraper
    async with FloridaStatutesScraper(
        rate_limit_delay=args.delay,
        output_dir=args.output,
        use_cache=not args.no_cache,
    ) as scraper:
        if args.dry_run:
            await dry_run(scraper, args.title, args.chapter)
            return 0

        # Actually scrape
        log.info(
            "starting_scrape",
            title=args.title,
            chapter=args.chapter,
            delay=args.delay,
            output=str(args.output),
        )

        start_time = datetime.now()

        if args.chapter:
            statutes = await scrape_single_chapter(scraper, args.chapter, args.title)
        else:
            statutes = await scrape_full_title(scraper, args.title)

        elapsed = datetime.now() - start_time

        # Print summary
        print_summary(statutes, args.output)
        log.info(
            "scrape_complete",
            sections=len(statutes),
            elapsed_seconds=elapsed.total_seconds(),
        )

        # Save final combined output
        output_file = args.output / f"title_{args.title}_statutes.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                [s.model_dump(mode="json") for s in statutes],
                f,
                indent=2,
                default=str,
            )
        log.info("saved_combined_output", filepath=str(output_file))

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
