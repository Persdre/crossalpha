"""CLI entry point for ar-scraper."""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

import click

from ar_scraper.config import UNIVERSES
from ar_scraper.utils import (
    get_data_path,
    group_symbols_by_region,
    load_symbols_from_csv,
)


@click.command()
@click.option(
    "--universe",
    type=click.Choice(list(UNIVERSES.keys())),
    help="Universe to scrape (e.g., us_russell_1000, jp_topix, hk_main)",
)
@click.option(
    "--symbols",
    type=str,
    help="Comma-separated symbols (e.g., AAPL,MSFT,7974.T)",
)
@click.option(
    "--symbols-file",
    type=click.Path(exists=True, path_type=Path),
    help="File with one symbol per line",
)
@click.option(
    "--year",
    type=int,
    default=datetime.now().year,
    help="Year to fetch (fiscal year for HK, filing year for others; default: current year)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("./reports"),
    help="Output directory (default: ./reports)",
)
@click.option("--limit", type=int, help="Limit number of symbols (for testing)")
@click.option("--force", is_flag=True, help="Re-download existing files")
@click.option("--verbose", is_flag=True, help="Verbose output")
@click.option("--quiet", is_flag=True, help="Minimal output")
@click.option(
    "--pdf-workers",
    type=int,
    default=10,
    help="PDF conversion workers for US (default: 10)",
)
def main(
    universe: Optional[str],
    symbols: Optional[str],
    symbols_file: Optional[Path],
    year: int,
    output_dir: Path,
    limit: Optional[int],
    force: bool,
    verbose: bool,
    quiet: bool,
    pdf_workers: int,
) -> None:
    """Annual Report Scraper - Download annual reports from US, Japan, Taiwan, Korea, Hong Kong.

    Examples:

        ar-scraper --universe us_russell_1000 --year 2024

        ar-scraper --symbols AAPL,MSFT,7974.T,005930.KS,0700.HK,600519.SS --year 2024

        ar-scraper --symbols-file my_symbols.txt --year 2024 --verbose
    """
    # Validate input
    input_count = sum([
        universe is not None,
        symbols is not None,
        symbols_file is not None,
    ])

    if input_count == 0:
        raise click.UsageError("Must specify either --universe or --symbols or --symbols-file")
    if input_count > 1:
        raise click.UsageError("Cannot combine --universe, --symbols, and --symbols-file")

    # Determine verbosity
    if quiet:
        verbose = False
    elif not verbose:
        verbose = True  # Default to verbose

    # Load symbols
    symbol_list: list[str] = []

    if universe:
        region_filter = UNIVERSES.get(universe)
        if not region_filter:
            raise click.UsageError(f"Invalid universe: {universe}")
        symbol_dict_path = get_data_path("SymbolDict.csv")
        symbol_list = load_symbols_from_csv(symbol_dict_path, region_filter)
        if verbose:
            click.echo(f"Loaded {len(symbol_list)} symbols from {universe}")

    elif symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]

    elif symbols_file:
        symbol_list = [
            line.strip().upper()
            for line in symbols_file.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]

    if not symbol_list:
        raise click.UsageError("No symbols to process")

    if limit:
        symbol_list = symbol_list[:limit]

    # Group by region
    grouped = group_symbols_by_region(symbol_list)

    if verbose:
        click.echo(f"\nAnnual Report Scraper")
        click.echo("=" * 50)
        click.echo(f"Year: {year}")
        click.echo(f"Output: {output_dir}")
        click.echo(f"Symbols: {len(symbol_list)} total")
        for region, syms in grouped.items():
            click.echo(f"  {region.upper()}: {len(syms)}")
        click.echo()

    output_dir.mkdir(parents=True, exist_ok=True)
    skip_existing = not force

    total_successful = 0
    total_errors = 0

    # Run scrapers for each region
    if "us" in grouped:
        from ar_scraper.us_scraper import scrape_us

        if verbose:
            click.echo("=" * 50)
            click.echo("US SEC EDGAR")
            click.echo("=" * 50)

        result = scrape_us(
            symbols=grouped["us"],
            output_dir=output_dir,
            year=year,
            skip_existing=skip_existing,
            verbose=verbose,
            pdf_workers=pdf_workers,
        )
        total_successful += len(result.successful)
        total_errors += len(result.errors)

    if "jp" in grouped:
        from ar_scraper.jp_scraper import scrape_jp
        from ar_scraper.mappers import EdinetMapper

        if verbose:
            click.echo("=" * 50)
            click.echo("Japan IRBank")
            click.echo("=" * 50)

        mapper = EdinetMapper(get_data_path("EdinetcodeDlInfo.csv"))
        result = asyncio.run(
            scrape_jp(
                symbols=grouped["jp"],
                output_dir=output_dir,
                mapper=mapper,
                year=year,
                skip_existing=skip_existing,
                verbose=verbose,
            )
        )
        total_successful += len(result.successful)
        total_errors += len(result.errors)

    if "tw" in grouped:
        from ar_scraper.tw_scraper import scrape_tw
        from ar_scraper.mappers import TwseMapper

        if verbose:
            click.echo("=" * 50)
            click.echo("Taiwan MOPS")
            click.echo("=" * 50)

        mapper = TwseMapper(get_data_path("SymbolDict.csv"))
        result = scrape_tw(
            symbols=grouped["tw"],
            output_dir=output_dir,
            mapper=mapper,
            year=year,
            skip_existing=skip_existing,
            verbose=verbose,
        )
        total_successful += len(result.successful)
        total_errors += len(result.errors)

    if "hk" in grouped:
        from ar_scraper.hk_scraper import scrape_hk
        from ar_scraper.mappers import HkexMapper

        if verbose:
            click.echo("=" * 50)
            click.echo("Hong Kong HKEXnews")
            click.echo("=" * 50)

        mapper = HkexMapper()
        result = scrape_hk(
            symbols=grouped["hk"],
            output_dir=output_dir,
            mapper=mapper,
            year=year,
            skip_existing=skip_existing,
            verbose=verbose,
        )
        total_successful += len(result.successful)
        total_errors += len(result.errors)

    if "kr" in grouped:
        from ar_scraper.kr_scraper import scrape_kr
        from ar_scraper.mappers import DartMapper

        if verbose:
            click.echo("=" * 50)
            click.echo("Korea DART")
            click.echo("=" * 50)

        # Load DART API key from environment
        import os

        dart_api_key = os.environ.get("DART_API_KEY", "")
        if not dart_api_key:
            # Try loading from .env file in common locations
            for env_path in [
                Path(".env"),
                Path.home() / ".env",
            ]:
                if env_path.exists():
                    for line in env_path.read_text().splitlines():
                        if line.startswith("DART_API_KEY="):
                            dart_api_key = line.split("=", 1)[1].strip()
                            break
                if dart_api_key:
                    break

        if not dart_api_key:
            click.echo("Error: DART_API_KEY not found in environment or .env file")
            click.echo("Set it with: export DART_API_KEY=your_key")
            total_errors += len(grouped["kr"])
        else:
            mapper = DartMapper(api_key=dart_api_key)
            result = scrape_kr(
                symbols=grouped["kr"],
                output_dir=output_dir,
                mapper=mapper,
                year=year,
                skip_existing=skip_existing,
                verbose=verbose,
            )
            total_successful += len(result.successful)
            total_errors += len(result.errors)

    # Summary
    if verbose or not quiet:
        click.echo()
        click.echo("=" * 50)
        click.echo(f"COMPLETE: {total_successful} OK, {total_errors} errors")
        click.echo(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
