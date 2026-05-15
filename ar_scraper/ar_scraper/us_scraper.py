"""US scraper using SEC EDGAR for 10-K filings."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import aiohttp

from ar_scraper.base import FilingMetadata, ScraperResult
from ar_scraper.config import (
    DEFAULT_PDF_WORKERS,
    SEC_ARCHIVES_URL,
    SEC_BASE_URL,
    SEC_USER_AGENT,
    US_RATE_LIMIT,
)
from ar_scraper.utils import get_existing_symbols, save_json


@dataclass
class FilingInfo:
    """SEC filing metadata."""

    ticker: str
    cik: str
    company_name: str
    form_type: str
    filing_date: str
    accession_number: str
    primary_document: str

    @property
    def document_url(self) -> str:
        """Construct document URL."""
        acc = self.accession_number.replace("-", "")
        return f"{SEC_ARCHIVES_URL}/{self.cik}/{acc}/{self.primary_document}"

    @property
    def base_url(self) -> str:
        """Construct base URL for relative links."""
        acc = self.accession_number.replace("-", "")
        return f"{SEC_ARCHIVES_URL}/{self.cik}/{acc}/"


class AsyncSECClient:
    """Async SEC EDGAR client with rate limiting."""

    def __init__(self, rate_limit: int = US_RATE_LIMIT):
        """Initialize client."""
        self.headers = {"User-Agent": SEC_USER_AGENT}
        self.ticker_to_cik: dict[str, str] = {}
        self.semaphore = asyncio.Semaphore(rate_limit)
        self.min_interval = 1.0 / rate_limit
        self.last_request = 0.0
        self._lock = asyncio.Lock()

    async def _rate_limited_get(
        self, session: aiohttp.ClientSession, url: str, timeout: int = 60
    ) -> aiohttp.ClientResponse:
        """GET with rate limiting and 429 retry."""
        backoffs = [10.0, 30.0, 60.0, 120.0]
        for attempt, backoff in enumerate([0.0] + backoffs):
            if backoff:
                await asyncio.sleep(backoff)
            async with self.semaphore:
                async with self._lock:
                    now = time.time()
                    wait = self.min_interval - (now - self.last_request)
                    if wait > 0:
                        await asyncio.sleep(wait)
                    self.last_request = time.time()
                resp = await session.get(url, headers=self.headers, timeout=timeout)
            if resp.status != 429:
                return resp
            resp.close()
        return resp

    async def load_ticker_mapping(self, session: aiohttp.ClientSession) -> None:
        """Load SEC ticker to CIK mapping."""
        url = "https://www.sec.gov/files/company_tickers.json"
        async with await self._rate_limited_get(session, url) as resp:
            data = await resp.json()
            self.ticker_to_cik = {
                v["ticker"]: str(v["cik_str"]).zfill(10) for v in data.values()
            }

    async def fetch_filing(
        self, session: aiohttp.ClientSession, ticker: str, year: Optional[int] = None
    ) -> tuple[str, Optional[FilingInfo], Optional[str], Optional[str]]:
        """Fetch filing info and HTML for a ticker."""
        cik = self.ticker_to_cik.get(ticker.upper())
        if not cik:
            return (ticker, None, None, "No CIK mapping")

        try:
            url = f"{SEC_BASE_URL}/submissions/CIK{cik}.json"
            async with await self._rate_limited_get(session, url) as resp:
                if resp.status != 200:
                    return (ticker, None, None, f"HTTP {resp.status}")
                data = await resp.json()
        except Exception as e:
            return (ticker, None, None, f"Info fetch: {e}")

        filings = data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        info = None

        for form_type in ("10-K", "20-F"):
            for i, form in enumerate(forms):
                if form == form_type:
                    filing_date = filings["filingDate"][i]
                    if year is not None:
                        filing_year = int(filing_date.split("-")[0])
                        if filing_year != year:
                            continue
                    info = FilingInfo(
                        ticker=ticker.upper(),
                        cik=cik.lstrip("0"),
                        company_name=data.get("name", ticker),
                        form_type=form_type,
                        filing_date=filing_date,
                        accession_number=filings["accessionNumber"][i],
                        primary_document=filings["primaryDocument"][i],
                    )
                    break
            if info:
                break

        if not info:
            msg = f"No 10-K/20-F found for {year}" if year else "No 10-K/20-F found"
            return (ticker, None, None, msg)

        try:
            async with await self._rate_limited_get(
                session, info.document_url, timeout=90
            ) as resp:
                if resp.status != 200:
                    return (ticker, info, None, f"HTML HTTP {resp.status}")
                html = await resp.text()
                if len(html) < 10000:
                    return (ticker, info, None, "HTML too short")
                return (ticker, info, html, None)
        except Exception as e:
            return (ticker, info, None, f"HTML fetch: {e}")


def _init_pdf_worker() -> None:
    """Initialize PDF worker process."""
    if sys.platform == "darwin":
        lib = "/opt/homebrew/lib"
        if os.path.exists(lib):
            current = os.environ.get("DYLD_LIBRARY_PATH", "")
            os.environ["DYLD_LIBRARY_PATH"] = f"{lib}:{current}"


def _convert_to_pdf(args: tuple) -> tuple[str, Optional[bytes], Optional[str]]:
    """Convert HTML to PDF (worker function)."""
    try:
        from weasyprint import HTML, CSS
        from weasyprint.text.fonts import FontConfiguration
    except ImportError:
        return (args[0], None, "WeasyPrint not installed. Run: pip install ar-scraper[us]")

    ticker, html, base_url = args
    try:
        font_config = FontConfiguration()

        if "<base" not in html.lower():
            html = re.sub(
                r"(<head[^>]*>)",
                rf'\1<base href="{base_url}">',
                html,
                count=1,
                flags=re.IGNORECASE,
            )

        css = CSS(
            string="""
            @page { size: letter; margin: 0.5in; }
            body { font-size: 10pt; line-height: 1.4; }
            table { border-collapse: collapse; width: 100%; font-size: 9pt; }
            td, th { border: 1px solid #ddd; padding: 4px; }
        """,
            font_config=font_config,
        )

        pdf = HTML(string=html, base_url=base_url).write_pdf(
            stylesheets=[css], font_config=font_config
        )
        return (ticker, pdf, None)
    except Exception as e:
        return (ticker, None, str(e))


async def _fetch_all(
    client: AsyncSECClient,
    symbols: list[str],
    year: Optional[int],
    verbose: bool,
) -> tuple[list[tuple[FilingInfo, str]], dict[str, str]]:
    """Fetch all filings asynchronously."""
    results: list[tuple[FilingInfo, str]] = []
    errors: dict[str, str] = {}

    async with aiohttp.ClientSession() as session:
        if verbose:
            print("US: Loading ticker mapping...")
        await client.load_ticker_mapping(session)

        tasks = [
            asyncio.create_task(client.fetch_filing(session, sym, year=year))
            for sym in symbols
        ]

        done = 0
        total = len(tasks)
        for task in asyncio.as_completed(tasks):
            ticker, info, html, error = await task
            done += 1

            if error:
                errors[ticker] = error
                if verbose:
                    print(f"[{done}/{total}] {ticker}: ERROR - {error}")
            elif info and html:
                results.append((info, html))
                if verbose:
                    kb = len(html) / 1024
                    print(
                        f"[{done}/{total}] {ticker}: OK - {info.form_type} "
                        f"({info.filing_date}, {kb:.0f}KB)"
                    )

    return results, errors


def _convert_all_to_pdf(
    fetched: list[tuple[FilingInfo, str]],
    output_dir: Path,
    max_workers: int,
    verbose: bool,
) -> tuple[list[FilingInfo], dict[str, str]]:
    """Convert all HTML to PDF using multiprocessing."""
    if not fetched:
        return [], {}

    if verbose:
        print(f"US: Converting {len(fetched)} to PDF (workers={max_workers})")

    args_list = [(f.ticker, html, f.base_url) for f, html in fetched]
    ticker_to_info = {f.ticker: f for f, _ in fetched}

    successful: list[FilingInfo] = []
    errors: dict[str, str] = {}

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=max_workers, mp_context=ctx, initializer=_init_pdf_worker
    ) as executor:
        futures = {executor.submit(_convert_to_pdf, a): a[0] for a in args_list}

        done = 0
        total = len(futures)
        for future in as_completed(futures):
            ticker = futures[future]
            done += 1

            try:
                _, pdf_bytes, error = future.result()
                if error:
                    errors[ticker] = error
                    if verbose:
                        print(f"[{done}/{total}] {ticker}: PDF ERROR - {error[:60]}")
                elif pdf_bytes:
                    info = ticker_to_info[ticker]
                    path = output_dir / f"{info.ticker}_{info.form_type}_{info.filing_date}.pdf"
                    path.write_bytes(pdf_bytes)
                    successful.append(info)
                    if verbose:
                        print(f"[{done}/{total}] {ticker}: PDF OK ({len(pdf_bytes)/1024:.0f}KB)")
            except Exception as e:
                errors[ticker] = str(e)
                if verbose:
                    print(f"[{done}/{total}] {ticker}: WORKER ERROR - {e}")

    return successful, errors


def scrape_us(
    symbols: list[str],
    output_dir: Path,
    year: Optional[int] = None,
    skip_existing: bool = True,
    verbose: bool = True,
    pdf_workers: int = DEFAULT_PDF_WORKERS,
) -> ScraperResult:
    """Scrape US 10-K filings from SEC EDGAR.

    Args:
        symbols: List of ticker symbols (e.g., ["AAPL", "MSFT"])
        output_dir: Directory to save PDFs
        year: Optional year filter (filters by filing_date year)
        skip_existing: Skip already downloaded symbols
        verbose: Print progress
        pdf_workers: Number of PDF conversion workers

    Returns:
        ScraperResult with successful metadata and errors
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = get_existing_symbols(output_dir) if skip_existing else set()
    to_process = [s for s in symbols if s not in existing]

    if verbose:
        year_str = f" for {year}" if year else ""
        print(f"US: Processing {len(to_process)} symbols{year_str} ({len(existing)} skipped)")

    if not to_process:
        return ScraperResult(successful=[], errors={})

    # Phase 1: Async fetch
    if verbose:
        print("US: Phase 1 - Fetching from SEC EDGAR")

    client = AsyncSECClient()
    fetched, fetch_errors = asyncio.run(
        _fetch_all(client, to_process, year, verbose)
    )

    # Phase 2: PDF conversion
    if verbose:
        print("US: Phase 2 - Converting to PDF")

    converted, pdf_errors = _convert_all_to_pdf(
        fetched, output_dir, pdf_workers, verbose
    )

    # Build results
    successful: list[FilingMetadata] = []
    for info in converted:
        meta = FilingMetadata(
            symbol=info.ticker,
            region="us",
            company_name=info.company_name,
            form_type=info.form_type,
            filing_date=info.filing_date,
            filename=f"{info.ticker}_{info.form_type}_{info.filing_date}.pdf",
            extra=asdict(info),
        )
        successful.append(meta)

    all_errors = {**fetch_errors, **pdf_errors}

    if successful:
        save_json(
            {m.symbol: asdict(m) for m in successful},
            output_dir / "metadata.json",
        )
    if all_errors:
        save_json(all_errors, output_dir / "errors.json")

    return ScraperResult(successful=successful, errors=all_errors)
