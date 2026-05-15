"""Japan scraper using IRBank for securities reports."""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import aiohttp


from ar_scraper.base import FilingMetadata, ScraperResult
from ar_scraper.config import (
    IRBANK_HEADERS,
    JP_RATE_LIMIT,
    MAX_RETRIES,
    MIN_PDF_SIZE,
    RETRY_BACKOFF,
)
from ar_scraper.mappers import EdinetMapper
from ar_scraper.utils import get_existing_symbols, save_json


@dataclass
class DocumentInfo:
    """Information about a securities report document."""

    doc_id: str
    edinet_code: str
    period: int
    period_start: str
    period_end: str
    filing_date: str

    @property
    def pdf_url(self) -> str:
        """Construct the PDF download URL."""
        return f"https://f.irbank.net/pdf/{self.edinet_code}/ir/{self.doc_id}.pdf"


def parse_irbank_html(
    html: str, edinet_code: str, year: Optional[int] = None
) -> Optional[DocumentInfo]:
    """Parse IRBank HTML to find securities report.

    Args:
        html: HTML content from irbank.net/{edinet_code}/edinet?t=1
        edinet_code: The EDINET code
        year: Optional filing year filter. If specified, returns report filed
              in that year. If None, returns the latest report.

    Returns:
        DocumentInfo for matching annual report, or None
    """
    pattern = (
        rf'href="/{re.escape(edinet_code)}/(S100[A-Z0-9]+)"[^>]*>'
        rf'有価証券報告書-第(\d+)期\((\d{{4}}/\d{{2}}/\d{{2}})-(\d{{4}}/\d{{2}}/\d{{2}})\)'
    )

    # Find all securities reports
    matches = list(re.finditer(pattern, html))
    if not matches:
        return None

    # Build DocumentInfo for each match and filter by year if specified
    for match in matches:
        doc_id = match.group(1)
        period = int(match.group(2))
        period_start = match.group(3)
        period_end = match.group(4)

        # Extract filing date
        date_pattern = (
            rf'(\d{{4}}/\d{{2}}/\d{{2}})</dt><dd[^>]*>'
            rf'<a[^>]*href="/{re.escape(edinet_code)}/{re.escape(doc_id)}"'
        )
        date_match = re.search(date_pattern, html)
        filing_date = date_match.group(1) if date_match else period_end
        filing_date = filing_date.replace("/", "-")

        # Check year filter
        if year is not None:
            filing_year = int(filing_date.split("-")[0])
            if filing_year != year:
                continue  # Try next report

        return DocumentInfo(
            doc_id=doc_id,
            edinet_code=edinet_code,
            period=period,
            period_start=period_start,
            period_end=period_end,
            filing_date=filing_date,
        )

    return None  # No matching report found


class AsyncIRBankClient:
    """Async client for IRBank with rate limiting."""

    def __init__(self, rate_limit: int = JP_RATE_LIMIT):
        """Initialize client."""
        self.semaphore = asyncio.Semaphore(rate_limit)
        self.min_interval = 1.0 / rate_limit
        self._last_request_time = 0.0
        self._lock = asyncio.Lock()

    async def _rate_limited_request(
        self,
        session: aiohttp.ClientSession,
        url: str,
        timeout: int = 30,
    ) -> aiohttp.ClientResponse:
        """Make a rate-limited request."""
        async with self.semaphore:
            async with self._lock:
                now = time.time()
                wait = self.min_interval - (now - self._last_request_time)
                if wait > 0:
                    await asyncio.sleep(wait)
                self._last_request_time = time.time()

            return await session.get(
                url,
                headers=IRBANK_HEADERS,
                timeout=aiohttp.ClientTimeout(total=timeout),
            )

    async def fetch_document_info(
        self,
        session: aiohttp.ClientSession,
        symbol: str,
        edinet_code: str,
        year: Optional[int] = None,
    ) -> tuple[str, Optional[DocumentInfo], Optional[str]]:
        """Fetch document info for a company."""
        url = f"https://irbank.net/{edinet_code}/edinet?t=1"

        for attempt in range(MAX_RETRIES):
            try:
                async with await self._rate_limited_request(session, url) as resp:
                    if resp.status == 200:
                        html = await resp.text()
                        doc_info = parse_irbank_html(html, edinet_code, year)
                        if doc_info:
                            return (symbol, doc_info, None)
                        if year:
                            return (symbol, None, f"No report filed in {year}")
                        return (symbol, None, "No 有価証券報告書 found")
                    elif resp.status == 404:
                        return (symbol, None, "Company page not found")
                    else:
                        if attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(RETRY_BACKOFF[attempt])
                            continue
                        return (symbol, None, f"HTTP {resp.status}")
            except asyncio.TimeoutError:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_BACKOFF[attempt])
                    continue
                return (symbol, None, "Timeout")
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_BACKOFF[attempt])
                    continue
                return (symbol, None, str(e))

        return (symbol, None, "Max retries exceeded")

    async def download_pdf(
        self,
        session: aiohttp.ClientSession,
        doc_info: DocumentInfo,
        output_path: Path,
    ) -> tuple[bool, Optional[str]]:
        """Download PDF for a document."""
        for attempt in range(MAX_RETRIES):
            try:
                async with await self._rate_limited_request(
                    session, doc_info.pdf_url
                ) as resp:
                    if resp.status == 200:
                        content = await resp.read()
                        if len(content) < MIN_PDF_SIZE:
                            return False, "PDF too small"
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        output_path.write_bytes(content)
                        return True, None
                    else:
                        if attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(RETRY_BACKOFF[attempt])
                            continue
                        return False, f"HTTP {resp.status}"
            except asyncio.TimeoutError:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_BACKOFF[attempt])
                    continue
                return False, "Timeout"
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_BACKOFF[attempt])
                    continue
                return False, str(e)

        return False, "Max retries exceeded"


async def scrape_jp(
    symbols: list[str],
    output_dir: Path,
    mapper: EdinetMapper,
    year: Optional[int] = None,
    skip_existing: bool = True,
    verbose: bool = True,
) -> ScraperResult:
    """Scrape Japan securities reports from IRBank.

    Args:
        symbols: List of yfinance symbols (e.g., ["7974.T"])
        output_dir: Directory to save PDFs
        mapper: EdinetMapper instance
        year: Optional year filter (filters by filing_date year)
        skip_existing: Skip already downloaded symbols
        verbose: Print progress

    Returns:
        ScraperResult with successful metadata and errors
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = get_existing_symbols(output_dir) if skip_existing else set()
    to_process = [s for s in symbols if s not in existing]

    if verbose:
        print(f"JP: Processing {len(to_process)} symbols ({len(existing)} skipped)")

    # Map symbols to EDINET codes
    symbols_with_edinet: list[tuple[str, str]] = []
    errors: dict[str, str] = {}

    for symbol in to_process:
        edinet_code = mapper.get_edinet_code(symbol)
        if edinet_code:
            symbols_with_edinet.append((symbol, edinet_code))
        else:
            errors[symbol] = "No EDINET code mapping"

    if not symbols_with_edinet:
        return ScraperResult(successful=[], errors=errors)

    client = AsyncIRBankClient()
    successful: list[FilingMetadata] = []

    async with aiohttp.ClientSession() as session:
        # Phase 1: Fetch document info
        if verbose:
            print(f"JP: Fetching document info for {len(symbols_with_edinet)} symbols")

        tasks = [
            client.fetch_document_info(session, symbol, edinet_code, year)
            for symbol, edinet_code in symbols_with_edinet
        ]

        doc_results: list[tuple[str, DocumentInfo]] = []
        for coro in asyncio.as_completed(tasks):
            symbol, doc_info, error = await coro
            if error:
                errors[symbol] = error
            elif doc_info:
                doc_results.append((symbol, doc_info))

        # Phase 2: Download PDFs
        if verbose:
            print(f"JP: Downloading {len(doc_results)} PDFs")

        for idx, (symbol, doc_info) in enumerate(doc_results, 1):
            filename = f"{symbol}_SecReport_{doc_info.filing_date}.pdf"
            output_path = output_dir / filename

            success, error = await client.download_pdf(session, doc_info, output_path)

            if success:
                info = mapper.get_info(symbol) or {}
                meta = FilingMetadata(
                    symbol=symbol,
                    region="jp",
                    company_name=info.get("company_name", ""),
                    form_type="SecReport",
                    filing_date=doc_info.filing_date,
                    filename=filename,
                    extra=asdict(doc_info),
                )
                successful.append(meta)
                if verbose:
                    size_kb = output_path.stat().st_size / 1024
                    print(f"[{idx}/{len(doc_results)}] {symbol}: OK ({size_kb:.0f}KB)")
            else:
                errors[symbol] = error or "Download failed"
                if verbose:
                    print(f"[{idx}/{len(doc_results)}] {symbol}: FAILED - {error}")

    # Save metadata and errors
    if successful:
        save_json(
            {m.symbol: asdict(m) for m in successful},
            output_dir / "metadata.json",
        )
    if errors:
        save_json(errors, output_dir / "errors.json")

    return ScraperResult(successful=successful, errors=errors)
