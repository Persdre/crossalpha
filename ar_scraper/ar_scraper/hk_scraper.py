"""Hong Kong scraper using HKEXnews for annual reports."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import requests

from ar_scraper.base import FilingMetadata, ScraperResult
from ar_scraper.config import (
    HKEX_BASE_URL,
    HKEX_HEADERS,
    HKEX_SEARCH_URL,
    HKEX_T1_FINANCIAL,
    HKEX_T2_ANNUAL_REPORT,
    HK_BATCH_PAUSE,
    HK_BATCH_SIZE,
    HK_RATE_LIMIT,
    MAX_RETRIES,
    MIN_PDF_SIZE,
    RETRY_BACKOFF,
)
from ar_scraper.mappers import HkexMapper
from ar_scraper.utils import get_existing_symbols, save_json


@dataclass
class ReportInfo:
    """Information about a Hong Kong annual report."""

    symbol: str
    stock_code: str
    company_name: str
    title: str
    filing_date: str
    file_link: str
    file_size: str

    @property
    def filename(self) -> str:
        """Generate filename for saving."""
        return f"{self.symbol}_AnnualReport_{self.filing_date}.pdf"


class HkexClient:
    """Synchronous client for HKEXnews with rate limiting."""

    def __init__(self, rate_limit: float = HK_RATE_LIMIT):
        """Initialize client."""
        self.session = requests.Session()
        self.session.headers.update(HKEX_HEADERS)
        self._rate_limit_interval = rate_limit
        self._last_request_time: float = 0
        self._request_count: int = 0

    def _rate_limit(self, verbose: bool = False) -> None:
        """Ensure we don't exceed rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._rate_limit_interval:
            time.sleep(self._rate_limit_interval - elapsed)
        self._last_request_time = time.time()

        self._request_count += 1
        if self._request_count % HK_BATCH_SIZE == 0:
            if verbose:
                print(f"  [Batch pause: {HK_BATCH_PAUSE}s]")
            time.sleep(HK_BATCH_PAUSE)

    def _get(self, url: str, params: Optional[dict] = None, stream: bool = False) -> requests.Response:
        """Make rate-limited GET request."""
        for attempt in range(MAX_RETRIES):
            try:
                self._rate_limit()
                response = self.session.get(url, params=params, timeout=(10, 120), stream=stream)
                response.raise_for_status()
                return response
            except requests.RequestException:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF[attempt])
                    continue
                raise
        raise RuntimeError("Max retries exceeded")

    def search_annual_report(
        self, internal_id: str, fiscal_year: int
    ) -> Optional[ReportInfo]:
        """Search HKEXnews for annual report for a given fiscal year.

        Annual reports for fiscal year N are typically filed in year N+1
        (between Jan and Sep). Search the filing year N+1.
        """
        filing_year = fiscal_year + 1
        params = {
            "sortDir": "0",
            "sortByOptions": "DateTime",
            "category": "0",
            "market": "SEHK",
            "stockId": internal_id,
            "documentType": "-1",
            "fromDate": f"{filing_year}0101",
            "toDate": f"{filing_year}1231",
            "title": "",
            "searchType": "1",
            "t1code": HKEX_T1_FINANCIAL,
            "t2Gcode": "-1",
            "t2code": HKEX_T2_ANNUAL_REPORT,
            "rowRange": "20",
            "lang": "EN",
        }

        for attempt in range(MAX_RETRIES):
            try:
                response = self._get(HKEX_SEARCH_URL, params=params)
                data = response.json()

                result_str = data.get("result", "")
                if not result_str:
                    return None

                results = json.loads(result_str)
                if not results:
                    return None

                item = results[0]
                stock_code = item.get("STOCK_CODE", "").split("<")[0].strip()
                date_time = item.get("DATE_TIME", "")
                date_match = re.match(r"(\d{2})/(\d{2})/(\d{4})", date_time)
                if date_match:
                    filing_date = f"{date_match.group(3)}-{date_match.group(2)}-{date_match.group(1)}"
                else:
                    filing_date = f"{filing_year}-12-31"

                yf_code = stock_code.lstrip("0") or "0"
                symbol = f"{yf_code}.HK"

                return ReportInfo(
                    symbol=symbol,
                    stock_code=stock_code,
                    company_name=item.get("STOCK_NAME", "").split("<")[0].strip(),
                    title=item.get("TITLE", ""),
                    filing_date=filing_date,
                    file_link=item.get("FILE_LINK", ""),
                    file_size=item.get("FILE_INFO", ""),
                )
            except (requests.RequestException, json.JSONDecodeError):
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF[attempt])
                    continue
                return None
        return None

    def download_pdf(
        self, report: ReportInfo, output_path: Path
    ) -> tuple[bool, Optional[str]]:
        """Download PDF for a report."""
        if not report.file_link:
            return False, "No file link"

        pdf_url = f"{HKEX_BASE_URL}{report.file_link}"

        for attempt in range(MAX_RETRIES):
            try:
                response = self._get(pdf_url, stream=True)
                content = response.content

                if not content.startswith(b"%PDF"):
                    return False, "Not a valid PDF"
                if len(content) < MIN_PDF_SIZE:
                    return False, "PDF too small"

                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(content)
                return True, None
            except requests.Timeout:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF[attempt])
                    continue
                return False, "Timeout"
            except requests.RequestException as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF[attempt])
                    continue
                return False, str(e)
        return False, "Max retries exceeded"

    def close(self) -> None:
        """Close the session."""
        self.session.close()


def scrape_hk(
    symbols: list[str],
    output_dir: Path,
    mapper: HkexMapper,
    year: int,
    skip_existing: bool = True,
    verbose: bool = True,
) -> ScraperResult:
    """Scrape Hong Kong annual reports from HKEXnews.

    Args:
        symbols: List of yfinance symbols (e.g., ["0700.HK"])
        output_dir: Directory to save PDFs
        mapper: HkexMapper instance
        year: Fiscal year to fetch reports for
        skip_existing: Skip already downloaded symbols
        verbose: Print progress

    Returns:
        ScraperResult with successful metadata and errors
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = get_existing_symbols(output_dir) if skip_existing else set()
    to_process = [s for s in symbols if s not in existing]

    if verbose:
        print(f"HK: Processing {len(to_process)} symbols ({len(existing)} skipped)")
        print(f"HK: Year {year}")

    if not to_process:
        return ScraperResult(successful=[], errors={})

    client = HkexClient()
    successful: list[FilingMetadata] = []
    errors: dict[str, str] = {}

    try:
        for idx, symbol in enumerate(to_process, 1):
            if verbose:
                print(f"[{idx}/{len(to_process)}] {symbol}: Searching...", end="", flush=True)

            internal_id = mapper.get_internal_id(symbol)
            if not internal_id:
                errors[symbol] = "No HKEXnews mapping"
                if verbose:
                    print(f" SKIP - No mapping")
                continue

            report = client.search_annual_report(internal_id, year)
            if not report:
                report = client.search_annual_report(internal_id, year - 1)
                if report and verbose:
                    print(
                        f"[{idx}/{len(to_process)}] {symbol}: "
                        f"Using FY{year - 1} (FY{year} not found)"
                    )
            if not report:
                errors[symbol] = "Report not found"
                if verbose:
                    print(f" SKIP - Not found")
                continue

            report.symbol = symbol
            output_path = output_dir / report.filename

            if verbose:
                print(f" Found, downloading...", end="", flush=True)

            success, error = client.download_pdf(report, output_path)

            if success:
                meta = FilingMetadata(
                    symbol=symbol,
                    region="hk",
                    company_name=report.company_name,
                    form_type="AnnualReport",
                    filing_date=report.filing_date,
                    filename=report.filename,
                    extra=asdict(report),
                )
                successful.append(meta)
                if verbose:
                    size_kb = output_path.stat().st_size / 1024
                    print(f" OK ({size_kb:.0f}KB)")
            else:
                errors[symbol] = error or "Download failed"
                if verbose:
                    print(f" FAILED - {error}")
    finally:
        client.close()

    if successful:
        save_json(
            {m.symbol: asdict(m) for m in successful},
            output_dir / "metadata.json",
        )
    if errors:
        save_json(errors, output_dir / "errors.json")

    return ScraperResult(successful=successful, errors=errors)
