"""Taiwan scraper using MOPS for annual reports."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import requests
import urllib3
from bs4 import BeautifulSoup

from ar_scraper.base import FilingMetadata, ScraperResult
from ar_scraper.config import (
    MAX_RETRIES,
    MIN_PDF_SIZE,
    MOPS_HEADERS,
    MOPS_RATE_LIMIT_MSG,
    MOPS_URL,
    RETRY_BACKOFF,
    TW_BATCH_PAUSE,
    TW_BATCH_SIZE,
    TW_RATE_LIMIT,
)
from ar_scraper.mappers import TwseMapper
from ar_scraper.utils import get_existing_symbols, save_json

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


@dataclass
class ReportInfo:
    """Information about a Taiwan annual report."""

    symbol: str
    stock_id: str
    company_name: str
    year: int
    roc_year: int
    filing_date: str
    pdf_filename: str

    @property
    def filename(self) -> str:
        """Generate filename for saving."""
        return f"{self.symbol}_AnnualReport_{self.filing_date}.pdf"


class MopsClient:
    """Synchronous client for MOPS with rate limiting."""

    def __init__(self, rate_limit: float = TW_RATE_LIMIT):
        """Initialize client."""
        self.session = requests.Session()
        self.session.headers.update(MOPS_HEADERS)
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
        if self._request_count % TW_BATCH_SIZE == 0:
            if verbose:
                print(f"  [Batch pause: {TW_BATCH_PAUSE}s]")
            time.sleep(TW_BATCH_PAUSE)

    def _is_rate_limited(self, response: requests.Response) -> bool:
        """Check if response indicates rate limiting."""
        if response.status_code == 200:
            response.encoding = "big5"
            if len(response.text) < 1000:
                return MOPS_RATE_LIMIT_MSG in response.text
        return False

    def _handle_rate_limit(self, attempt: int, verbose: bool = False) -> None:
        """Handle rate limit by waiting."""
        wait_time = 15 * (attempt + 1)
        if verbose:
            print(f"  [Rate limited, waiting {wait_time}s]")
        time.sleep(wait_time)

    def _get(self, url: str, stream: bool = False) -> requests.Response:
        """Make rate-limited GET request."""
        for attempt in range(3):
            try:
                self._rate_limit()
                response = self.session.get(
                    url, timeout=60, stream=stream, verify=False
                )
                response.raise_for_status()
                if not stream and self._is_rate_limited(response):
                    if attempt < 2:
                        self._handle_rate_limit(attempt)
                        continue
                return response
            except requests.RequestException:
                if attempt < 2:
                    self._handle_rate_limit(attempt)
                    continue
                raise
        raise RuntimeError("Max retries exceeded")

    def _post(self, url: str, data: dict) -> requests.Response:
        """Make rate-limited POST request."""
        for attempt in range(3):
            try:
                self._rate_limit()
                response = self.session.post(
                    url, data=data, timeout=60, verify=False
                )
                response.raise_for_status()
                if self._is_rate_limited(response):
                    if attempt < 2:
                        self._handle_rate_limit(attempt)
                        continue
                return response
            except requests.RequestException:
                if attempt < 2:
                    self._handle_rate_limit(attempt)
                    continue
                raise
        raise RuntimeError("Max retries exceeded")

    def get_annual_report(self, stock_id: str, year: int) -> Optional[ReportInfo]:
        """Query MOPS for annual report info."""
        roc_year = year - 1911

        for attempt in range(MAX_RETRIES):
            try:
                listing_url = (
                    f"{MOPS_URL}?step=1&colorchg=1"
                    f"&co_id={stock_id}&year={roc_year}&mtype=F&"
                )
                response = self._get(listing_url)
                response.encoding = "big5"
                return self._parse_listing(response.text, stock_id, year, roc_year)
            except requests.Timeout:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF[attempt])
                    continue
                return None
            except requests.RequestException:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF[attempt])
                    continue
                return None
        return None

    def _parse_listing(
        self, html: str, stock_id: str, year: int, roc_year: int
    ) -> Optional[ReportInfo]:
        """Parse file listing to find annual report."""
        if MOPS_RATE_LIMIT_MSG in html:
            return None

        soup = BeautifulSoup(html, "html.parser")

        company_name = ""
        name_match = re.search(r"公司名稱[：:]\s*(.+?)(?:<|\n|$)", html)
        if name_match:
            company_name = name_match.group(1).strip()

        rows = soup.find_all("tr")
        pdf_filename = None
        filing_date = None
        report_year = roc_year - 1

        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 8:
                continue

            row_text = row.get_text()
            if "股東會年報" not in row_text:
                continue

            year_cell = cells[1].get_text(strip=True) if len(cells) > 1 else ""
            if f"{report_year}" not in year_cell and f"{roc_year}" not in year_cell:
                continue

            link = row.find("a", href=True)
            if link:
                href = link.get("href", "")
                match = re.search(r'readfile2\([^,]+,[^,]+,"([^"]+\.pdf)"', href)
                if match:
                    pdf_filename = match.group(1)
                    date_cell = cells[-1].get_text(strip=True) if cells else ""
                    date_match = re.search(r"(\d{3})/(\d{2})/(\d{2})", date_cell)
                    if date_match:
                        roc_y = int(date_match.group(1))
                        m = date_match.group(2)
                        d = date_match.group(3)
                        western_y = roc_y + 1911
                        filing_date = f"{western_y}-{m}-{d}"
                    break

        if not pdf_filename:
            return None
        if not filing_date:
            filing_date = f"{year + 1}-06-30"

        return ReportInfo(
            symbol=f"{stock_id}.TW",
            stock_id=stock_id,
            company_name=company_name,
            year=year,
            roc_year=roc_year,
            filing_date=filing_date,
            pdf_filename=pdf_filename,
        )

    def download_pdf(
        self, report: ReportInfo, output_path: Path
    ) -> tuple[bool, Optional[str]]:
        """Download PDF for a report."""
        for attempt in range(MAX_RETRIES):
            try:
                data = {
                    "colorchg": "1",
                    "step": "9",
                    "kind": "F",
                    "co_id": report.stock_id,
                    "filename": report.pdf_filename,
                }
                response = self._post(MOPS_URL, data)
                response.encoding = "big5"

                pdf_match = re.search(r"href='(/pdf/[^']+)'", response.text)
                if not pdf_match:
                    return False, "Could not find PDF URL"

                pdf_path = pdf_match.group(1)
                pdf_url = f"https://doc.twse.com.tw{pdf_path}"

                pdf_response = self._get(pdf_url, stream=True)
                content = pdf_response.content

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


def scrape_tw(
    symbols: list[str],
    output_dir: Path,
    mapper: TwseMapper,
    year: int,
    skip_existing: bool = True,
    verbose: bool = True,
) -> ScraperResult:
    """Scrape Taiwan annual reports from MOPS.

    Args:
        symbols: List of yfinance symbols (e.g., ["2330.TW"])
        output_dir: Directory to save PDFs
        mapper: TwseMapper instance
        year: Year to fetch reports for (filing year)
        skip_existing: Skip already downloaded symbols
        verbose: Print progress

    Returns:
        ScraperResult with successful metadata and errors
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = get_existing_symbols(output_dir) if skip_existing else set()
    to_process = [s for s in symbols if s not in existing]

    if verbose:
        print(f"TW: Processing {len(to_process)} symbols ({len(existing)} skipped)")
        print(f"TW: Year {year} (ROC {year - 1911})")

    if not to_process:
        return ScraperResult(successful=[], errors={})

    client = MopsClient()
    successful: list[FilingMetadata] = []
    errors: dict[str, str] = {}

    try:
        for idx, symbol in enumerate(to_process, 1):
            stock_id = mapper.get_stock_id(symbol)
            if not stock_id:
                errors[symbol] = "No stock ID mapping"
                if verbose:
                    print(f"[{idx}/{len(to_process)}] {symbol}: SKIP - No mapping")
                continue

            report = client.get_annual_report(stock_id, year)
            if not report:
                errors[symbol] = "Report not found"
                if verbose:
                    print(f"[{idx}/{len(to_process)}] {symbol}: SKIP - Not found")
                continue

            output_path = output_dir / report.filename
            success, error = client.download_pdf(report, output_path)

            if success:
                meta = FilingMetadata(
                    symbol=symbol,
                    region="tw",
                    company_name=report.company_name,
                    form_type="AnnualReport",
                    filing_date=report.filing_date,
                    filename=report.filename,
                    extra=asdict(report),
                )
                successful.append(meta)
                if verbose:
                    size_kb = output_path.stat().st_size / 1024
                    print(f"[{idx}/{len(to_process)}] {symbol}: OK ({size_kb:.0f}KB)")
            else:
                errors[symbol] = error or "Download failed"
                if verbose:
                    print(f"[{idx}/{len(to_process)}] {symbol}: FAILED - {error}")
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
