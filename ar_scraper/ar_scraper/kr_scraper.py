"""Korean scraper using DART for annual reports (사업보고서)."""

from __future__ import annotations

import io
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import requests

from ar_scraper.base import FilingMetadata, ScraperResult
from ar_scraper.config import (
    DART_API_URL,
    DART_HEADERS,
    DART_RATE_LIMIT,
    MAX_RETRIES,
    RETRY_BACKOFF,
)
from ar_scraper.mappers import DartMapper
from ar_scraper.utils import get_existing_symbols, save_json


@dataclass
class ReportInfo:
    """Information about a Korean annual report."""

    symbol: str
    corp_code: str
    corp_name: str
    rcept_no: str
    report_nm: str
    filing_date: str

    @property
    def filename(self) -> str:
        """Generate filename for saving."""
        return f"{self.symbol}_BusinessReport_{self.filing_date}.zip"


class DartClient:
    """Synchronous client for DART API with rate limiting."""

    def __init__(self, api_key: str, rate_limit: float = DART_RATE_LIMIT):
        """Initialize client.

        Args:
            api_key: DART API key (crtfc_key)
            rate_limit: Minimum seconds between requests
        """
        self.session = requests.Session()
        self.session.headers.update(DART_HEADERS)
        self._api_key = api_key
        self._rate_limit_interval = rate_limit
        self._last_request_time: float = 0

    def _rate_limit(self) -> None:
        """Ensure we don't exceed rate limits."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._rate_limit_interval:
            time.sleep(self._rate_limit_interval - elapsed)
        self._last_request_time = time.time()

    def _get(self, url: str, params: Optional[dict] = None, **kwargs) -> requests.Response:
        """Make rate-limited GET request."""
        if params is None:
            params = {}
        params["crtfc_key"] = self._api_key

        for attempt in range(MAX_RETRIES):
            try:
                self._rate_limit()
                response = self.session.get(url, params=params, timeout=60, **kwargs)
                response.raise_for_status()
                return response
            except requests.RequestException:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF[attempt])
                    continue
                raise
        raise RuntimeError("Max retries exceeded")

    def search_annual_report(
        self, corp_code: str, year: int
    ) -> Optional[ReportInfo]:
        """Search DART for annual report (사업보고서) for a given year.

        Args:
            corp_code: 8-digit DART corp code
            year: Filing year

        Returns:
            ReportInfo if found, None otherwise
        """
        url = f"{DART_API_URL}/list.json"
        params = {
            "corp_code": corp_code,
            "bgn_de": f"{year}0101",
            "end_de": f"{year}1231",
            "pblntf_ty": "A",
            "last_reprt_at": "Y",
            "page_count": "100",
        }

        for attempt in range(MAX_RETRIES):
            try:
                response = self._get(url, params=params)
                data = response.json()

                status = data.get("status", "")
                if status == "013":  # No data
                    return None
                if status == "020":  # Rate limit
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_BACKOFF[attempt] * 5)
                        continue
                    return None
                if status != "000":
                    return None

                for item in data.get("list", []):
                    report_nm = item.get("report_nm", "")
                    if "사업보고서" in report_nm and "정정" not in report_nm:
                        stock_code = item.get("stock_code", "").strip()
                        symbol = f"{stock_code}.KS" if stock_code else ""
                        rcept_dt = item.get("rcept_dt", "")
                        filing_date = (
                            f"{rcept_dt[:4]}-{rcept_dt[4:6]}-{rcept_dt[6:8]}"
                            if len(rcept_dt) == 8
                            else rcept_dt
                        )
                        return ReportInfo(
                            symbol=symbol,
                            corp_code=corp_code,
                            corp_name=item.get("corp_name", ""),
                            rcept_no=item.get("rcept_no", ""),
                            report_nm=report_nm,
                            filing_date=filing_date,
                        )
                return None
            except requests.RequestException:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF[attempt])
                    continue
                return None
        return None

    def download_document(
        self, rcept_no: str, output_path: Path
    ) -> tuple[bool, Optional[str]]:
        """Download disclosure document ZIP from DART.

        Args:
            rcept_no: Receipt number from filing search
            output_path: Path to save the ZIP file

        Returns:
            (success, error_message) tuple
        """
        url = f"{DART_API_URL}/document.xml"
        params = {"rcept_no": rcept_no}

        for attempt in range(MAX_RETRIES):
            try:
                response = self._get(url, params=params)

                # DART returns JSON on error, ZIP on success
                content_type = response.headers.get("Content-Type", "")
                if "application/json" in content_type:
                    data = response.json()
                    status = data.get("status", "")
                    if status == "020":  # Rate limit
                        if attempt < MAX_RETRIES - 1:
                            time.sleep(RETRY_BACKOFF[attempt] * 5)
                            continue
                    return False, f"DART error: {data.get('message', status)}"

                content = response.content
                if len(content) < 1000:
                    return False, "Response too small"

                # Verify it's a valid ZIP
                try:
                    with zipfile.ZipFile(io.BytesIO(content)) as zf:
                        if not zf.namelist():
                            return False, "Empty ZIP file"
                except zipfile.BadZipFile:
                    return False, "Not a valid ZIP file"

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


def scrape_kr(
    symbols: list[str],
    output_dir: Path,
    mapper: DartMapper,
    year: int,
    skip_existing: bool = True,
    verbose: bool = True,
) -> ScraperResult:
    """Scrape Korean annual reports from DART.

    Args:
        symbols: List of yfinance symbols (e.g., ["005930.KS"])
        output_dir: Directory to save ZIPs
        mapper: DartMapper instance
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
        print(f"KR: Processing {len(to_process)} symbols ({len(existing)} skipped)")
        print(f"KR: Year {year}")

    if not to_process:
        return ScraperResult(successful=[], errors={})

    client = DartClient(api_key=mapper._api_key)
    successful: list[FilingMetadata] = []
    errors: dict[str, str] = {}

    try:
        for idx, symbol in enumerate(to_process, 1):
            corp_code = mapper.get_corp_code(symbol)
            if not corp_code:
                errors[symbol] = "No corp code mapping"
                if verbose:
                    print(f"[{idx}/{len(to_process)}] {symbol}: SKIP - No mapping")
                continue

            report = client.search_annual_report(corp_code, year)
            if not report:
                errors[symbol] = "Report not found"
                if verbose:
                    print(f"[{idx}/{len(to_process)}] {symbol}: SKIP - Not found")
                continue

            # Use the original symbol (from input) not from DART response
            report.symbol = symbol
            output_path = output_dir / report.filename
            success, error = client.download_document(report.rcept_no, output_path)

            if success:
                meta = FilingMetadata(
                    symbol=symbol,
                    region="kr",
                    company_name=report.corp_name,
                    form_type="BusinessReport",
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
