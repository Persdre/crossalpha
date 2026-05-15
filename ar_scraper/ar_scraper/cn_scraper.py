"""China A-share scraper using CNINFO (巨潮资讯网) for annual reports (年度报告)."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

from ar_scraper.base import FilingMetadata, ScraperResult
from ar_scraper.config import (
    CNINFO_HEADERS,
    CNINFO_SEARCH_URL,
    CNINFO_STATIC_URL,
    CN_BATCH_PAUSE,
    CN_BATCH_SIZE,
    CN_RATE_LIMIT,
    MAX_RETRIES,
    MIN_PDF_SIZE,
    RETRY_BACKOFF,
)
from ar_scraper.utils import get_existing_symbols, save_json


@dataclass
class ReportInfo:
    """Information about a China A-share annual report."""

    symbol: str
    stock_code: str
    company_name: str
    title: str
    filing_date: str
    adjunct_url: str
    announcement_id: str

    @property
    def filename(self) -> str:
        """Generate filename for saving."""
        return f"{self.symbol}_AnnualReport_{self.filing_date}.pdf"


class CninfoClient:
    """Synchronous client for CNINFO with rate limiting."""

    def __init__(self, rate_limit: float = CN_RATE_LIMIT):
        """Initialize client."""
        self.session = requests.Session()
        self.session.headers.update(CNINFO_HEADERS)
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
        if self._request_count % CN_BATCH_SIZE == 0:
            if verbose:
                print(f"  [Batch pause: {CN_BATCH_PAUSE}s]")
            time.sleep(CN_BATCH_PAUSE)

    def _post(self, url: str, data: dict) -> requests.Response:
        """Make rate-limited POST request."""
        for attempt in range(MAX_RETRIES):
            try:
                self._rate_limit()
                response = self.session.post(url, data=data, timeout=60)
                response.raise_for_status()
                return response
            except requests.RequestException:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF[attempt])
                    continue
                raise
        raise RuntimeError("Max retries exceeded")

    def _get(self, url: str, stream: bool = False) -> requests.Response:
        """Make rate-limited GET request."""
        for attempt in range(MAX_RETRIES):
            try:
                self._rate_limit()
                response = self.session.get(url, timeout=(10, 120), stream=stream)
                response.raise_for_status()
                return response
            except requests.RequestException:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BACKOFF[attempt])
                    continue
                raise
        raise RuntimeError("Max retries exceeded")

    @staticmethod
    def _symbol_to_code(symbol: str) -> str:
        """Convert yfinance symbol to CNINFO stock code.

        600519.SS -> 600519, 000001.SZ -> 000001
        """
        return symbol.split(".")[0]

    @staticmethod
    def _symbol_to_org_id(symbol: str) -> tuple[str, str]:
        """Return (orgId, column) for CNINFO hisAnnouncement API.

        CNINFO requires stock=code,orgId and column=szse|sse.
        OrgId pattern: gssz0<code> for SZ, gssh0<code> for SS.
        """
        parts = symbol.upper().split(".")
        code = parts[0]
        suffix = parts[1] if len(parts) > 1 else ""
        if suffix == "SZ":
            return f"gssz0{code}", "szse"
        elif suffix == "SS":
            return f"gssh0{code}", "sse"
        else:
            # Default guess: Shenzhen
            return f"gssz0{code}", "szse"

    @staticmethod
    def _strip_html(text: str) -> str:
        """Strip HTML tags (e.g. <em>) from CNINFO response text."""
        return re.sub(r"<[^>]+>", "", text)

    def _match_annual_report(
        self, announcements: list[dict], stock_code: str, fiscal_year: int
    ) -> Optional[ReportInfo]:
        """Find the best matching annual report from a list of announcements."""
        fy_str = str(fiscal_year)

        for ann in announcements:
            # Verify this announcement is for our stock
            sec_code = ann.get("secCode", "")
            if sec_code != stock_code:
                continue

            title = self._strip_html(ann.get("announcementTitle", ""))

            # Must contain "年度报告" (annual report) and the fiscal year
            if "年度报告" not in title:
                continue
            if fy_str not in title:
                continue

            # Skip: summaries, supplements, corrections, English,
            # semi-annual (半年度), cancelled
            skip_keywords = [
                "摘要", "补充", "英文", "更正", "修订稿",
                "已取消", "半年度", "英文版",
            ]
            if any(kw in title for kw in skip_keywords):
                continue

            adjunct_url = ann.get("adjunctUrl", "")
            if not adjunct_url:
                continue

            # Only accept PDFs
            if not adjunct_url.upper().endswith(".PDF"):
                continue

            # Parse filing date
            ann_time = ann.get("announcementTime", 0)
            if ann_time:
                dt = datetime.fromtimestamp(
                    ann_time / 1000, tz=timezone.utc
                )
                filing_date = dt.strftime("%Y-%m-%d")
            else:
                filing_date = f"{fiscal_year + 1}-04-30"

            return ReportInfo(
                symbol="",  # filled by caller
                stock_code=stock_code,
                company_name=self._strip_html(ann.get("secName", "")),
                title=title,
                filing_date=filing_date,
                adjunct_url=adjunct_url,
                announcement_id=str(ann.get("announcementId", "")),
            )

        return None

    def search_annual_report(
        self, symbol: str, year: int
    ) -> Optional[ReportInfo]:
        """Search CNINFO for annual report (年度报告) for a given fiscal year.

        Uses the hisAnnouncement API with stock+orgId+column parameters.
        Annual reports for fiscal year N are published in year N+1.

        Args:
            symbol: yfinance symbol (e.g., "600519.SS")
            year: Fiscal year for the annual report

        Returns:
            ReportInfo if found, None otherwise
        """
        stock_code = self._symbol_to_code(symbol)
        org_id, column = self._symbol_to_org_id(symbol)

        # Strategy 1: search with annual report category and date range
        report = self._search_with_category(stock_code, org_id, column, year)
        if report:
            return report

        # Strategy 2: search without category filter (edge-case stocks)
        report = self._search_without_category(stock_code, org_id, column, year)
        if report:
            return report

        # Strategy 3: fulltextSearch API (slower fallback)
        return self._search_fulltext(stock_code, year)

    def _search_with_category(
        self, stock_code: str, org_id: str, column: str, fiscal_year: int
    ) -> Optional[ReportInfo]:
        """Search with stock+orgId and category_ndbg_szsh filter."""
        # Annual reports for FY N are published between Jan and May of N+1
        se_date = f"{fiscal_year + 1}-01-01~{fiscal_year + 1}-12-31"
        data = {
            "stock": f"{stock_code},{org_id}",
            "tabName": "fulltext",
            "pageSize": "30",
            "pageNum": "1",
            "column": column,
            "category": "category_ndbg_szsh",
            "seDate": se_date,
            "isHLtitle": "true",
        }

        try:
            response = self._post(CNINFO_SEARCH_URL, data=data)
            result = response.json()
            announcements = result.get("announcements") or []
            if not announcements:
                return None
            return self._match_annual_report(announcements, stock_code, fiscal_year)
        except (requests.RequestException, ValueError):
            return None

    def _search_without_category(
        self, stock_code: str, org_id: str, column: str, fiscal_year: int
    ) -> Optional[ReportInfo]:
        """Fallback: search without category filter for edge-case stocks."""
        se_date = f"{fiscal_year + 1}-01-01~{fiscal_year + 1}-12-31"
        data = {
            "stock": f"{stock_code},{org_id}",
            "tabName": "fulltext",
            "pageSize": "100",
            "pageNum": "1",
            "column": column,
            "seDate": se_date,
            "isHLtitle": "true",
        }

        try:
            response = self._post(CNINFO_SEARCH_URL, data=data)
            result = response.json()
            announcements = result.get("announcements") or []
            if not announcements:
                return None
            return self._match_annual_report(announcements, stock_code, fiscal_year)
        except (requests.RequestException, ValueError):
            return None

    def _search_fulltext(
        self, stock_code: str, fiscal_year: int
    ) -> Optional[ReportInfo]:
        """Last resort: use fulltextSearch API (may be rate-limited)."""
        from ar_scraper.config import CNINFO_FULLTEXT_URL

        data = {
            "searchkey": f"{stock_code} {fiscal_year} 年度报告",
            "sdate": "",
            "edate": "",
            "isfulltext": "false",
            "sortName": "pubdate",
            "sortType": "desc",
            "pageNum": "1",
            "pageSize": "30",
        }

        try:
            response = self._post(CNINFO_FULLTEXT_URL, data=data)
            result = response.json()
            announcements = result.get("announcements") or []
            if not announcements:
                return None
            return self._match_annual_report(announcements, stock_code, fiscal_year)
        except (requests.RequestException, ValueError):
            return None

    def download_pdf(
        self, report: ReportInfo, output_path: Path
    ) -> tuple[bool, Optional[str]]:
        """Download PDF for a report."""
        if not report.adjunct_url:
            return False, "No adjunct URL"

        pdf_url = f"{CNINFO_STATIC_URL}/{report.adjunct_url}"

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


def scrape_cn(
    symbols: list[str],
    output_dir: Path,
    year: int,
    skip_existing: bool = True,
    verbose: bool = True,
) -> ScraperResult:
    """Scrape China A-share annual reports from CNINFO.

    Args:
        symbols: List of yfinance symbols (e.g., ["600519.SS", "000001.SZ"])
        output_dir: Directory to save PDFs
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
        print(f"CN: Processing {len(to_process)} symbols ({len(existing)} skipped)")
        print(f"CN: Fiscal year {year}")

    if not to_process:
        return ScraperResult(successful=[], errors={})

    client = CninfoClient()
    successful: list[FilingMetadata] = []
    errors: dict[str, str] = {}

    try:
        for idx, symbol in enumerate(to_process, 1):
            if verbose:
                print(f"[{idx}/{len(to_process)}] {symbol}: Searching...", end="", flush=True)

            report = client.search_annual_report(symbol, year)
            if not report:
                # If target year not found, try year-1 as fallback
                if verbose:
                    print(f" trying FY{year-1}...", end="", flush=True)
                report = client.search_annual_report(symbol, year - 1)
                if report and verbose:
                    print(f" found FY{year-1},", end="", flush=True)

            if not report:
                errors[symbol] = "Report not found"
                if verbose:
                    print(f" SKIP - Not found")
                continue

            report.symbol = symbol
            output_path = output_dir / report.filename

            if verbose:
                print(f" downloading...", end="", flush=True)

            success, error = client.download_pdf(report, output_path)

            if success:
                meta = FilingMetadata(
                    symbol=symbol,
                    region="cn",
                    company_name=report.company_name,
                    form_type="AnnualReport",
                    filing_date=report.filing_date,
                    filename=report.filename,
                    extra=asdict(report),
                )
                successful.append(meta)
                if verbose:
                    size_kb = output_path.stat().st_size / 1024
                    print(f" OK ({size_kb:.0f}KB) - {report.title}")
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
