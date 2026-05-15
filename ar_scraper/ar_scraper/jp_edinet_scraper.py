"""Japan scraper using EDINET API v2 for securities reports.

Requires a free EDINET API key from:
https://disclosure.edinet-fsa.go.jp/EKW0EZ0015.html

Set the key as environment variable EDINET_API_KEY or pass it directly.

Usage:
    python jp_edinet_scraper.py --year 2024 --output-dir ./reports/securities_reports_fy2023
    python jp_edinet_scraper.py --year 2023 --output-dir ./reports/securities_reports_fy2022
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date, timedelta
from pathlib import Path
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EDINET_API_BASE = "https://disclosure.edinet-fsa.go.jp/api/v2"
EDINET_PDF_BASE = "https://disclosure2dl.edinet-fsa.go.jp/searchdocument/pdf"

# 有価証券報告書 = Securities Report (Annual)
DOC_TYPE_CODE = "120"

# Rate limit: ~5 requests/second for document list, be conservative
RATE_LIMIT = 0.3  # seconds between requests

# ---------------------------------------------------------------------------
# EDINET API functions
# ---------------------------------------------------------------------------


def get_documents_for_date(
    api_key: str, date_str: str, include_details: bool = True
) -> list[dict]:
    """Get all filings for a specific date from EDINET API.

    Args:
        api_key: EDINET API subscription key
        date_str: Date in YYYY-MM-DD format
        include_details: Whether to include full metadata

    Returns:
        List of document dicts from EDINET API
    """
    url = f"{EDINET_API_BASE}/documents.json"
    params = {
        "date": date_str,
        "type": 2 if include_details else 1,
        "Subscription-Key": api_key,
    }

    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code == 401:
        raise ValueError("Invalid EDINET API key. Register at: "
                         "https://disclosure.edinet-fsa.go.jp/EKW0EZ0015.html")
    resp.raise_for_status()

    data = resp.json()
    metadata = data.get("metadata", {})
    if metadata.get("status") != "200":
        return []

    return data.get("results", [])


def find_securities_reports(
    api_key: str,
    edinet_codes: set[str],
    start_date: str,
    end_date: str,
    verbose: bool = True,
) -> dict[str, dict]:
    """Search EDINET filings date-by-date to find securities reports.

    Args:
        api_key: EDINET API key
        edinet_codes: Set of EDINET codes to look for
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
        verbose: Print progress

    Returns:
        Dict mapping edinet_code -> {docID, filerName, filingDate, docDescription, ...}
    """
    found: dict[str, dict] = {}
    remaining = set(edinet_codes)

    current = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    total_days = (end - current).days + 1
    day_count = 0

    while current <= end and remaining:
        date_str = current.isoformat()
        day_count += 1

        if verbose and day_count % 10 == 0:
            print(f"  Scanning {date_str} ({day_count}/{total_days} days, "
                  f"found {len(found)}/{len(edinet_codes)}, "
                  f"{len(remaining)} remaining)")

        try:
            docs = get_documents_for_date(api_key, date_str)

            for doc in docs:
                ec = doc.get("edinetCode", "")
                doc_type = doc.get("docTypeCode", "")
                doc_id = doc.get("docID", "")

                # Match: securities report for a company we're looking for
                if ec in remaining and doc_type == DOC_TYPE_CODE and doc_id:
                    found[ec] = {
                        "docID": doc_id,
                        "edinetCode": ec,
                        "filerName": doc.get("filerName", ""),
                        "filingDate": date_str,
                        "docDescription": doc.get("docDescription", ""),
                        "periodStart": doc.get("periodStart", ""),
                        "periodEnd": doc.get("periodEnd", ""),
                    }
                    remaining.discard(ec)
                    if verbose:
                        print(f"    Found: {ec} {doc.get('filerName', '')[:20]} "
                              f"docID={doc_id}")

        except Exception as e:
            if verbose:
                print(f"  Error on {date_str}: {e}")

        time.sleep(RATE_LIMIT)
        current += timedelta(days=1)

    if verbose:
        print(f"\nSearch complete: found {len(found)}/{len(edinet_codes)} reports")
        if remaining:
            print(f"  Missing {len(remaining)} companies")

    return found


def download_pdf(doc_id: str, output_path: Path, api_key: str) -> bool:
    """Download PDF for a document from EDINET.

    First tries the public CDN (no key needed), falls back to API.

    Args:
        doc_id: EDINET document ID (e.g., S100W543)
        output_path: Where to save the PDF
        api_key: EDINET API key (for fallback)

    Returns:
        True if successful
    """
    # Method 1: Public CDN (no API key needed, faster)
    url = f"{EDINET_PDF_BASE}/{doc_id}.pdf"
    try:
        resp = requests.get(url, timeout=60)
        if resp.status_code == 200 and len(resp.content) > 10000:
            if resp.headers.get("Content-Type", "").startswith("application/pdf"):
                output_path.write_bytes(resp.content)
                return True
    except Exception:
        pass

    # Method 2: EDINET API (needs key)
    url = f"{EDINET_API_BASE}/documents/{doc_id}"
    params = {"type": 2, "Subscription-Key": api_key}
    try:
        resp = requests.get(url, params=params, timeout=60)
        if resp.status_code == 200 and len(resp.content) > 10000:
            output_path.write_bytes(resp.content)
            return True
    except Exception:
        pass

    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Download JP securities reports from EDINET")
    parser.add_argument("--year", type=int, required=True,
                        help="Filing year (e.g., 2024 for FY2023 reports)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for PDFs")
    parser.add_argument("--api-key", type=str, default=None,
                        help="EDINET API key (or set EDINET_API_KEY env var)")
    parser.add_argument("--universe", type=str, default="jp_topix_500",
                        help="Universe to use for symbol list")
    parser.add_argument("--force", action="store_true",
                        help="Re-download existing files")
    args = parser.parse_args()

    # Get API key
    import os
    api_key = args.api_key or os.environ.get("EDINET_API_KEY", "")
    if not api_key:
        # Check .env files
        for env_path in [Path(".env"), Path.home() / ".env"]:
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    if line.startswith("EDINET_API_KEY="):
                        api_key = line.split("=", 1)[1].strip()
                        break
            if api_key:
                break

    if not api_key:
        print("ERROR: EDINET API key not found.")
        print("Register for free at: https://disclosure.edinet-fsa.go.jp/EKW0EZ0015.html")
        print("Then set: export EDINET_API_KEY=your_key")
        return

    # Load EDINET code mapping
    from ar_scraper.mappers import EdinetMapper
    from ar_scraper.utils import get_data_path, load_symbols_from_csv

    data_path = get_data_path("SymbolDict.csv")
    symbols = load_symbols_from_csv(data_path, args.universe)
    mapper = EdinetMapper(get_data_path("EdinetcodeDlInfo.csv"))

    # Build edinet_code -> symbol mapping
    edinet_to_symbol: dict[str, str] = {}
    for sym in symbols:
        ec = mapper.get_edinet_code(sym)
        if ec:
            edinet_to_symbol[ec] = sym

    print(f"Universe: {args.universe} ({len(symbols)} symbols, "
          f"{len(edinet_to_symbol)} with EDINET codes)")

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check existing
    if not args.force:
        existing = {p.stem.split("_")[0] for p in output_dir.glob("*.pdf")}
    else:
        existing = set()

    # Filter out already downloaded
    to_find = {ec for ec, sym in edinet_to_symbol.items() if sym not in existing}
    print(f"Need to find: {len(to_find)} reports (year={args.year})")

    if not to_find:
        print("All reports already downloaded!")
        return

    # Search EDINET for securities reports filed in the target year
    # Japanese companies typically file 有価証券報告書 within 3 months of fiscal year end
    # Most fiscal years end March 31, so reports are filed Apr-Sep
    start_date = f"{args.year}-01-01"
    end_date = f"{args.year}-12-31"

    print(f"\nSearching EDINET for 有価証券報告書 filed in {args.year}...")
    found = find_securities_reports(api_key, to_find, start_date, end_date, verbose=True)

    # Download PDFs
    print(f"\nDownloading {len(found)} PDFs...")
    success = 0
    errors = 0

    for ec, info in found.items():
        sym = edinet_to_symbol[ec]
        doc_id = info["docID"]
        filing_date = info["filingDate"]
        filename = f"{sym}_SecReport_{filing_date}.pdf"
        output_path = output_dir / filename

        if output_path.exists() and not args.force:
            success += 1
            continue

        ok = download_pdf(doc_id, output_path, api_key)
        if ok:
            success += 1
            size_kb = output_path.stat().st_size / 1024
            print(f"  [{success + errors}/{len(found)}] {sym}: OK ({size_kb:.0f}KB)")
        else:
            errors += 1
            print(f"  [{success + errors}/{len(found)}] {sym}: FAILED")

        time.sleep(RATE_LIMIT)

    # Save metadata
    metadata = {}
    for ec, info in found.items():
        sym = edinet_to_symbol[ec]
        metadata[sym] = info
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Summary
    print(f"\n{'=' * 50}")
    print(f"COMPLETE: {success} OK, {errors} errors")
    print(f"Found {len(found)}/{len(to_find)} reports in EDINET")
    print(f"Output: {output_dir}")

    if to_find - set(found.keys()):
        missing_syms = [edinet_to_symbol[ec] for ec in (to_find - set(found.keys()))]
        print(f"Missing {len(missing_syms)} companies: {missing_syms[:10]}...")


if __name__ == "__main__":
    main()
