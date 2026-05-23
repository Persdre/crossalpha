#!/usr/bin/env python3
"""Parse annual reports using Claude Haiku API.

Universal parser that works across markets (US, TW, KR, HK).
Uses the same 10-category extraction as the existing OpenAI-based parsers,
but routes through Anthropic's Claude API instead.

Usage:
    # Parse remaining US FY2023 reports
    python scripts/parse_reports_claude.py \
        --input-dir /path/to/pdfs \
        --output-dir /path/to/output \
        --market us

    # Parse TW reports with page filtering
    python scripts/parse_reports_claude.py \
        --input-dir /path/to/tw_pdfs \
        --output-dir /path/to/output \
        --market tw
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fitz  # pymupdf
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL = "claude-haiku-4-5-20251001"
MAX_INPUT_CHARS = 200000  # ~50K tokens
DEFAULT_WORKERS = 20  # Conservative to avoid OOM from PDF loading
DEFAULT_BATCH_SIZE = 50

# API keys - try multiple locations
def get_api_key() -> str:
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if key:
        return key
    for env_path in [
        Path(__file__).resolve().parent.parent / ".env",
        Path("${CROSSALPHA_PROJECT_ROOT}/.env"),
        Path("${CROSSALPHA_DATA_ROOT}/.env"),
    ]:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"')
    return ""

SYSTEM_PROMPT = """You are analyzing a company's annual report. The document may be in any language (English, Chinese, Japanese, Korean, etc.). You MUST respond entirely in English.

Extract the following 10 categories of business information from the document.
For each category, provide a concise but comprehensive summary (2-5 sentences) in English.

Categories to extract:
1. main_business_segments: Company's primary lines of business or divisions.
2. core_technologies_production_methods: Main technologies, manufacturing or development methods underlying products or services.
3. primary_customers_markets: Key customer types or industries served and major end markets.
4. geographic_coverage: Key operating regions, major countries, or geographic segments.
5. supply_chain_position: Whether the company operates upstream, midstream, downstream, or at another point in the supply chain.
6. strategic_focus_rd_direction: Strategic priorities, innovation themes, and R&D initiatives.
7. revenue_model: How the company generates revenue (e.g., product sales, subscription, licensing, advertising).
8. key_competitors_industry_positioning: Major competitors or describe the company's relative market position.
9. financial_scale_growth_profile: Company size, revenue scale, and growth trajectory.
10. value_proposition_product_differentiation: Unique customer value or competitive differentiation the company emphasizes.

Respond ONLY with a valid JSON object containing these exact keys. No markdown, no explanation. All values must be in English."""

# TW page filtering
TW_RELEVANT = re.compile(
    r"公司簡介|公司概況|營運概況|業務內容|主要產品"
    r"|營業比重|研究發展|經營策略|市場分析|競爭優勢"
    r"|產業概況|公司治理|Corporate Profile|Business Overview"
    r"|Products|Operations|Strategy|R&D"
)

TW_IRRELEVANT = re.compile(
    r"會計師查核報告|合併資產負債表|合併綜合損益表"
    r"|財務報表附註|股東會|Auditor.*Report"
    r"|Consolidated.*Statement|Notes to.*Financial",
    re.IGNORECASE,
)

# KR uses section-based filtering (ZIP files with HTML sections)
KR_RELEVANT_SECTIONS = {
    "사업의 내용", "회사의 개요", "주요 제품", "매출", "연구개발",
    "시장", "경쟁", "사업의개요", "주요제품",
}

KR_IRRELEVANT_SECTIONS = {
    "재무제표", "감사보고서", "이사회", "주주총회", "임원",
    "주식", "배당", "기타", "재무에 관한 사항",
}


def filter_pages_tw(doc: fitz.Document, report_name: str) -> str:
    """Filter TW annual report pages."""
    page_texts = [page.get_text() for page in doc]
    kept = []

    for i, text in enumerate(page_texts):
        if i < 6:
            kept.append(text)
            continue
        if TW_IRRELEVANT.search(text):
            continue
        if TW_RELEVANT.search(text) or len(text) > 500:
            kept.append(text)

    if len(kept) < max(8, len(page_texts) * 0.15):
        return "\n".join(page_texts)
    return "\n".join(kept)


def extract_kr_sections(zip_path: Path) -> str:
    """Extract relevant sections from KR business report ZIP."""
    import zipfile
    try:
        with zipfile.ZipFile(zip_path) as zf:
            texts = []
            for name in zf.namelist():
                if name.endswith(('.html', '.htm', '.xml')):
                    try:
                        content = zf.read(name).decode('utf-8', errors='ignore')
                        # Strip HTML tags
                        clean = re.sub(r'<[^>]+>', ' ', content)
                        clean = re.sub(r'\s+', ' ', clean).strip()
                        if len(clean) > 100:
                            texts.append(clean)
                    except Exception:
                        pass
            return "\n".join(texts)
    except Exception as e:
        return ""


def extract_text(file_path: Path, market: str) -> str:
    """Extract and filter text from a report file."""
    name = file_path.stem

    if market == "kr" and file_path.suffix == ".zip":
        text = extract_kr_sections(file_path)
    else:
        doc = fitz.open(str(file_path))
        if market == "tw":
            text = filter_pages_tw(doc, name)
        else:
            text = "\n".join(page.get_text() for page in doc)
        doc.close()

    # Truncate if too long
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS] + "\n\n[Truncated]"

    return text


def parse_filename(filename: str) -> dict:
    """Parse report filename to extract metadata."""
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) >= 3:
        return {"symbol": parts[0], "report_type": parts[1], "filing_date": parts[2]}
    return {"symbol": parts[0] if parts else stem, "report_type": "AnnualReport", "filing_date": "unknown"}


# ---------------------------------------------------------------------------
# Claude API
# ---------------------------------------------------------------------------

async def call_claude_api(
    text: str,
    company: str,
    semaphore: asyncio.Semaphore,
    api_key: str,
) -> dict:
    """Call Claude Haiku to extract categories."""
    import httpx

    max_retries = 5
    async with semaphore:
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=180) as client:
                    response = await client.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={
                            "x-api-key": api_key,
                            "anthropic-version": "2023-06-01",
                            "content-type": "application/json",
                        },
                        json={
                            "model": MODEL,
                            "max_tokens": 2048,
                            "temperature": 0.1,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": f"{SYSTEM_PROMPT}\n\nAnalyze this annual report for {company}:\n\n{text}",
                                }
                            ],
                        },
                    )

                    if response.status_code == 429:
                        wait = min(2 ** attempt * 5, 60)
                        await asyncio.sleep(wait)
                        continue

                    if response.status_code != 200:
                        error_text = response.text[:200]
                        raise Exception(f"API error {response.status_code}: {error_text}")
                    break
            except httpx.TimeoutException:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt * 3)
                    continue
                raise
        else:
            raise Exception(f"Max retries ({max_retries}) exceeded for {company}")

            data = response.json()
            content = data["content"][0]["text"]

            # Strip markdown code fences if present
            content = content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\n?", "", content)
                content = re.sub(r"\n?```$", "", content)

            # Try to find JSON object in the response
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from the text
                match = re.search(r'\{[\s\S]*\}', content)
                if match:
                    return json.loads(match.group())
                raise Exception(f"Could not parse JSON from response: {content[:200]}")


async def process_single(
    file_path: Path,
    market: str,
    semaphore: asyncio.Semaphore,
    executor: ThreadPoolExecutor,
    api_key: str,
) -> dict:
    """Process a single report file."""
    loop = asyncio.get_event_loop()

    # Extract text in thread pool (CPU-bound PDF parsing)
    text = await loop.run_in_executor(executor, extract_text, file_path, market)

    if not text or len(text) < 100:
        raise Exception(f"No text extracted from {file_path.name}")

    meta = parse_filename(file_path.name)
    company = meta["symbol"]

    categories = await call_claude_api(text, company, semaphore, api_key)

    return {
        "symbol": meta["symbol"],
        "report_type": meta["report_type"],
        "filing_date": meta["filing_date"],
        "source_file": file_path.name,
        "categories": categories,
        "model": MODEL,
    }


async def process_batch(
    file_paths: list[Path],
    market: str,
    max_workers: int,
    api_key: str,
) -> list:
    """Process a batch of reports."""
    semaphore = asyncio.Semaphore(max_workers)

    with ThreadPoolExecutor(max_workers=min(max_workers, 10)) as executor:
        tasks = [
            process_single(fp, market, semaphore, executor, api_key)
            for fp in file_paths
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    return results


def save_results(results: list, output_dir: Path) -> int:
    """Save results to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for result in results:
        if isinstance(result, Exception):
            print(f"  Error: {type(result).__name__}: {result}")
            continue
        filename = result["source_file"].replace(".pdf", ".json").replace(".zip", ".json")
        output_path = output_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        saved += 1
    return saved


def get_existing_parsed(output_dir: Path) -> set[str]:
    """Get set of already parsed symbols."""
    if not output_dir.exists():
        return set()
    return {f.stem.split("_")[0] for f in output_dir.glob("*.json")
            if f.stem != "metadata" and f.stem != "errors"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parse annual reports using Claude Haiku")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--market", type=str, required=True,
                        choices=["us", "tw", "kr", "hk"],
                        help="Market determines page filtering strategy")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-skip", action="store_true")
    args = parser.parse_args()

    api_key = get_api_key()
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not found")
        return

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find files
    if args.market == "kr":
        file_paths = sorted(input_dir.glob("*.zip")) + sorted(input_dir.glob("*.pdf"))
    else:
        file_paths = sorted(input_dir.glob("*.pdf"))

    # Skip already parsed
    if not args.no_skip:
        existing = get_existing_parsed(output_dir)
        before = len(file_paths)
        file_paths = [f for f in file_paths if f.stem.split("_")[0] not in existing]
        skipped = before - len(file_paths)
        if skipped:
            print(f"Skipping {skipped} already parsed reports")

    if args.limit:
        file_paths = file_paths[:args.limit]

    if not file_paths:
        print(f"No files to process in {input_dir}")
        return

    print(f"Found {len(file_paths)} reports to process")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Market: {args.market}, Model: {MODEL}")
    print(f"Workers: {args.workers}, Batch size: {args.batch_size}")
    print()

    total_saved = 0
    total_files = len(file_paths)
    batch_size = args.batch_size

    for batch_start in range(0, total_files, batch_size):
        batch_end = min(batch_start + batch_size, total_files)
        batch = file_paths[batch_start:batch_end]
        batch_num = batch_start // batch_size + 1
        total_batches = (total_files + batch_size - 1) // batch_size

        print(f"{'=' * 50}")
        print(f"BATCH {batch_num}/{total_batches} ({len(batch)} files)")
        print(f"{'=' * 50}")

        results = asyncio.run(
            process_batch(batch, args.market, args.workers, api_key)
        )

        saved = save_results(results, output_dir)
        total_saved += saved
        errors = sum(1 for r in results if isinstance(r, Exception))

        print(f"Batch {batch_num} complete: {saved}/{len(batch)} saved, {errors} errors")
        print()

    print(f"{'=' * 50}")
    print(f"COMPLETE: {total_saved}/{total_files} reports parsed successfully")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
