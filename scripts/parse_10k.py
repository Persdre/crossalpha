#!/usr/bin/env python3
"""Parse 10-K/20-F filings using OpenAI GPT-4.1 to extract business categories.

Processes PDF files from reports/10_k_filings/ and saves parsed JSON
to parsed_reports/10_k_filings/.

Usage:
    python parse_10k.py                           # Process all filings
    python parse_10k.py --limit 10                # Process first 10 filings
    python parse_10k.py --symbol AAPL             # Process single company
    python parse_10k.py --input-dir custom/path   # Custom input directory
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fitz  # pymupdf
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()

# Configuration
PROJECT_ROOT = Path(__file__).parent
INPUT_DIR = PROJECT_ROOT / "reports" / "10_k_filings"
OUTPUT_DIR = PROJECT_ROOT / "parsed_reports" / "10_k_filings"
DEFAULT_WORKERS = 500
DEFAULT_BATCH_SIZE = 500
MODEL = "gpt-4.1"

# 10 Evaluation Categories (from Morgan Stanley paper Exhibit 4)
CATEGORIES = {
    "main_business_segments": (
        "Company's primary lines of business or divisions."
    ),
    "core_technologies_production_methods": (
        "Main technologies, manufacturing or development methods "
        "underlying products or services."
    ),
    "primary_customers_markets": (
        "Key customer types or industries served and major end markets."
    ),
    "geographic_coverage": (
        "Key operating regions, major countries, or geographic segments."
    ),
    "supply_chain_position": (
        "Whether the company operates upstream, midstream, downstream, "
        "or at another point in the supply chain."
    ),
    "strategic_focus_rd_direction": (
        "Strategic priorities, innovation themes, and R&D initiatives."
    ),
    "revenue_model": (
        "How the company generates revenue (e.g., product sales, "
        "subscription, licensing, advertising)."
    ),
    "key_competitors_industry_positioning": (
        "Major competitors or describe the company's relative market position."
    ),
    "financial_scale_growth_profile": (
        "Company size, revenue scale, and growth trajectory."
    ),
    "value_proposition_product_differentiation": (
        "Unique customer value or competitive differentiation "
        "the company emphasizes."
    ),
}

SYSTEM_PROMPT = """You are analyzing a Form 10-K or 20-F annual report filed with the SEC.
Extract the following 10 categories of business information from the document.
For each category, provide a concise but comprehensive summary (2-5 sentences).

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

Respond ONLY with a valid JSON object containing these exact keys. No markdown, no explanation."""


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF file using pymupdf."""
    doc = fitz.open(str(pdf_path))
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n".join(text_parts)


def parse_filename(filename: str) -> dict:
    """Parse filing filename to extract metadata.

    Expected format: SYMBOL_FORMTYPE_DATE.pdf
    Example: AAPL_10-K_2024-11-01.pdf
    """
    stem = Path(filename).stem
    parts = stem.split("_")

    if len(parts) >= 3:
        return {
            "symbol": parts[0],
            "form_type": parts[1],
            "filing_date": parts[2],
        }
    elif len(parts) >= 1:
        return {
            "symbol": parts[0],
            "form_type": "10-K",
            "filing_date": "unknown",
        }
    return {"symbol": stem, "form_type": "10-K", "filing_date": "unknown"}


async def call_openai_api(
    client: AsyncOpenAI,
    text: str,
    company: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Call OpenAI API to extract categories from filing text."""
    async with semaphore:
        print(f"  Processing {company}...")

        # Truncate text if too long (~125K tokens max)
        max_chars = 500000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[Text truncated due to length]"

        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Analyze this SEC filing for {company}:\n\n{text}",
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        content = response.choices[0].message.content
        return json.loads(content)


async def process_single_filing(
    client: AsyncOpenAI,
    file_path: Path,
    semaphore: asyncio.Semaphore,
    executor: ThreadPoolExecutor,
) -> dict:
    """Process a single filing file."""
    loop = asyncio.get_event_loop()

    # Extract text from PDF in thread pool (CPU-bound)
    text = await loop.run_in_executor(executor, extract_text_from_pdf, file_path)

    # Parse filename for metadata
    meta = parse_filename(file_path.name)
    company = meta["symbol"]

    # Call OpenAI API
    categories = await call_openai_api(client, text, company, semaphore)

    return {
        "symbol": meta["symbol"],
        "form_type": meta["form_type"],
        "filing_date": meta["filing_date"],
        "source_file": file_path.name,
        "categories": categories,
    }


async def process_all_filings(
    file_paths: list[Path], max_workers: int = DEFAULT_WORKERS
) -> list[dict]:
    """Process all filing files in parallel."""
    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_URL", "https://api.openai.com/v1"),
    )
    semaphore = asyncio.Semaphore(max_workers)

    with ThreadPoolExecutor(max_workers=min(max_workers, 50)) as executor:
        tasks = [
            process_single_filing(client, file_path, semaphore, executor)
            for file_path in file_paths
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    return results


def save_results(results: list[dict], output_dir: Path) -> int:
    """Save results to JSON files. Returns count of successful saves."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for result in results:
        if isinstance(result, Exception):
            print(f"  Error: {result}")
            continue

        # Use same naming convention as input
        filename = result["source_file"].replace(".pdf", ".json")
        output_path = output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"  Saved: {output_path.name}")
        saved += 1

    return saved


def get_existing_parsed(output_dir: Path) -> set[str]:
    """Get set of already parsed symbols."""
    if not output_dir.exists():
        return set()
    return {f.stem.split("_")[0] for f in output_dir.glob("*.json")}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse 10-K/20-F filings using GPT-4.1"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of filings to process",
    )
    parser.add_argument(
        "--symbol", type=str, default=None,
        help="Process a single symbol only",
    )
    parser.add_argument(
        "--input-dir", type=str, default=None,
        help="Custom input directory (default: reports/10_k_filings)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Custom output directory (default: parsed_reports/10_k_filings)",
    )
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"Max concurrent API calls (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for processing (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--no-skip", action="store_true",
        help="Re-parse already processed filings",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir) if args.input_dir else INPUT_DIR
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find filing files (.pdf)
    if args.symbol:
        file_paths = list(input_dir.glob(f"{args.symbol.upper()}_*.pdf"))
    else:
        file_paths = list(input_dir.glob("*.pdf"))

    # Sort and limit
    file_paths = sorted(file_paths, key=lambda p: p.name)

    # Skip already parsed
    if not args.no_skip:
        existing = get_existing_parsed(output_dir)
        before = len(file_paths)
        file_paths = [
            f for f in file_paths if f.stem.split("_")[0] not in existing
        ]
        skipped = before - len(file_paths)
        if skipped:
            print(f"Skipping {skipped} already parsed filings")

    if args.limit:
        file_paths = file_paths[: args.limit]

    if not file_paths:
        print(f"No PDF files to process in {input_dir}")
        return

    print(f"Found {len(file_paths)} filings to process")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Workers: {args.workers}, Batch size: {args.batch_size}")
    print()

    # Process in batches
    total_saved = 0
    total_files = len(file_paths)
    batch_size = args.batch_size

    for batch_start in range(0, total_files, batch_size):
        batch_end = min(batch_start + batch_size, total_files)
        batch_files = file_paths[batch_start:batch_end]
        batch_num = batch_start // batch_size + 1
        total_batches = (total_files + batch_size - 1) // batch_size

        print(f"\n{'='*50}")
        print(f"BATCH {batch_num}/{total_batches} ({len(batch_files)} files)")
        print(f"{'='*50}")

        results = asyncio.run(process_all_filings(batch_files, args.workers))
        saved = save_results(results, output_dir)
        total_saved += saved

        print(f"Batch {batch_num} complete: {saved}/{len(batch_files)} saved")

    print(f"\n{'='*50}")
    print(f"COMPLETE: {total_saved}/{total_files} filings parsed successfully")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
