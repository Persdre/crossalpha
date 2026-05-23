#!/usr/bin/env python3
"""Extract explicit supply chain relationships using Claude.

For each US company, extract named customers/suppliers from parsed
annual report text. Build a supply chain graph, then construct
peer momentum factors using graph edges vs embedding similarity.

Uses Claude Haiku for cost efficiency (~$0.25/1M input tokens).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_ROOT = Path("${CROSSALPHA_DATA_ROOT}")
MARKET_DATA_PATH = str(DATA_ROOT / "market_data")
SYMBOL_DICT_PATH = PROJECT_ROOT / "docs" / "SymbolDict.csv"

PARSED_US = DATA_ROOT / "parsed_reports" / "10_k_filings"
PARSED_JP = DATA_ROOT / "parsed_reports" / "securities_reports"

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


EXTRACTION_PROMPT = """You are a financial analyst. From the following business description of {company} ({symbol}), identify companies that are their customers, suppliers, partners, or competitors.

There are TWO types of identification:
1. EXPLICIT: Companies directly named in the text
2. INFERRED: Companies not named but strongly implied by your knowledge (e.g., if Apple says "outsources manufacturing to partners in China", you know this means Foxconn/Hon Hai)

Return a JSON object with exactly this format:
{{
  "explicit": {{
    "customers": ["Company A"],
    "suppliers": ["Company B"],
    "partners": ["Company C"],
    "competitors": ["Company D"]
  }},
  "inferred": {{
    "customers": ["Company E"],
    "suppliers": ["Company F"],
    "partners": [],
    "competitors": ["Company G"]
  }}
}}

Rules:
- Use SPECIFIC publicly-traded COMPANY NAMES only (ticker-identifiable)
- For INFERRED: only include companies you are highly confident about (>80%)
- Use parent company names (e.g., "Samsung" not "Samsung Electronics Co., Ltd.")
- Empty list [] if none found
- Keep each list to max 10 most important companies

Text:
{text}"""


async def extract_relationships(texts: dict[str, dict[str, str]], max_concurrent: int = 30) -> dict:
    """Extract supply chain relationships from parsed texts using GPT-4.1-mini."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(max_concurrent)
    results = {}
    errors = 0

    async def process_one(symbol: str, company: str, text: str):
        nonlocal errors
        async with semaphore:
            try:
                resp = await client.chat.completions.create(
                    model="gpt-4.1-mini",
                    max_tokens=500,
                    temperature=0,
                    messages=[{
                        "role": "user",
                        "content": EXTRACTION_PROMPT.format(
                            company=company, symbol=symbol, text=text
                        )
                    }]
                )
                content = resp.choices[0].message.content
                # Parse JSON from response
                start = content.find("{")
                end = content.rfind("}") + 1
                if start >= 0 and end > start:
                    data = json.loads(content[start:end])
                    results[symbol] = data
                else:
                    errors += 1
            except Exception as e:
                errors += 1

    tasks = []
    for symbol, info in texts.items():
        # Combine supply chain + customer text
        text = ""
        if info.get("supply_chain_position"):
            text += "Supply chain: " + info["supply_chain_position"] + "\n"
        if info.get("primary_customers_markets"):
            text += "Customers: " + info["primary_customers_markets"] + "\n"
        if info.get("key_competitors_industry_positioning"):
            text += "Competitors: " + info["key_competitors_industry_positioning"] + "\n"
        if text:
            company = info.get("company", symbol)
            tasks.append(process_one(symbol, company, text))

    print(f"  Extracting from {len(tasks)} reports...", flush=True)

    # Process in batches
    batch_size = 50
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i+batch_size]
        await asyncio.gather(*batch)
        print(f"  [{i+len(batch)}/{len(tasks)}] extracted={len(results)}, errors={errors}", flush=True)

    return results


def build_supply_chain_graph(us_relations: dict, symbol_set: set) -> dict[str, dict[str, list[str]]]:
    """Build graph: for each company, find which other companies in our universe
    are named as customers/suppliers/competitors."""

    # Normalize company names for matching
    from difflib import get_close_matches

    # Build name → symbol mapping
    sym_df = pl.read_csv(SYMBOL_DICT_PATH)
    name_to_sym = {}
    for row in sym_df.iter_rows(named=True):
        sym = row["SYMBOL"]
        company = row.get("company", "")
        if sym in symbol_set and company:
            name_to_sym[company.lower()] = sym
            # Also add symbol without suffix
            name_to_sym[sym.split(".")[0].lower()] = sym

    graph = {}  # symbol → {"customers": [syms], "suppliers": [syms], "competitors": [syms]}

    for symbol, rels in us_relations.items():
        edges = {"customers": [], "suppliers": [], "competitors": [], "partners": []}
        for rel_type in ["customers", "suppliers", "competitors", "partners"]:
            for name in rels.get(rel_type, []):
                name_lower = name.lower()
                # Try exact match
                if name_lower in name_to_sym:
                    edges[rel_type].append(name_to_sym[name_lower])
                    continue
                # Try close match
                matches = get_close_matches(name_lower, name_to_sym.keys(), n=1, cutoff=0.8)
                if matches:
                    edges[rel_type].append(name_to_sym[matches[0]])

        if any(edges[k] for k in edges):
            graph[symbol] = edges

    return graph


def main():
    print("=" * 90)
    print("SUPPLY CHAIN EXTRACTION — Explicit vs Implicit Peer Momentum")
    print("=" * 90)

    # Load parsed US reports
    print("\nLoading parsed reports...")
    us_texts = {}
    sym_df = pl.read_csv(SYMBOL_DICT_PATH)
    r1000 = set(sym_df.filter(pl.col("region") == "us_russell_1000")["SYMBOL"].to_list())

    for f in PARSED_US.glob("*.json"):
        try:
            d = json.load(open(f))
            sym = d.get("symbol", "")
            if sym in r1000:
                cats = d.get("categories", {})
                if cats:
                    us_texts[sym] = cats
        except:
            pass
    print(f"  US: {len(us_texts)} reports with categories")

    # Extract relationships using Claude
    print("\nExtracting supply chain relationships with Claude Haiku...")
    relations = asyncio.run(extract_relationships(us_texts))

    # Save raw extraction
    out_path = PROJECT_ROOT / "backtest_results_daily" / "supply_chain_relations.json"
    with open(out_path, "w") as f:
        json.dump(relations, f, indent=2)
    print(f"\n  Saved {len(relations)} extractions to {out_path}")

    # Stats
    total_edges = 0
    for sym, rels in relations.items():
        for k in ["customers", "suppliers", "competitors", "partners"]:
            total_edges += len(rels.get(k, []))
    print(f"  Total named entities: {total_edges}")
    print(f"  Avg per company: {total_edges / len(relations):.1f}")

    # Build graph
    print("\nBuilding supply chain graph...")
    all_symbols = set(us_texts.keys())
    graph = build_supply_chain_graph(relations, all_symbols)

    # Graph stats
    total_matched = sum(
        len(v.get("customers", [])) + len(v.get("suppliers", [])) +
        len(v.get("competitors", [])) + len(v.get("partners", []))
        for v in graph.values()
    )
    print(f"  Companies with matched edges: {len(graph)}")
    print(f"  Total matched edges (in universe): {total_matched}")

    # Save graph
    graph_path = PROJECT_ROOT / "backtest_results_daily" / "supply_chain_graph.json"
    with open(graph_path, "w") as f:
        json.dump(graph, f, indent=2)

    # ============================================================
    # Build factor from explicit graph edges
    # ============================================================
    print("\nBuilding explicit supply chain peer momentum...")

    from scripts.kernels import load_daily_close_prices, sigmoid_weights
    from scripts.daily_pooled_pipeline import compute_daily_returns

    # JP target, US peer (same as implicit)
    jp_prices = load_daily_close_prices("jp_topix_500", "2015-01-01", "2025-12-31", source_path=MARKET_DATA_PATH)
    us_prices = load_daily_close_prices("us_russell_1000", "2015-01-01", "2025-12-31", source_path=MARKET_DATA_PATH)
    us_wide = (us_prices.pivot(index="DATETIME", on="SYMBOL", values="close")
               .sort("DATETIME").fill_null(strategy="forward")
               .unpivot(index="DATETIME", variable_name="SYMBOL", value_name="close")
               .drop_nulls("close"))

    labels = (jp_prices.sort(["SYMBOL", "DATETIME"])
              .with_columns(((pl.col("close").shift(-1) / pl.col("close")) - 1).over("SYMBOL").alias("daily_return"))
              .drop_nulls("daily_return").select(["DATETIME", "SYMBOL", "daily_return"]))

    # Load JP→US embedding similarity for cross-referencing
    from scripts.similarity_cache import load_similarity_cache
    cached = load_similarity_cache("jp_topix_500", "us_russell_1000", "w128",
                                    cache_dir=DATA_ROOT / "cache" / "similarity")

    # For each JP stock, find US stocks that are explicitly named as
    # customers/suppliers/competitors of JP stock's most similar US peers
    # This is a two-hop: JP stock → similar US peer → US peer's named relationships

    # Simpler approach: For each US stock with graph edges, weight its peers'
    # returns by graph connectivity (vs embedding similarity)

    # Actually, let's compare:
    # 1. Implicit: embedding similarity (what we already have)
    # 2. Explicit: graph-based (customers/suppliers/competitors)
    # 3. Combined: implicit + explicit

    # For explicit factor on JP:
    # JP stock i's explicit peers = US stocks that are supply-chain connected
    # to JP stock i's embedding-similar US peers
    # This is complex. Simpler: use graph to find US stock clusters,
    # then check if graph-clustered US returns predict JP returns.

    # Actually simplest and most interesting comparison:
    # For US→US domestic: compare embedding vs graph peer momentum
    # This directly tests "do explicit supply chain links predict better?"

    print("\n  Testing US domestic: explicit graph vs embedding peer momentum...")

    us_labels = (us_prices.sort(["SYMBOL", "DATETIME"])
                 .with_columns(((pl.col("close").shift(-1) / pl.col("close")) - 1).over("SYMBOL").alias("daily_return"))
                 .drop_nulls("daily_return").select(["DATETIME", "SYMBOL", "daily_return"]))

    us_sym_to_idx = {}
    us_syms_list = sorted(set(us_wide["SYMBOL"].unique().to_list()) & set(us_labels["SYMBOL"].unique().to_list()))
    us_sym_to_idx = {s: i for i, s in enumerate(us_syms_list)}

    # Build explicit weight matrix from graph
    n_us = len(us_syms_list)
    explicit_weights = np.zeros((n_us, n_us))
    for sym, edges in graph.items():
        if sym not in us_sym_to_idx:
            continue
        i = us_sym_to_idx[sym]
        all_peers = set()
        for k in ["customers", "suppliers", "competitors", "partners"]:
            for peer in edges.get(k, []):
                if peer in us_sym_to_idx:
                    all_peers.add(peer)
        for peer in all_peers:
            j = us_sym_to_idx[peer]
            explicit_weights[i, j] = 1.0
            explicit_weights[j, i] = 1.0  # bidirectional

    # Count coverage
    has_explicit = np.sum(explicit_weights.sum(axis=1) > 0)
    avg_peers = explicit_weights.sum() / max(has_explicit, 1)
    print(f"  Explicit graph: {has_explicit}/{n_us} stocks with edges, avg {avg_peers:.1f} peers")

    # Compute factor: for each US stock, avg return of its explicit peers
    from scripts.daily_pooled_pipeline import PrecomputedWeights, compute_daily_factor_single_peer

    pw_explicit = PrecomputedWeights(explicit_weights, us_syms_list, us_syms_list, us_sym_to_idx)

    results = []
    for lb in [5, 20]:
        returns = compute_daily_returns(us_wide, lb)

        # Explicit factor
        factor_exp = compute_daily_factor_single_peer(pw_explicit, returns)
        if factor_exp.height > 0:
            # Winsorize
            recs = []
            for dt in factor_exp["DATETIME"].unique().sort().to_list():
                day = factor_exp.filter(pl.col("DATETIME") == dt)
                vals = day["factor_value"].to_numpy().astype(np.float64)
                syms = day["SYMBOL"].to_list()
                m = np.isfinite(vals)
                if m.sum() < 5: continue
                v = vals[m]; s = [ss for ss, mm in zip(syms, m) if mm]
                lo, hi = np.nanpercentile(v, 1), np.nanpercentile(v, 99)
                v = np.clip(v, lo, hi)
                mu, sd = np.mean(v), np.std(v)
                v = (v - mu) / sd if sd > 1e-10 else v - mu
                for sym, fv in zip(s, v):
                    recs.append({"DATETIME": dt, "SYMBOL": sym, "factor_value": float(fv)})
            fwz = pl.DataFrame(recs)
            df = fwz.join(us_labels, on=["DATETIME", "SYMBOL"], how="inner")
            ics = []
            for dt in df["DATETIME"].unique().sort().to_list():
                day = df.filter(pl.col("DATETIME") == dt)
                f, r = day["factor_value"].to_numpy(), day["daily_return"].to_numpy()
                mask = np.isfinite(f) & np.isfinite(r)
                if mask.sum() < 10: continue
                rho, _ = spearmanr(f[mask], r[mask])
                ics.append(rho)
            ics = np.array(ics)
            icm, ics_s = float(np.nanmean(ics)), float(np.nanstd(ics))
            icir = icm / ics_s * np.sqrt(252) if ics_s > 0 else 0
            print(f"  Explicit graph {lb}d: IC={icm:+.4f} ICIR={icir:.3f} ({len(ics)} days)")
            results.append({"method": "explicit_graph", "lookback": lb, "ic": icm, "icir": icir, "n_days": len(ics)})

    # Compare with embedding domestic (US→US)
    print("\n  Embedding domestic (US→US) for comparison...")
    us_cached = load_similarity_cache("us_russell_1000", "us_russell_1000", "w128",
                                       cache_dir=DATA_ROOT / "cache" / "similarity")
    us_emb_weights = sigmoid_weights(us_cached.pct_ranks, k=50, c=0.99)
    pw_emb = PrecomputedWeights(us_emb_weights, us_cached.target_symbols, us_cached.peer_symbols,
                                 {s: i for i, s in enumerate(us_cached.peer_symbols)})

    us_wide2 = (us_prices.pivot(index="DATETIME", on="SYMBOL", values="close")
                .sort("DATETIME").fill_null(strategy="forward")
                .unpivot(index="DATETIME", variable_name="SYMBOL", value_name="close")
                .drop_nulls("close"))

    for lb in [5, 20]:
        returns = compute_daily_returns(us_wide2, lb)
        factor_emb = compute_daily_factor_single_peer(pw_emb, returns)
        if factor_emb.height > 0:
            recs = []
            for dt in factor_emb["DATETIME"].unique().sort().to_list():
                day = factor_emb.filter(pl.col("DATETIME") == dt)
                vals = day["factor_value"].to_numpy().astype(np.float64)
                syms = day["SYMBOL"].to_list()
                m = np.isfinite(vals)
                if m.sum() < 5: continue
                v = vals[m]; s = [ss for ss, mm in zip(syms, m) if mm]
                lo, hi = np.nanpercentile(v, 1), np.nanpercentile(v, 99)
                v = np.clip(v, lo, hi)
                mu, sd = np.mean(v), np.std(v)
                v = (v - mu) / sd if sd > 1e-10 else v - mu
                for sym, fv in zip(s, v):
                    recs.append({"DATETIME": dt, "SYMBOL": sym, "factor_value": float(fv)})
            fwz = pl.DataFrame(recs)
            df = fwz.join(us_labels, on=["DATETIME", "SYMBOL"], how="inner")
            ics = []
            for dt in df["DATETIME"].unique().sort().to_list():
                day = df.filter(pl.col("DATETIME") == dt)
                f, r = day["factor_value"].to_numpy(), day["daily_return"].to_numpy()
                mask = np.isfinite(f) & np.isfinite(r)
                if mask.sum() < 10: continue
                rho, _ = spearmanr(f[mask], r[mask])
                ics.append(rho)
            ics = np.array(ics)
            icm, ics_s = float(np.nanmean(ics)), float(np.nanstd(ics))
            icir = icm / ics_s * np.sqrt(252) if ics_s > 0 else 0
            print(f"  Embedding domestic {lb}d: IC={icm:+.4f} ICIR={icir:.3f}")
            results.append({"method": "embedding_domestic", "lookback": lb, "ic": icm, "icir": icir, "n_days": len(ics)})

    # Save results
    out = PROJECT_ROOT / "backtest_results_daily" / "experiment_supply_chain_explicit.json"
    with open(out, "w") as f:
        json.dump({"relations_count": len(relations), "graph_edges": total_matched,
                    "graph_coverage": int(has_explicit), "results": results}, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
