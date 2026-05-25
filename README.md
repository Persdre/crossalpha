# CrossAlpha: A Cross-Market Annual Report Benchmark for Global Equity Factor Research

**Official code and data for the CrossAlpha benchmark.**

This repository contains the official implementation of **CrossAlpha**, a cross-market
annual-report benchmark for global equity factor research. It standardizes annual reports
from five equity markets into a fixed ten-category business schema, builds cross-market
peer-similarity graphs from the resulting firm-level text embeddings, and aligns those
graphs with daily price data — then ships fixed evaluation entry points and reference
results so cross-market peer-momentum factors can be compared under a common protocol. It
is the anonymized supplement for a double-blind submission.

<p align="center">
  <img src="figures/crossalpha_overview.png" alt="CrossAlpha Overview" width="100%">
</p>

| | |
|---|---|
| Markets | US (Russell 1000), Japan (TOPIX 500), Taiwan (TWSE), Korea (KOSPI), Hong Kong (main board) |
| Coverage | ~3,600 companies, ~10,700 firm-years |
| Price panel | 11 years of daily OHLCV (January 2015 – December 2025) |
| Similarity graph | ~19M-edge dense cross-market graph from PCA-whitened embeddings |

## Installation

Use Python 3.11 or newer.

```bash
pip install -r requirements.txt
```

The scripts read the project and data roots from environment variables, so no
user-specific absolute paths are hard-coded:

```bash
export CROSSALPHA_PROJECT_ROOT=/path/to/this/repo
export CROSSALPHA_DATA_ROOT=/path/to/crossalpha-data
```

Stages that call hosted LLM APIs read `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` from the
environment; no keys are stored in the repo.

## Project Structure

```text
crossalpha/
├── requirements.txt
├── scripts/
│   │  # — construction pipeline (stages 1–5) —
│   ├── parse_10k.py                         # 1. US filings → ten-category schema (LLM)
│   ├── parse_reports_claude.py              # 1. all-market parser → ten-category schema
│   ├── generate_embeddings.py               # 2. category text → 3072-d embeddings
│   ├── whiten_embeddings.py                 # 3. PCA whitening → shared 128-d basis
│   ├── cache_similarity.py                  # 4. cosine → percentile-rank similarity graph
│   ├── run_graph_monthly.py                 # 5. cross-market peer-momentum factor
│   ├── run_us_ema_mr_to_jp_mvp.py           # 5. US→JP factor variant
│   ├── run_agentic_eval.py                  # agent-weighted graph evaluation
│   ├── agentic_eval_helpers.py              # agent eval helpers
│   ├── agentic_retrieval_tools.py           # vector-search retrieval tool
│   │  # — evaluation entry points —
│   ├── run_baselines_monthly.py             # US→JP peer-definition comparison
│   ├── run_paper_multi_market_monthly.py    # raw US→target generalization
│   ├── run_cross_vs_same_market_matrix.py   # 5×5 cross- vs same-market matrix
│   ├── run_pooled_similarity_mild.py        # best-single vs pooled sources
│   ├── run_category_ablation_multi_target.py# ten-category schema ablation
│   ├── run_rq3_mild_rerun.py                # event-driven daily evaluation
│   ├── similarity_cache.py                  # similarity-matrix cache helper
│   ├── kernels/                             # similarity, neutralization, ranking, returns
│   ├── backtesting/                         # portfolio-sort engines, metrics, reports
│   └── h5_data/                             # HDF5 panel loaders
├── ar_scraper/                          # annual-report scraper toolkit (5 markets)
└── supplement/
    ├── results/                         # JSON outputs behind the paper tables/figures
    ├── metadata/                        # SymbolDict.csv, DataDict.csv
    ├── standardized_reports/            # ten-category JSON text, all 5 markets (FY2022–2024)
    │   ├── 10_k_filings/                #   US  FY2024 (767)
    │   ├── 10_k_filings_fy2022/         #   US  FY2022 vintage (863)
    │   ├── 10_k_filings_fy2023/         #   US  FY2023 vintage (671)
    │   ├── securities_reports/          #   JP  (1,372)
    │   ├── securities_reports_tw/       #   TW  (1,040)
    │   ├── securities_reports_kr/       #   KR  (612)
    │   └── securities_reports_hk/       #   HK  (572)
    └── DATA_MANIFEST.md                 # full-release layout
```

## Data Preparation

The construction pipeline runs in order; each stage produces an artifact consumed by the
next. Stages 1–4 need the raw corpus and LLM API keys (released after review).

```bash
python scripts/parse_10k.py                   # 1. filings → ten-category schema (LLM)
python scripts/generate_embeddings.py         # 2. category text → 3072-d embeddings
python scripts/whiten_embeddings.py           # 3. PCA whitening → shared 128-d basis
python scripts/cache_similarity.py            # 4. cosine → percentile-rank similarity graph
```

Raw filings are collected with the `ar_scraper/` toolkit (one scraper per market). The
exact LLM `SYSTEM_PROMPT` used for schema extraction lives in `parse_10k.py` and
`parse_reports_claude.py`.

## Building the Factor

The cross-market factor aggregates the sector-relative returns of source-market peers,
weighted by a sigmoid-transformed text-similarity rank (top-1% per row). For a target
stock *i* in market *a* and a source market *b*:

```
f_i(t) = sum_j ( alpha_ij * r_j(t) ) / sum_j ( alpha_ij ),   j in source market b
```

where `r_j(t)` is peer *j*'s L-month sector-relative cumulative return and `alpha_ij` is
the similarity weight. Build it from the cached similarity graph:

```bash
python scripts/run_graph_monthly.py           # cross-market peer-momentum factor
python scripts/run_us_ema_mr_to_jp_mvp.py     # US→JP factor variant
```

## Evaluation

Each entry point runs off the cached similarity graph and the shipped results. Override
the lookback or source/target arguments via `argparse` flags.

```bash
# US→JP peer-definition comparison (text vs GICS / return-corr / domestic)
python scripts/run_baselines_monthly.py --lookback 12

# raw US→target generalization across the four Asian targets
python scripts/run_paper_multi_market_monthly.py

# 5×5 cross- vs same-market matrix (per-source-target factor)
python scripts/run_cross_vs_same_market_matrix.py

# best single source vs pooled sources per target
python scripts/run_pooled_similarity_mild.py

# ten-category business-schema ablation
python scripts/run_category_ablation_multi_target.py

# event-conditioned daily basket evaluation
python scripts/run_rq3_mild_rerun.py
```

The JSON files under `supplement/results/` are the exact outputs behind the reported
numbers, so the tables and figures can be inspected without re-running anything.

## Construction Stages

| # | Stage | Script | Output |
|---|-------|--------|--------|
| 1 | Schema extraction | `parse_10k.py`, `parse_reports_claude.py` | per-firm JSON under a fixed ten-category business schema, distilled by an LLM |
| 2 | Embedding | `generate_embeddings.py` | a 3072-d `text-embedding-3-large` vector per category per firm |
| 3 | PCA whitening | `whiten_embeddings.py` | a shared 128-d whitened basis (`Cov(Z)=I`) fit jointly across all markets |
| 4 | Similarity graph | `cache_similarity.py` | weighted cross-category cosine similarity → per-row percentile-rank matrices |
| 5 | Factor | `run_graph_monthly.py` | the monthly cross-market peer-momentum factor |

## Data Availability

This anonymous review package ships the **standardized ten-category JSON text for all five
markets** (US, JP, TW, KR, HK; ~5,900 firm-filings under `supplement/standardized_reports/`),
the result summaries, and the metadata dictionaries. The US filings are released as three
fiscal-year vintages (FY2022–2024); the other four markets ship the FY2024 vintage used by
the cross-market evaluation. Reports are organized by filing-year cohort, so for firms with
non-December fiscal years the cohort label may lead the fiscal year by one period.

The raw filing PDFs/HTML, PCA-whitened embedding matrices, directed similarity matrices,
and the complete daily OHLCV panel are large and will be
released through public dataset hosting with an archival DOI after review. See
`supplement/DATA_MANIFEST.md` for the full-release layout.
