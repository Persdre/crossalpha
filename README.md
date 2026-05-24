# CrossAlpha

This repository contains the code and lightweight data artifacts for the paper
**"CrossAlpha: A Cross-Market Annual Report Dataset for Global Equity Factor Research."**
It is the anonymized supplement for a double-blind submission.

## Overview

![CrossAlpha Overview](figures/crossalpha_overview.png)

*CrossAlpha standardises annual reports from multiple equity markets, constructs
cross-market peer-similarity graphs from the resulting firm-level text
representations, and aligns the graph with price data for return-prediction
benchmarks.*

When a US firm's annual report describes its Asian suppliers, customers, and
competitors, that disclosure can carry predictive information about firms in
*other* markets. Existing financial-NLP resources offer little infrastructure
for evaluating such cross-market firm links at scale.

**CrossAlpha** is a public cross-market annual-report benchmark for firm-link
reasoning and information-flow evaluation. It harmonizes filings from five
regulatory systems and releases three aligned artifacts:

- a **~19M-edge dense cross-market similarity graph** built from PCA-whitened
  embeddings of an LLM-distilled ten-category business schema;
- an **8,962-edge typed economic-linkage overlay** (plus 744 retriever-only
  edges) on the US→Japan corridor;
- **11 years of daily OHLCV data** with monthly and event-conditioned
  evaluation harnesses.

| | |
|---|---|
| Markets | US (Russell 1000), Japan (TOPIX 500), Taiwan (TWSE), Korea (KOSPI), Hong Kong (main board) |
| Coverage | ~3,600 companies, ~10,700 firm-years |
| Evaluation window | January 2015 – December 2025 (105 monthly rebalances) |
| Source–target pairs | 25 ordered market pairs |
| Event task | 25 systematic source stocks, 606 GPT-5 post-cutoff events |

## How the Factor Is Built

All monthly experiments share one standardized, monthly-rebalanced
portfolio-sort pipeline and differ only in how peers are defined. For a target
stock *i* in market *a* and a source market *b*, the cross-market factor
aggregates the sector-relative returns of source-market peers, weighted by a
sigmoid-transformed text-similarity rank (top-1% per row):

```
f_i(t) = sum_j ( alpha_ij * r_j(t) ) / sum_j ( alpha_ij ),   j in source market b
```

where `r_j(t)` is peer *j*'s L-month sector-relative cumulative return and
`alpha_ij` is the similarity weight. At each month-end, target stocks are ranked
by factor value, split into quintiles, and evaluated as a long–short (Q5−Q1) and
a long-only (Q5) book at 2 bp one-way cost. Reported metrics are the monthly
rank IC, its annualized information ratio (ICIR), and the long–short Sharpe and
maximum drawdown. A strict 1-month execution lag is applied.

## Dataset Construction Pipeline

The repository ships the full pipeline so the construction process is auditable
end to end, not just the final evaluation. The stages run in order; each
produces an artifact consumed by the next.

| # | Stage | Script | Output |
|---|---|---|---|
| 1 | **Schema extraction** | `scripts/parse_10k.py` (US 10-K/20-F), `scripts/parse_reports_claude.py` (all markets) | per-firm JSON under a fixed **ten-category business schema**, distilled by an LLM (the exact `SYSTEM_PROMPT` is in these files) |
| 2 | **Embedding** | `scripts/generate_embeddings.py` | a 3072-d `text-embedding-3-large` vector per category per firm |
| 3 | **PCA whitening** | `scripts/whiten_embeddings.py` | a shared 128-d whitened basis (`Cov(Z)=I`) fit jointly across all markets |
| 4 | **Similarity graph** | `scripts/cache_similarity.py` | weighted cross-category cosine similarity → per-row percentile-rank matrices (the dense cross-market graph) |
| 5 | **Typed edges** | `scripts/run_supply_chain_extraction.py` | LLM-extracted supplier/customer/competitor edges on the US→JP corridor |
| 6 | **Factor** | `scripts/run_graph_monthly.py`, `scripts/run_us_ema_mr_to_jp_mvp.py` | the monthly cross-market peer-momentum factor (Eq. above) |
| 7 | **RQ3 agent** | `scripts/run_agentic_eval.py`, `scripts/agentic_eval_helpers.py`, `scripts/agentic_retrieval_tools.py` | agent-weighted variants of the graph and their event-conditioned evaluation |

Raw filings are collected with the `ar_scraper/` toolkit (one scraper per
market). Stages 1, 5, and 7 call hosted LLM APIs and read `OPENAI_API_KEY` /
`ANTHROPIC_API_KEY` from the environment; no keys are stored in the repo.

## Repository Structure

```text
crossalpha/
├── README.md                            # this file
├── README_REVIEWER.md                   # short reviewer pointer
├── requirements.txt
├── pyproject.toml
├── ANONYMIZATION_CHECKLIST.md
├── scripts/
│   │  # — construction pipeline (stages 1–7) —
│   ├── parse_10k.py                         # 1. US filings → ten-category schema (LLM)
│   ├── parse_reports_claude.py              # 1. all-market parser → ten-category schema
│   ├── generate_embeddings.py               # 2. category text → 3072-d embeddings
│   ├── whiten_embeddings.py                 # 3. PCA whitening → shared 128-d basis
│   ├── cache_similarity.py                  # 4. cosine → percentile-rank similarity graph
│   ├── run_supply_chain_extraction.py       # 5. typed supplier/customer/competitor edges
│   ├── run_graph_monthly.py                 # 6. cross-market peer-momentum factor
│   ├── run_us_ema_mr_to_jp_mvp.py           # 6. US→JP factor variant
│   ├── run_agentic_eval.py                  # 7. agent-weighted graph evaluation
│   ├── agentic_eval_helpers.py              # 7. RQ3 agent eval helpers
│   ├── agentic_retrieval_tools.py           # 7. RQ3 vector-search retrieval tool
│   │  # — evaluation entry points (paper tables/figures) —
│   ├── run_baselines_monthly.py             # RQ1: US→JP peer-definition comparison
│   ├── run_paper_multi_market_monthly.py    # RQ2: raw US→target generalization
│   ├── run_cross_vs_same_market_matrix.py   # RQ2: 5×5 Information-Geography Matrix
│   ├── run_pooled_similarity_mild.py        # RQ2: best-single vs pooled sources
│   ├── run_category_ablation_multi_target.py# ablation: ten-category schema
│   ├── run_rq3_mild_rerun.py                # RQ3: event-driven daily evaluation
│   ├── similarity_cache.py                  # similarity-matrix cache helper
│   ├── kernels/                             # similarity, neutralization, ranking, returns
│   ├── backtesting/                         # portfolio-sort engines, metrics, reports
│   └── h5_data/                             # HDF5 panel loaders
├── ar_scraper/                          # annual-report scraper toolkit (5 markets)
└── supplement/
    ├── results/                         # JSON outputs behind the paper tables/figures
    │   ├── baselines_monthly_12mo.json          # RQ1 (Table: baselines)
    │   ├── cross_vs_same_market_matrix_12mo.json# RQ2 (Information-Geography Matrix)
    │   ├── pooled_similarity_12mo.json          # RQ2 (best-single vs pooled)
    │   ├── category_ablation_multi_target_12mo.json # schema ablation
    │   └── rq3_event_results.json               # RQ3 (event strategy)
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

## Setup

Use Python 3.11 or newer.

```bash
pip install -r requirements.txt
```

The experiment scripts read the project and data roots from environment
variables (no user-specific absolute paths are hard-coded):

```bash
export CROSSALPHA_PROJECT_ROOT=/path/to/this/repo
export CROSSALPHA_DATA_ROOT=/path/to/crossalpha-data
```

## Reproduction

**Building the dataset from raw filings** (stages 1–5; needs the raw corpus and
LLM API keys, released after review):

```bash
python scripts/parse_10k.py                  # 1. filings → ten-category schema
python scripts/generate_embeddings.py        # 2. embeddings
python scripts/whiten_embeddings.py          # 3. PCA whitening → 128-d basis
python scripts/cache_similarity.py           # 4. similarity graph
python scripts/run_supply_chain_extraction.py# 5. typed economic-linkage edges
```

**Evaluation** — each entry point corresponds to a table or figure in the
paper and runs off the cached similarity / shipped results:

```bash
# RQ1 — US→JP peer-definition comparison (text vs GICS / return-corr / domestic)
python scripts/run_baselines_monthly.py --lookback 12

# RQ2 — raw US→target generalization across the four Asian targets
python scripts/run_paper_multi_market_monthly.py

# RQ2 — 5×5 Information-Geography Matrix (per-source-target rank ICIR)
python scripts/run_cross_vs_same_market_matrix.py

# RQ2 — best single source vs pooled sources per target
python scripts/run_pooled_similarity_mild.py

# Ablation — ten-category business schema
python scripts/run_category_ablation_multi_target.py

# RQ3 — event-conditioned daily basket evaluation
python scripts/run_rq3_mild_rerun.py
```

The JSON files under `supplement/results/` are the exact outputs behind the
reported numbers, so reviewers can verify the paper's tables and figures without
re-running anything.

## Data Availability

This anonymous review package ships the **standardized ten-category JSON text
for all five markets** (US, JP, TW, KR, HK; ~5,900 firm-filings under
`supplement/standardized_reports/`), the result summaries, and the metadata
dictionaries — enough to inspect the inputs and verify the reported numbers.
The US filings are released as three fiscal-year vintages (FY2022–2024); the
other four markets ship the FY2024 vintage used by the cross-market evaluation.
Reports are organised by filing-year cohort, so for firms with non-December
fiscal years the cohort label may lead the fiscal year by one period.
The raw filing PDFs/HTML, PCA-whitened embedding matrices,
directed similarity matrices, typed economic-linkage edges, and the complete
daily OHLCV panel are large and will be released through public dataset hosting
with an archival DOI after review. See `supplement/DATA_MANIFEST.md` for the
full-release layout.

## Anonymity Note

This repository intentionally contains no author names, affiliations, local
machine paths, personal repository links, or project-account URLs. See
`ANONYMIZATION_CHECKLIST.md`.
