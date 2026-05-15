# CrossAlpha Anonymous Supplement

This repository is an anonymized supplement for a double-blind ARR submission.

## What Is Included

- `scripts/`: code for monthly baselines, cross-market matrices, category ablations, pooled source experiments, and event-spillover evaluation.
- `scripts/backtesting/`, `scripts/kernels/`, `scripts/h5_data/`: shared evaluation utilities.
- `ar_scraper/`: source code for the annual-report scraper toolkit. Downloaded report corpora are excluded.
- `supplement/results/`: lightweight JSON outputs used by paper tables and figures.
- `supplement/standardized_reports/`: compact ten-category annual-report JSON files in the data archive.
- `supplement/metadata/`: symbol and data dictionaries.
- `supplement/DATA_MANIFEST.md`: description of the full release layout.

## Setup

Use Python 3.11 or newer.

```bash
pip install -r requirements.txt
```

Set paths before running experiments:

```bash
export CROSSALPHA_PROJECT_ROOT=/path/to/this/repo
export CROSSALPHA_DATA_ROOT=/path/to/crossalpha-data
```

## Reproduction Entry Points

```bash
python scripts/run_baselines_monthly.py --lookback 12
python scripts/run_cross_vs_same_market_matrix.py
python scripts/run_category_ablation_multi_target.py
python scripts/run_paper_multi_market_monthly.py
```

The repository includes lightweight result files for inspection. The full report corpus, embeddings, similarity matrices, and market-data panel are large and will be released through public dataset hosting after review.

## Double-Blind Note

This repository intentionally contains no author names, affiliations, local machine paths, personal repository links, or project-account URLs.
