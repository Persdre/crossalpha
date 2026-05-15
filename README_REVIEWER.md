# CrossAlpha Anonymous Supplement

This anonymized supplement contains code and lightweight data artifacts for reproducing the experiments described in the paper.

## Contents

- `scripts/`: experiment, backtesting, similarity, and data-loading code.
- `ar_scraper/`: source code for the annual-report scraper toolkit.
- `metadata/`: symbol and data dictionaries.
- `results/`: JSON outputs used by the paper tables and figures.
- `standardized_reports/`: compact ten-category annual-report JSON files.
- `DATA_MANIFEST.md`: description of the full dataset release.

## Setup

Create a Python environment with Python 3.11 or newer, then install:

```bash
pip install -r requirements.txt
```

Set the data root before running experiments:

```bash
export CROSSALPHA_DATA_ROOT=/path/to/crossalpha-data
export CROSSALPHA_PROJECT_ROOT=/path/to/crossalpha-code
```

## Reproduction Entry Points

```bash
python scripts/run_baselines_monthly.py --lookback 12
python scripts/run_cross_vs_same_market_matrix.py
python scripts/run_category_ablation_multi_target.py
python scripts/run_paper_multi_market_monthly.py
```

The review package includes lightweight result files so reviewers can inspect the reported numbers without downloading the full raw corpus.

## Notes

The full raw filing corpus and full market-data panel are large and are described in the data manifest. They will be released through public dataset hosting after review.
