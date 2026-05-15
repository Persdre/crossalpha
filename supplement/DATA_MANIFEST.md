# CrossAlpha Data Manifest

The full CrossAlpha release contains:

- Standardized annual-report text under a ten-category business schema.
- PCA-whitened embedding matrices.
- Directed cross-market similarity matrices.
- Typed economic-linkage edges for the US-to-Japan corridor.
- Daily OHLCV data aligned to the benchmark universe.
- Monthly and event-conditioned evaluation outputs.
- Metadata dictionaries for ticker, market, sector, and filing year.

The anonymous review package includes only lightweight metadata and result summaries. The full dataset should be released after review through a public dataset host and an archival DOI.

The review data archive also includes `standardized_reports/`, a compact JSON version of parsed annual reports under the ten-category schema. These files are included for inspection and auditability without requiring reviewers to download the full raw PDF/HTML corpus.

## Suggested Public Dataset Layout

```text
crossalpha/
  README.md
  dataset_card.md
  metadata/
  standardized_text/
  embeddings/
  similarity/
  typed_edges/
  market_data/
  benchmark_results/
  scripts/
```
