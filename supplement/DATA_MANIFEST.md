# CrossAlpha Data Manifest

The full CrossAlpha release contains:

- Standardized annual-report text under a ten-category business schema.
- PCA-whitened embedding matrices.
- Directed cross-market similarity matrices.
- Typed economic-linkage edges for the US-to-Japan corridor.
- Daily OHLCV data aligned to the benchmark universe.
- Monthly and event-conditioned evaluation outputs.
- Metadata dictionaries for ticker, market, sector, and filing year.

The anonymous review package includes the standardized text, metadata, and result summaries. The remaining large artifacts (raw PDF/HTML, embedding/similarity matrices, daily OHLCV) are released after review through a public dataset host and an archival DOI.

`standardized_reports/` holds the compact ten-category JSON text for **all five markets**, one file per firm-filing, so reviewers can inspect the model inputs and audit the schema without downloading the raw PDF/HTML corpus. Directory-to-market mapping:

| Directory | Market | Source filing | Files |
|---|---|---|---|
| `10_k_filings/` | US (Russell 1000), FY2024 | 10-K / 20-F | 767 |
| `10_k_filings_fy2022/` | US (Russell 1000), FY2022 | 10-K / 20-F | 863 |
| `10_k_filings_fy2023/` | US (Russell 1000), FY2023 | 10-K / 20-F | 671 |
| `securities_reports/` | Japan (TOPIX 500), FY2024 | Securities Report | 1,372 |
| `securities_reports_tw/` | Taiwan (TWSE), FY2024 | Annual Report | 1,040 |
| `securities_reports_kr/` | South Korea (KOSPI), FY2024 | Business Report | 612 |
| `securities_reports_hk/` | Hong Kong (main board), FY2024 | Annual Report | 572 |

The US filings ship three fiscal-year vintages (FY2022–2024); the other four markets ship
the FY2024 vintage used by the cross-market evaluation. Total: 5,897 standardized firm-filing
JSON files.

Each JSON has `symbol`, `form_type`, `filing_date`, `source_file`, and a `categories` object with the ten business-description fields.

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
