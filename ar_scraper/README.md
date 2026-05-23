# ar-scraper

Annual Report Scraper for US, Japan, Taiwan, Korea, and Hong Kong markets.

## Installation

```bash
# Base installation (Japan, Taiwan, Korea, Hong Kong support)
pip install ar-scraper

# With US support (requires WeasyPrint)
pip install ar-scraper[us]
```

## Usage

```bash
# By universe
ar-scraper --universe us_russell_1000 --year 2024
ar-scraper --universe jp_topix --year 2024
ar-scraper --universe tw_twse --year 2024
ar-scraper --universe kr_kospi --year 2024
ar-scraper --universe hk_hsci --year 2024

# By symbols (auto-detects region)
ar-scraper --symbols AAPL,MSFT,7974.T,2330.TW,005930.KS,0700.HK --year 2024

# From file
ar-scraper --symbols-file symbols.txt --year 2024

# Options
ar-scraper --universe us_russell_1000 --year 2024 --limit 10  # Test with 10
ar-scraper --universe us_russell_1000 --year 2024 --force     # Re-download
ar-scraper --universe us_russell_1000 --year 2024 --verbose   # Detailed output
ar-scraper --universe us_russell_1000 --year 2024 --quiet     # Minimal output
```

## Supported Universes

- `us_russell_1000` - US Russell 1000 (SEC EDGAR)
- `jp_topix` - Japan TOPIX (IRBank)
- `jp_topix_500` - Japan TOPIX 500 (IRBank)
- `tw_twse` - Taiwan TWSE (MOPS)
- `kr_kospi` - Korea KOSPI (DART) — requires `DART_API_KEY` env var
- `hk_hsci` - Hong Kong HSCI (HKEXnews)

## Output

Downloads are saved to `./reports/` (or `--output-dir`):

```
reports/
  AAPL_10-K_2024-02-01.pdf
  7974.T_SecReport_2024-06-26.pdf
  2330.TW_AnnualReport_2024-05-18.pdf
  005930.KS_BusinessReport_2024-03-15.zip
  0700.HK_AnnualReport_2024-04-18.pdf
  metadata.json
  errors.json
```

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```
