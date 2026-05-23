"""Configuration constants for ar-scraper."""

from __future__ import annotations

# Universe name -> SymbolDict.csv region filter
UNIVERSES: dict[str, str] = {
    "us_russell_1000": "us_russell_1000",
    "jp_topix": "jp_topix",
    "jp_topix_500": "jp_topix_500",
    "tw_twse": "tw_twse",
    "kr_kospi": "kr_kospi",
    "hk_hsci": "hk_hsci",
    "hk_main": "hk_main",
}

# Symbol suffix -> region code
SYMBOL_SUFFIXES: dict[str, str] = {
    ".T": "jp",
    ".TW": "tw",
    ".KS": "kr",
    ".HK": "hk",
}

# Rate limits (preserved from original scrapers)
US_RATE_LIMIT = 3  # requests/second
JP_RATE_LIMIT = 10  # requests/second
TW_RATE_LIMIT = 1.0  # seconds between requests
TW_BATCH_SIZE = 8  # pause after this many requests
TW_BATCH_PAUSE = 15  # seconds to pause

# Retry settings
MAX_RETRIES = 3
RETRY_BACKOFF = [1.0, 2.0, 4.0]

# Validation
MIN_PDF_SIZE = 10_000  # bytes

# US-specific
SEC_USER_AGENT = "MS Capital Research research@example.com"
SEC_BASE_URL = "https://data.sec.gov"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"
DEFAULT_PDF_WORKERS = 10

# Japan-specific (IRBank)
IRBANK_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
}

# Taiwan-specific (MOPS)
MOPS_URL = "https://doc.twse.com.tw/server-java/t57sb01"
MOPS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
}
MOPS_RATE_LIMIT_MSG = "查詢過量"

# Korea-specific (DART)
DART_API_URL = "https://opendart.fss.or.kr/api"
DART_RATE_LIMIT = 0.2  # seconds between requests (5 req/s)
DART_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}

# Hong Kong-specific (HKEXnews)
HKEX_SEARCH_URL = "https://www1.hkexnews.hk/search/titleSearchServlet.do"
HKEX_STOCK_LIST_URL = "https://www1.hkexnews.hk/ncms/script/eds/activestock_sehk_e.json"
HKEX_BASE_URL = "https://www1.hkexnews.hk"
HK_RATE_LIMIT = 0.5  # seconds between requests (~2 req/s)
HK_BATCH_SIZE = 10
HK_BATCH_PAUSE = 5  # seconds
HKEX_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "X-Requested-With": "XMLHttpRequest",
    "Referer": "https://www1.hkexnews.hk/search/titlesearch.xhtml?lang=en",
}
# Annual Report tier codes
HKEX_T1_FINANCIAL = "40000"
HKEX_T2_ANNUAL_REPORT = "40100"


def detect_region_from_symbol(symbol: str) -> str:
    """Detect region from symbol suffix.

    Args:
        symbol: Stock symbol (e.g., "AAPL", "7974.T", "2330.TW")

    Returns:
        Region code: "us", "jp", "tw", "kr", or "hk"
    """
    symbol_upper = symbol.upper()
    for suffix, region in SYMBOL_SUFFIXES.items():
        if symbol_upper.endswith(suffix):
            return region
    return "us"  # Default to US for symbols without suffix
