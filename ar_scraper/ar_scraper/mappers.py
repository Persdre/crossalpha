"""Symbol mappers for Japan, Taiwan, Korea, and Hong Kong markets."""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Optional

import polars as pl
import requests

from ar_scraper.config import DART_API_URL, DART_HEADERS


class EdinetMapper:
    """Maps yfinance symbols to EDINET codes for Japan."""

    def __init__(self, csv_path: Path):
        """Initialize mapper with EDINET code list CSV.

        Args:
            csv_path: Path to EdinetcodeDlInfo.csv (cp932 encoded)
        """
        self._mapping: dict[str, dict] = {}
        self._load_csv(csv_path)

    def _load_csv(self, csv_path: Path) -> None:
        """Load and parse the EDINET code list."""
        df = pl.read_csv(
            csv_path,
            encoding="cp932",
            skip_rows=1,
            infer_schema_length=10000,
        )

        for row in df.iter_rows():
            edinet_code = str(row[0]).strip() if row[0] else ""
            company_name = str(row[6]).strip() if len(row) > 6 and row[6] else ""
            sec_code = str(row[11]).strip() if len(row) > 11 and row[11] else ""
            listed_status = str(row[2]).strip() if len(row) > 2 and row[2] else ""

            if (
                sec_code
                and len(sec_code) == 5
                and sec_code.isdigit()
                and "Listed" in listed_status
            ):
                symbol = sec_code[:4] + ".T"
                self._mapping[symbol] = {
                    "edinet_code": edinet_code,
                    "sec_code": sec_code,
                    "company_name": company_name,
                }

    def get_edinet_code(self, symbol: str) -> Optional[str]:
        """Get EDINET code for a yfinance symbol."""
        info = self._mapping.get(symbol.upper())
        return info["edinet_code"] if info else None

    def get_info(self, symbol: str) -> Optional[dict]:
        """Get full info for a yfinance symbol."""
        return self._mapping.get(symbol.upper())

    def get_all_symbols(self) -> list[str]:
        """Get all mapped symbols."""
        return list(self._mapping.keys())

    def __len__(self) -> int:
        return len(self._mapping)


class TwseMapper:
    """Maps yfinance symbols to MOPS stock codes for Taiwan."""

    def __init__(self, symbol_dict_path: Path):
        """Initialize mapper with SymbolDict.csv.

        Args:
            symbol_dict_path: Path to SymbolDict.csv
        """
        self._mapping: dict[str, dict] = {}
        self._load_csv(symbol_dict_path)

    def _load_csv(self, csv_path: Path) -> None:
        """Load Taiwan symbols from SymbolDict.csv."""
        df = pl.read_csv(csv_path)
        tw_df = df.filter(pl.col("region") == "tw_twse")

        for row in tw_df.iter_rows(named=True):
            symbol = row.get("SYMBOL", "")
            if symbol and symbol.endswith(".TW"):
                stock_id = symbol.replace(".TW", "")
                self._mapping[symbol] = {
                    "stock_id": stock_id,
                    "company_name": row.get("company", ""),
                    "industry": row.get("industry", ""),
                }

    def get_stock_id(self, symbol: str) -> Optional[str]:
        """Get MOPS stock ID for a yfinance symbol."""
        info = self._mapping.get(symbol.upper())
        return info["stock_id"] if info else None

    def get_info(self, symbol: str) -> Optional[dict]:
        """Get full info for a yfinance symbol."""
        return self._mapping.get(symbol.upper())

    def get_all_symbols(self) -> list[str]:
        """Get all mapped Taiwan symbols."""
        return list(self._mapping.keys())

    def __len__(self) -> int:
        return len(self._mapping)


class DartMapper:
    """Maps yfinance .KS symbols to DART corp codes for Korea."""

    CORP_CODE_CACHE = "dart_corp_codes.xml"

    def __init__(self, api_key: str, cache_dir: Optional[Path] = None):
        """Initialize mapper by downloading or loading cached corp code list.

        Args:
            api_key: DART API key (crtfc_key)
            cache_dir: Directory to cache corp code XML (default: data dir)
        """
        self._mapping: dict[str, dict] = {}
        self._cache_dir = cache_dir or Path(__file__).parent / "data"
        self._api_key = api_key
        self._load_corp_codes()

    def _load_corp_codes(self) -> None:
        """Load corp codes from cache or download from DART."""
        cache_path = self._cache_dir / self.CORP_CODE_CACHE
        if cache_path.exists():
            xml_bytes = cache_path.read_bytes()
        else:
            xml_bytes = self._download_corp_codes()
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(xml_bytes)
        self._parse_corp_codes(xml_bytes)

    def _download_corp_codes(self) -> bytes:
        """Download corp code ZIP from DART and extract XML."""
        url = f"{DART_API_URL}/corpCode.xml"
        resp = requests.get(
            url,
            params={"crtfc_key": self._api_key},
            headers=DART_HEADERS,
            timeout=60,
        )
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            xml_name = zf.namelist()[0]
            return zf.read(xml_name)

    def _parse_corp_codes(self, xml_bytes: bytes) -> None:
        """Parse corp code XML and build symbol mapping."""
        root = ET.fromstring(xml_bytes)
        for item in root.iter("list"):
            corp_code = (item.findtext("corp_code") or "").strip()
            corp_name = (item.findtext("corp_name") or "").strip()
            stock_code = (item.findtext("stock_code") or "").strip()

            if stock_code and len(stock_code) == 6 and stock_code.isdigit():
                symbol = f"{stock_code}.KS"
                self._mapping[symbol] = {
                    "corp_code": corp_code,
                    "corp_name": corp_name,
                    "stock_code": stock_code,
                }

    def get_corp_code(self, symbol: str) -> Optional[str]:
        """Get DART corp code for a yfinance .KS symbol."""
        info = self._mapping.get(symbol.upper())
        return info["corp_code"] if info else None

    def get_info(self, symbol: str) -> Optional[dict]:
        """Get full info for a yfinance symbol."""
        return self._mapping.get(symbol.upper())

    def get_all_symbols(self) -> list[str]:
        """Get all mapped Korean symbols."""
        return list(self._mapping.keys())

    def __len__(self) -> int:
        return len(self._mapping)


class HkexMapper:
    """Maps yfinance .HK symbols to HKEXnews internal IDs."""

    STOCK_LIST_CACHE = "hkex_active_stocks.json"

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize mapper by downloading or loading cached stock list.

        Args:
            cache_dir: Directory to cache stock list JSON (default: data dir)
        """
        self._mapping: dict[str, dict] = {}
        self._cache_dir = cache_dir or Path(__file__).parent / "data"
        self._load_stock_list()

    def _load_stock_list(self) -> None:
        """Load stock list from cache or download from HKEXnews."""
        import json
        import time

        from ar_scraper.config import HKEX_STOCK_LIST_URL

        cache_path = self._cache_dir / self.STOCK_LIST_CACHE
        cache_fresh = (
            cache_path.exists()
            and (time.time() - cache_path.stat().st_mtime) <= 7 * 24 * 3600
        )
        if cache_fresh:
            data = json.loads(cache_path.read_text())
        else:
            data = self._download_stock_list()
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(data, ensure_ascii=False))
        self._parse_stock_list(data)

    def _download_stock_list(self) -> list[dict]:
        """Download active stock list from HKEXnews."""
        from ar_scraper.config import HKEX_STOCK_LIST_URL

        resp = requests.get(HKEX_STOCK_LIST_URL, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def _parse_stock_list(self, data: list[dict]) -> None:
        """Parse stock list and build symbol mapping."""
        for item in data:
            code = item.get("c", "").strip()
            internal_id = item.get("i")
            name = item.get("n", "").strip()
            if code and internal_id is not None and len(code) == 5:
                # Convert to yfinance format: 00700 -> 0700.HK
                # yfinance uses 4-digit codes (strip first zero from 5-digit HKEX code)
                yf_code = code[1:] if code.startswith("0") else code
                symbol = f"{yf_code}.HK"
                self._mapping[symbol.upper()] = {
                    "internal_id": str(internal_id),
                    "stock_code": code,
                    "company_name": name,
                }

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """Normalize a .HK symbol to 4-digit yfinance format.

        Accepts various formats: 700.HK, 0700.HK, 00700.HK -> 0700.HK
        """
        s = symbol.upper()
        if s.endswith(".HK"):
            num_part = s[:-3]
            # Pad to 4 digits to match yfinance convention
            num_part = num_part.zfill(4)
            return f"{num_part}.HK"
        return s

    def get_internal_id(self, symbol: str) -> Optional[str]:
        """Get HKEXnews internal ID for a yfinance .HK symbol."""
        info = self._mapping.get(self._normalize_symbol(symbol))
        return info["internal_id"] if info else None

    def get_stock_code(self, symbol: str) -> Optional[str]:
        """Get 5-digit HKEX stock code for a yfinance symbol."""
        info = self._mapping.get(self._normalize_symbol(symbol))
        return info["stock_code"] if info else None

    def get_info(self, symbol: str) -> Optional[dict]:
        """Get full info for a yfinance symbol."""
        return self._mapping.get(self._normalize_symbol(symbol))

    def get_all_symbols(self) -> list[str]:
        """Get all mapped HK symbols."""
        return list(self._mapping.keys())

    def __len__(self) -> int:
        return len(self._mapping)
