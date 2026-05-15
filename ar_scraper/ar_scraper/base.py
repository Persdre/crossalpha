"""Base types and dataclasses for ar-scraper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FilingMetadata:
    """Metadata for a downloaded filing."""

    symbol: str
    region: str
    company_name: str
    form_type: str
    filing_date: str
    filename: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScraperResult:
    """Result from running a scraper."""

    successful: list[FilingMetadata]
    errors: dict[str, str]
