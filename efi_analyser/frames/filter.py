"""Document filtering for frame analysis workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd

from efi_core.protocols import Corpus


def extract_domain(url: Optional[str]) -> Optional[str]:
    """Extract base domain from URL, ignoring subdomains.

    Examples:
        https://www.bbc.co.uk/news -> bbc.co.uk
        https://kota.tribunnews.com/article -> tribunnews.com
    """
    if not url:
        return None
    parsed = urlparse(url)
    netloc = parsed.netloc or parsed.path
    if not netloc:
        return None
    domain = netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]

    parts = domain.split(".")
    if len(parts) >= 2:
        # Handle two-part TLDs like .co.uk, .co.id, .com.au
        if len(parts) >= 3 and parts[-2] in ("co", "com", "org", "net", "ac", "gov"):
            return ".".join(parts[-3:])
        return ".".join(parts[-2:])

    return domain or None


@dataclass
class DocumentFilter:
    """Document-level filtering configuration for aggregation.

    All filters are optional. Documents must pass ALL specified filters.
    """

    keywords: Optional[List[str]] = None  # Document must contain at least one keyword
    domain_whitelist: Optional[List[str]] = None  # Domain must be in whitelist
    date_from: Optional[str] = None  # published_at >= date_from
    date_windows: Optional[List[Tuple[str, str]]] = None  # published_at within any window

    def is_empty(self) -> bool:
        """Return True if no filters are configured."""
        return not any([
            self.keywords,
            self.domain_whitelist,
            self.date_from,
            self.date_windows,
        ])

    @classmethod
    def from_config(cls, config) -> "DocumentFilter":
        """Create DocumentFilter from a config object (FilterConfig or similar)."""
        if config is None:
            return cls()

        # Handle FilterConfig structure
        keywords = None
        domain_whitelist = None
        date_from = getattr(config, "date_from", None)
        date_windows = getattr(config, "date_windows", None)

        doc_filter = getattr(config, "document", None)
        if doc_filter:
            keywords = getattr(doc_filter, "keywords", None)
            domain_whitelist = getattr(doc_filter, "domain_whitelist", None)

        return cls(
            keywords=keywords,
            domain_whitelist=domain_whitelist,
            date_from=date_from,
            date_windows=date_windows,
        )

    def matches(self, doc_id: str, corpus: Corpus) -> bool:
        """Check if a document passes all configured filters.

        Args:
            doc_id: Local document ID within the corpus.
            corpus: Corpus handle to look up document metadata and text.

        Returns:
            True if the document passes all filters, False otherwise.
        """
        if self.is_empty():
            return True

        # Get index entry for metadata (url, published_at)
        index_entry = corpus.get_index_entry(doc_id)
        if index_entry is None:
            return False

        # Domain whitelist filter
        if self.domain_whitelist:
            url = index_entry.get("url")
            domain = extract_domain(url) if url else None
            if not domain or domain.lower() not in [d.lower() for d in self.domain_whitelist]:
                return False

        # Date filters
        published_at = index_entry.get("published_at")
        date = pd.to_datetime(published_at) if published_at else None

        if date is not None:
            # date_from filter
            if self.date_from:
                try:
                    date_from = pd.to_datetime(self.date_from)
                    if date < date_from:
                        return False
                except Exception:
                    pass

            # date_windows filter (document must be within at least one window)
            if self.date_windows:
                in_window = False
                for start, end in self.date_windows:
                    try:
                        window_start = pd.to_datetime(start)
                        window_end = pd.to_datetime(end)
                        if window_start <= date <= window_end:
                            in_window = True
                            break
                    except Exception:
                        continue
                if not in_window:
                    return False
        elif self.date_from or self.date_windows:
            # Document has no date but date filter is configured
            return False

        # Keyword filter (requires loading text)
        if self.keywords:
            try:
                text = corpus.get_text(doc_id).lower()
                title = str(index_entry.get("title", "")).lower()
                doc_text = f"{title} {text}"
                if not any(kw.lower() in doc_text for kw in self.keywords):
                    return False
            except FileNotFoundError:
                return False

        return True


__all__ = ["DocumentFilter", "extract_domain"]
