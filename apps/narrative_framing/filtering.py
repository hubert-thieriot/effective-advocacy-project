"""Centralized filtering helpers for narrative framing workflows.

Applies a consistent set of rules across sampling, classification and aggregation:
- Trim text after any of the configured markers
- Exclude text when it matches any of the regex patterns
- Exclude text when any configured phrase appears at least N times
- Optional keyword gating (lowercased substring match)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Pattern, Sequence
import re


@dataclass
class FilterSpec:
    patterns: Optional[List[Pattern]] = None
    min_hits: Optional[Dict[str, int]] = None
    trim_markers: Optional[List[str]] = None
    keywords: Optional[List[str]] = None  # normalized, lower-cased


def _compile_patterns(patterns: Optional[Sequence[str]]) -> Optional[List[Pattern]]:
    if not patterns:
        return None
    compiled: List[Pattern] = []
    for p in patterns:
        try:
            compiled.append(re.compile(str(p), flags=re.IGNORECASE | re.MULTILINE))
        except re.error:
            continue
    return compiled or None


def _normalize_min_hits(mapping: Optional[Dict[str, int]]) -> Optional[Dict[str, int]]:
    if not mapping:
        return None
    result: Dict[str, int] = {}
    for k, v in mapping.items():
        key = str(k).strip().lower()
        try:
            val = int(v)
        except Exception:
            continue
        if key and val >= 1:
            result[key] = val
    return result or None


def _normalize_list(items: Optional[Sequence[str]]) -> Optional[List[str]]:
    if not items:
        return None
    out = [str(m).strip() for m in items if str(m).strip()]
    return out or None


def _compile_keyword_matcher(keyword: str) -> callable:
    """Compile a keyword into a matcher function.
    
    Matching modes (specified in config):
    - Default: substring match (case-insensitive) - matches anywhere in text
    - 'word:keyword' - word boundary match (only matches whole words)
    - 'regex:pattern' - full regex pattern matching
    
    Examples:
    - 'veget' -> matches "végétarien", "végétalien", etc. (substring)
    - 'word:lov' -> only matches standalone "lov", not "djelovanje" (word boundary)
    - 'regex:veget|végét' -> custom regex pattern
    
    Args:
        keyword: The keyword string (may include 'word:' or 'regex:' prefix)
        
    Returns:
        A function that takes text and returns True if it matches
    """
    keyword = keyword.strip()
    if not keyword:
        return lambda text: False
    
    # Check if it's a regex pattern (starts with 'regex:')
    if keyword.startswith('regex:'):
        pattern_str = keyword[6:].strip()
        try:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            return lambda text: pattern.search(text) is not None
        except re.error:
            # Fallback to substring if regex is invalid
            return lambda text: pattern_str.lower() in text.lower()
    
    # Check if it's a word boundary match (starts with 'word:')
    if keyword.startswith('word:'):
        keyword_str = keyword[5:].strip()
        # Escape special regex characters and add word boundaries
        escaped = re.escape(keyword_str)
        pattern = re.compile(r'\b' + escaped + r'\b', re.IGNORECASE)
        return lambda text: pattern.search(text) is not None
    
    # Default: substring matching (case-insensitive)
    keyword_lower = keyword.lower()
    return lambda text: keyword_lower in text.lower()


def _matches_any_keyword(text: str, keywords: List[str]) -> bool:
    """Check if text matches any of the keywords using appropriate matching strategy.
    
    Uses _compile_keyword_matcher to handle regex patterns and word boundaries.
    """
    matchers = [_compile_keyword_matcher(kw) for kw in keywords]
    return any(matcher(text) for matcher in matchers)


def _normalize_keywords(keywords: Optional[Sequence[str]]) -> Optional[List[str]]:
    """Normalize keywords (preserve original format, just strip whitespace).
    
    Note: Keywords with 'word:' or 'regex:' prefixes are preserved as-is.
    Other keywords are lowercased for consistent matching.
    """
    if not keywords:
        return None
    normalized = []
    for kw in keywords:
        if not kw or not kw.strip():
            continue
        kw_stripped = kw.strip()
        # Preserve prefixes (word:, regex:) as-is, lowercase others
        if kw_stripped.startswith('word:') or kw_stripped.startswith('regex:'):
            normalized.append(kw_stripped)
        else:
            normalized.append(kw_stripped.lower())
    return normalized or None


def make_filter_spec(
    *,
    exclude_regex: Optional[Sequence[str]] = None,
    exclude_min_hits: Optional[Dict[str, int]] = None,
    trim_after_markers: Optional[Sequence[str]] = None,
    keywords: Optional[Sequence[str]] = None,
) -> FilterSpec:
    return FilterSpec(
        patterns=_compile_patterns(exclude_regex),
        min_hits=_normalize_min_hits(exclude_min_hits),
        trim_markers=_normalize_list(trim_after_markers),
        keywords=_normalize_keywords(keywords),
    )


def filter_text(text: str, spec: FilterSpec) -> Optional[str]:
    """Return cleaned text or None if excluded by spec."""
    if text is None:
        return None
    t = str(text).strip()
    if not t:
        return None
    # Trim tail after markers
    if spec.trim_markers:
        lowered = t.lower()
        cut = len(t)
        for m in spec.trim_markers:
            idx = lowered.find(m.lower())
            if idx != -1:
                cut = min(cut, idx)
        if cut < len(t):
            t = t[:cut].strip()
    if not t:
        return None
    # Exclude by regex patterns
    if spec.patterns and any(p.search(t) for p in spec.patterns):
        return None
    # Exclude by min_hits thresholds
    if spec.min_hits:
        lowered = t.lower()
        for phrase, threshold in spec.min_hits.items():
            if lowered.count(phrase) >= threshold:
                return None
    return t if t else None


def filter_chunks(chunks: Sequence[Dict[str, object]], spec: FilterSpec) -> List[Dict[str, object]]:
    """Return chunks with cleaned text, excluding those removed by spec."""
    out: List[Dict[str, object]] = []
    for ch in chunks or []:
        if not isinstance(ch, dict):
            continue
        cleaned = filter_text(str(ch.get("text", "")), spec)
        if cleaned is None:
            continue
        ch2 = dict(ch)
        ch2["text"] = cleaned
        out.append(ch2)
    return out


def any_chunk_matches_keywords(chunks: Sequence[Dict[str, object]], spec: FilterSpec) -> bool:
    """Return True if any chunk contains any keyword using regex-aware matching."""
    if not spec.keywords:
        return True
    for ch in chunks or []:
        if not isinstance(ch, dict):
            continue
        text = str(ch.get("text", ""))
        if _matches_any_keyword(text, spec.keywords):
            return True
    return False


@dataclass
class Filter:
    """Convenience wrapper for the standard narrative framing filters."""

    exclude_regex: Optional[Sequence[str]] = None
    exclude_min_hits: Optional[Dict[str, int]] = None
    trim_after_markers: Optional[Sequence[str]] = None
    keywords: Optional[Sequence[str]] = None

    def to_spec(self) -> FilterSpec:
        """Build a :class:`FilterSpec` for use with filter_text/filter_chunks."""
        return make_filter_spec(
            exclude_regex=self.exclude_regex,
            exclude_min_hits=self.exclude_min_hits,
            trim_after_markers=self.trim_after_markers,
            keywords=self.keywords,
        )

    def sampler_kwargs(self, domain_whitelist: Optional[Sequence[str]] = None) -> Dict[str, object]:
        """Return kwargs suitable for constructing a SamplerConfig."""
        kwargs = {
            "keywords": self.keywords,
            "exclude_regex": self.exclude_regex,
            "exclude_min_hits": self.exclude_min_hits,
            "trim_after_markers": self.trim_after_markers,
        }
        if domain_whitelist is not None:
            kwargs["domain_whitelist"] = domain_whitelist
        return kwargs

