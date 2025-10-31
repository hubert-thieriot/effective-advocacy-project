"""
Configuration helpers for the word count (theme keyword) application.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class Theme:
    """A theme groups multiple sector-specific keywords/phrases."""
    id: str
    name: str
    keywords: List[str]
    # Optional raw regex patterns for combined matching (compiled as-is)
    patterns: List[str] = field(default_factory=list)


@dataclass
class WordCountConfig:
    """Typed configuration for word-count by themed keywords."""

    corpus: str
    workspace_root: Path = Path("workspace")
    results_dir: Optional[Path] = None

    # Matching options
    case_sensitive: bool = False
    whole_word_only: bool = True
    allow_hyphenation: bool = True

    # Document aggregation options
    # Minimum number of distinct theme keywords in a document
    # to count the theme as present (weight = 1).
    min_words: int = 1

    # Optional limits
    doc_limit: Optional[int] = None
    date_from: Optional[str] = None  # YYYY-MM-DD
    date_to: Optional[str] = None    # YYYY-MM-DD

    # Themes
    themes: List[Theme] = field(default_factory=list)

    def all_keywords(self) -> List[str]:
        uniq: Dict[str, None] = {}
        for t in self.themes:
            for k in t.keywords:
                s = str(k).strip()
                if s:
                    uniq[s] = None
        return list(uniq.keys())

    def all_patterns(self) -> List[str]:
        pats: List[str] = []
        for t in self.themes:
            for p in (t.patterns or []):
                s = str(p).strip()
                if s:
                    pats.append(s)
        return pats


def _as_path(value: Optional[Any]) -> Optional[Path]:
    if value in (None, ""):
        return None
    return Path(value)


def load_config(path: Path) -> WordCountConfig:
    """Load configuration from a YAML file."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    data: Dict[str, Any] = payload or {}

    if "corpus" not in data:
        raise ValueError("config missing required field 'corpus'")

    cfg = WordCountConfig(corpus=str(data["corpus"]))
    if "workspace_root" in data and data["workspace_root"]:
        cfg.workspace_root = Path(str(data["workspace_root"]))
    if "results_dir" in data and data["results_dir"]:
        cfg.results_dir = Path(str(data["results_dir"]))

    # Matching
    cfg.case_sensitive = bool(data.get("case_sensitive", cfg.case_sensitive))
    cfg.whole_word_only = bool(data.get("whole_word_only", cfg.whole_word_only))
    cfg.allow_hyphenation = bool(data.get("allow_hyphenation", cfg.allow_hyphenation))
    # Aggregation
    try:
        mw = int(data.get("min_words", cfg.min_words))
        cfg.min_words = mw if mw >= 1 else 1
    except Exception:
        cfg.min_words = 1

    # Limits
    if data.get("doc_limit") is not None:
        try:
            dl = int(data["doc_limit"])  # type: ignore[arg-type]
            cfg.doc_limit = dl if dl > 0 else None
        except Exception:
            cfg.doc_limit = None
    if data.get("date_from"):
        cfg.date_from = str(data["date_from"])[:10]
    if data.get("date_to"):
        cfg.date_to = str(data["date_to"])[:10]

    # Themes
    raw_themes = data.get("themes") or []
    themes: List[Theme] = []
    for item in raw_themes:
        tid = str(item.get("id") or item.get("name") or "").strip()
        name = str(item.get("name") or tid)
        kws = [str(x).strip() for x in (item.get("keywords") or []) if str(x).strip()]
        pats = [str(x).strip() for x in (item.get("patterns") or []) if str(x).strip()]
        if not tid or not kws:
            continue
        themes.append(Theme(id=tid, name=name, keywords=kws, patterns=pats))
    if not themes:
        raise ValueError("config has no valid 'themes' entries")
    cfg.themes = themes

    return cfg
