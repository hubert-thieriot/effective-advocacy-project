"""
CorpusManager - Orchestrates building corpora from per-country config files.

Config format (YAML or JSON):

builder: mediacloud

corpus:
  base_dir: corpora/air_quality

parameters:
  # Either provide raw queries (preferred for full control)...
  # queries:
  #   - '("air pollution" OR "air quality")'
  #   - '("umoya ongcolile" OR "ikhwalithi yomoya")'
  # ...or structured keywords by language (manager compiles to a query):
  # keywords:
  #   en: ["air pollution", "air quality"]
  #   zu: ["umoya ongcolile", "ikhwalithi yomoya"]
  collections:
    - id: 34412238
      name: South Africa National
  date_from: 2025-07-01
  date_to: today
  rate_limit:
    requests_per_minute: 3
    min_interval: 20.0
    enabled: true

extra_meta:
  country_iso3: ZAF
  comment: National air quality corpus for South Africa
  topic: air_quality
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional
import json

try:
    import yaml  # type: ignore
    _YAML_AVAILABLE = True
except Exception:
    _YAML_AVAILABLE = False

from .builders.factory import create as create_builder
from .types import BuilderParams
from .rate_limiter import RateLimitConfig
from .utils import ensure_date


class CorpusManager:
    """Manager that builds corpora based on a single country/topic config file."""

    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        self.config = self._load_config(self.config_path)

    def run(self) -> Dict[str, Any]:
        builder_type = str(self.config.get("builder", "")).lower()

        corpus_cfg = self._get_required(self.config, "corpus")
        base_dir = self._get_required(corpus_cfg, "base_dir")

        params_cfg = self._get_required(self.config, "parameters")
        collections = self._get_required(params_cfg, "collections")
        if not isinstance(collections, list) or not collections:
            raise ValueError("parameters.collections must be a non-empty list")

        # Resolve dates (allow "today") and convert to ISO strings
        date_from_str = self._resolve_date_string(params_cfg.get("date_from"))
        date_to_str = self._resolve_date_string(params_cfg.get("date_to", "today"))

        # Build query strings
        queries: List[str] = []
        if "queries" in params_cfg and params_cfg["queries"]:
            if not isinstance(params_cfg["queries"], list):
                raise ValueError("parameters.queries must be a list of strings")
            queries = [str(q).strip() for q in params_cfg["queries"] if str(q).strip()]
        elif "keywords" in params_cfg and params_cfg["keywords"]:
            compiled = self._compile_keywords_to_query(params_cfg["keywords"])  # single combined query
            queries = [compiled]
        else:
            raise ValueError("Provide either parameters.queries or parameters.keywords")

        # Optional rate limit
        rl_cfg = params_cfg.get("rate_limit") or {}
        rate_limit = RateLimitConfig(
            requests_per_minute=int(rl_cfg.get("requests_per_minute", RateLimitConfig.requests_per_minute)),
            min_interval=float(rl_cfg.get("min_interval", RateLimitConfig.min_interval)),
            burst_size=int(rl_cfg.get("burst_size", RateLimitConfig.burst_size)),
            enabled=bool(rl_cfg.get("enabled", RateLimitConfig.enabled)),
        )

        # Extra metadata
        extra_meta: Dict[str, Any] = dict(self.config.get("extra_meta") or {})
        
        # Use base_dir directly as corpus directory
        corpus_dir = Path(base_dir)

        summary = {"collections": [], "total_runs": 0}

        for collection in collections:
            collection_id = collection.get("id")
            collection_name = collection.get("name") or str(collection_id)
            if collection_id is None:
                raise ValueError("Each collection must include an 'id'")

            builder = create_builder(
                builder_type,
                corpus_dir=corpus_dir,
                rate_limit_config=rate_limit,
                collection_id=collection_id,
                collection_name=collection_name,
            )

            runs_for_collection = 0
            for query in queries:
                # Build params: pass query as a single keyword to preserve the raw query string
                params = BuilderParams(
                    keywords=[query],
                    date_from=date_from_str,
                    date_to=date_to_str,
                    extra={
                        **extra_meta,
                        "collection_id": collection_id,
                        "collection_name": collection_name,
                    }
                )

                print(
                    f"Running builder for {base_dir} | collection {collection_name} ({collection_id}) "
                    f"| query: {query[:80]}{'…' if len(query)>80 else ''} | dates: {date_from_str}→{date_to_str}"
                )
                builder.run(params=params)
                runs_for_collection += 1

            summary["collections"].append({
                "collection_id": collection_id,
                "collection_name": collection_name,
                "runs": runs_for_collection,
            })
            summary["total_runs"] += runs_for_collection

        print(f"Completed manager run: {summary}")
        return summary

    # ------------- helpers -------------

    @staticmethod
    def _load_config(path: Path) -> Dict[str, Any]:
        text = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            if not _YAML_AVAILABLE:
                raise RuntimeError("PyYAML is required to read YAML configs")
            return yaml.safe_load(text) or {}
        if suffix == ".json":
            return json.loads(text)
        # Try YAML first if available, else JSON
        if _YAML_AVAILABLE:
            return yaml.safe_load(text) or {}
        return json.loads(text)

    @staticmethod
    def _get_required(obj: Dict[str, Any], key: str) -> Any:
        if key not in obj or obj[key] is None:
            raise ValueError(f"Missing required config key: {key}")
        return obj[key]

    @staticmethod
    def _resolve_date_string(value: Optional[str]) -> str:
        if value is None:
            raise ValueError("date value is required")
        v = str(value).strip().lower()
        if v == "today":
            from datetime import datetime
            return datetime.now().date().isoformat()
        # ensure_date returns a date; convert to ISO string
        return ensure_date(value).isoformat()

    @staticmethod
    def _compile_keywords_to_query(keywords_by_lang: Dict[str, List[str]]) -> str:
        """
        Compile a keywords-by-language mapping into a single OR-combined query string.
        Example: {en: ["air pollution", "air quality"], hi: ["…", "…"]}
        becomes: (("air pollution" OR "air quality") OR ("…" OR "…"))
        """
        if not isinstance(keywords_by_lang, dict):
            raise ValueError("parameters.keywords must be a mapping of language->list[str]")
        group_exprs: List[str] = []
        for _, words in keywords_by_lang.items():
            if not words:
                continue
            terms = [f'"{str(w).strip()}"' for w in words if str(w).strip()]
            if not terms:
                continue
            group_exprs.append(f"({' OR '.join(terms)})")
        if not group_exprs:
            raise ValueError("parameters.keywords produced an empty query")
        return f"({' OR '.join(group_exprs)})"


def run_config(config_path: str | Path) -> Dict[str, Any]:
    manager = CorpusManager(Path(config_path))
    return manager.run()


