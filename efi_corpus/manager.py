#!/usr/bin/env python3
"""
Generic Corpus Manager that delegates to specific builders
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from .rate_limiter import RateLimitConfig
from .utils import ensure_date

# Try to import YAML, but don't fail if not available
try:
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


class CorpusManager:
    """Generic corpus manager that delegates to specific builders"""
    
    def __init__(self, config_path: Path):
        """Initialize with a config file path"""
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        self.config = self._load_config(self.config_path)
        self._preprocess_config()

    def run(self) -> Dict[str, Any]:
        """Run the corpus building process using the specified builder"""
        config = self.config
        builder_type = str(self.config.get("builder", "")).lower()
        corpus_cfg = self._get_required(self.config, "corpus")
        base_dir = self._get_required(corpus_cfg, "base_dir")
        
        print(f"Creating {builder_type} builder for corpus: {base_dir}")
        
        # Create the appropriate builder
        builder = self._create_builder(builder_type, base_dir)
        
        # Let the builder handle its own config parsing and execution
        return builder.run_from_config(config)
    
    def _create_builder(self, builder_type: str, base_dir: str):
        """Create a builder instance based on type"""
        if builder_type == "mediacloud":
            from .builders.mediacloud import MediaCloudCorpusBuilder
            # For MediaCloud, we need to extract collection or source info from config
            params_cfg = self.config.get("parameters", {})
            collections = params_cfg.get("collections", [])
            sources = params_cfg.get("sources", [])
            
            if not collections and not sources:
                raise ValueError("MediaCloud builder requires either collections or sources in parameters")
            
            if collections and sources:
                raise ValueError("Cannot specify both collections and sources. Use one or the other.")
            
            if sources:
                # Use sources (media_ids) - names are optional
                source_ids = [source.get("id") for source in sources if source.get("id")]
                source_names = [source.get("name") for source in sources if source.get("name")] or None
                
                if not source_ids:
                    raise ValueError("MediaCloud builder requires at least one source with an id")
                
                return MediaCloudCorpusBuilder(
                    corpus_dir=base_dir,
                    source_ids=source_ids,
                    source_names=source_names
                )
            else:
                # Use collections (legacy behavior)
                # Use the first collection for now
                collection = collections[0]
                collection_id = collection.get("id")
                collection_name = collection.get("name", str(collection_id))
                
                return MediaCloudCorpusBuilder(
                    corpus_dir=base_dir,
                    collection_id=collection_id,
                    collection_name=collection_name
                )
        elif builder_type == "youtube":
            from .builders.youtube import YouTubeCorpusBuilder
            return YouTubeCorpusBuilder(corpus_dir=base_dir)
        elif builder_type == "manifesto":
            from .builders.manifesto import ManifestoCorpusBuilder
            # Extract countries from config
            params_cfg = self.config.get("parameters", {})
            countries = params_cfg.get("countries", [])
            return ManifestoCorpusBuilder(
                corpus_dir=base_dir,
                countries=countries
            )
        else:
            raise ValueError(f"Unsupported builder type: {builder_type}. Supported types: 'mediacloud', 'youtube', 'manifesto'")

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

    def _preprocess_config(self):
        """Preprocess config to handle special values like 'today' in date fields"""
        def _process_value(value):
            if isinstance(value, str):
                if value.strip().lower() == "today":
                    today_date = datetime.now().date().isoformat()
                    print(f"📅 Converting 'today' to: {today_date}")
                    return today_date
                return value
            elif isinstance(value, dict):
                return {k: _process_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [_process_value(item) for item in value]
            else:
                return value

        self.config = _process_value(self.config)

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
    def _compile_keywords_to_query(keywords_by_lang: Dict[str, list]) -> str:
        """
        Compile a keywords-by-language mapping into a single OR-combined query string.
        Example: {en: ["air pollution", "air quality"], hi: ["…", "…"]}
        becomes: (("air pollution" OR "air quality") OR ("…" OR "…"))
        """
        if not isinstance(keywords_by_lang, dict):
            raise ValueError("parameters.keywords must be a mapping of language->list[str]")
        group_exprs: list = []
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
    """Run a single config file"""
    manager = CorpusManager(Path(config_path))
    return manager.run()
