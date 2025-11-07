"""
BaseCorpusBuilder - Abstract base class for all corpus builders
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any

from ..types import BuilderParams
from ..corpus_handle import CorpusHandle
from ..fetcher import Fetcher


class BaseCorpusBuilder(ABC):
    """Abstract base class for all corpus builders"""

    def __init__(self, corpus_dir: Path, fetcher: Fetcher = None, cache_root: Path = None):
        self.corpus = CorpusHandle(corpus_dir, read_only=False)

        # Use provided fetcher or create default one
        if fetcher is not None:
            self.fetcher = fetcher
        else:
            if cache_root is None:
                cache_root = Path("cache")
            self.fetcher = Fetcher(cache_root)

    @abstractmethod
    def run(self, *, params: BuilderParams | None = None, override: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Run the corpus builder

        Args:
            params: Build parameters (if None, load from manifest)
            override: Override specific parameters

        Returns:
            Summary of the build process
        """
        pass

    def run_from_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the corpus builder using a config dictionary

        Args:
            config: Configuration dictionary loaded from YAML/JSON

        Returns:
            Summary of the build process
        """
        # Extract parameters from config
        params_cfg = config.get("parameters", {})

        # Create BuilderParams
        params = BuilderParams(
            keywords=params_cfg.get("keywords", []),
            date_from=params_cfg.get("date_from", ""),
            date_to=params_cfg.get("date_to", ""),
            skip_previously_failed=params_cfg.get("skip_previously_failed", False),
            extra=params_cfg
        )

        # Run the builder
        return self.run(params=params)

