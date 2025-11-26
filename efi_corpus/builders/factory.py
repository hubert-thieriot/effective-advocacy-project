"""
Factory for creating corpus builders from string identifiers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Any, Type

from .mediacloud import MediaCloudCorpusBuilder
from .manifesto import ManifestoCorpusBuilder
from ..rate_limiter import RateLimitConfig


# Registry mapping builder name -> class/constructor
_REGISTRY: Dict[str, Callable[..., Any]] = {
    "mediacloud": MediaCloudCorpusBuilder,
    "manifesto": ManifestoCorpusBuilder,
}


def register(builder_name: str, ctor: Callable[..., Any]) -> None:
    """Register a new builder under a name."""
    _REGISTRY[builder_name.lower()] = ctor


def create(builder_name: str, *, corpus_dir: Path, rate_limit_config: RateLimitConfig, **kwargs) -> Any:
    """
    Create a builder instance for the given name.
    Passes corpus_dir and rate_limit_config along with any builder-specific kwargs.
    """
    builder_key = builder_name.lower()
    ctor = _REGISTRY.get(builder_key)
    if ctor is None:
        raise ValueError(f"Unknown builder: {builder_name}")
    return ctor(corpus_dir=corpus_dir, rate_limit_config=rate_limit_config, **kwargs)


