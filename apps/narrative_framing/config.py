"""Configuration helpers for the narrative framing application.

This module provides backward compatibility wrappers around the new Pydantic-based
configuration system in config_models.py.
"""

from __future__ import annotations

from pathlib import Path

# Re-export all Pydantic models for backward compatibility
from .config_models import (
    NarrativeFramingConfig,
    FilterConfig,
    DocumentFilter,
    ChunkFilter,
    ChunkingConfig,
    InductionConfig,
    AnnotationConfig,
    ClassificationConfig,
    ClassifierSettings,
    ReportConfig,
    PlotSettings,
    CustomPlotSettings,
)

# Legacy constants (kept for backward compatibility)
DEFAULT_CORPORA_ROOT = Path("corpora")
DEFAULT_WORKSPACE_ROOT = Path("workspace")
DEFAULT_INDUCTION_SAMPLE = 100
DEFAULT_APPLICATION_SAMPLE = 1000
DEFAULT_SEED = 42
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_INDUCTION_MODEL = "gpt-4o"
DEFAULT_APPLICATION_MODEL = "gpt-4o-mini"
DEFAULT_TARGET_WORDS = 200
DEFAULT_APPLICATION_BATCH = 8
DEFAULT_APPLICATION_TOP_K = 3


def load_config(path: Path) -> NarrativeFramingConfig:
    """Load configuration from a YAML file.

    This is a thin wrapper around NarrativeFramingConfig.from_yaml() for
    backward compatibility with existing code.

    Args:
        path: Path to the YAML configuration file

    Returns:
        Loaded and validated configuration object
    """
    return NarrativeFramingConfig.from_yaml(path)


__all__ = [
    # Main config class
    "NarrativeFramingConfig",
    # Filter configs
    "FilterConfig",
    "DocumentFilter",
    "ChunkFilter",
    # Sub-configs
    "ChunkingConfig",
    "InductionConfig",
    "AnnotationConfig",
    "ClassificationConfig",
    "ClassifierSettings",
    # Report configs
    "ReportConfig",
    "PlotSettings",
    "CustomPlotSettings",
    # Main loader function
    "load_config",
]
