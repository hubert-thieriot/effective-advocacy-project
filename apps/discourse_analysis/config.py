"""Configuration helpers for discourse analysis."""

from __future__ import annotations

from pathlib import Path

from .config_models import (
    DiscourseAnalysisConfig,
    FramingConfig,
    FramingSchemaConfig,
    FramingAnnotationConfig,
    StanceConfig,
    StanceAnnotationConfig,
    ReportConfig,
    FilterConfig,
    ChunkingConfig,
    ClassifierSettings,
    ClassificationConfig,
)


def load_config(path: Path) -> DiscourseAnalysisConfig:
    """Load configuration from a YAML file."""
    return DiscourseAnalysisConfig.from_yaml(path)


__all__ = [
    "DiscourseAnalysisConfig",
    "FramingConfig",
    "FramingSchemaConfig",
    "FramingAnnotationConfig",
    "StanceConfig",
    "StanceAnnotationConfig",
    "ReportConfig",
    "FilterConfig",
    "ChunkingConfig",
    "ClassifierSettings",
    "ClassificationConfig",
    "load_config",
]
