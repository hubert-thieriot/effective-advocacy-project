"""Discourse analysis application package."""

from .config import load_config, DiscourseAnalysisConfig
from .pipeline import DiscourseAnalysisPipeline

__all__ = ["load_config", "DiscourseAnalysisConfig", "DiscourseAnalysisPipeline"]
