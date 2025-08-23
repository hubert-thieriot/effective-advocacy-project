"""
EFI Analyser - Pipeline implementations
"""

from .base import AbstractPipeline
from .linear import LinearPipeline

# Legacy alias for compatibility in tests
AnalysisPipeline = LinearPipeline

__all__ = ["AbstractPipeline", "LinearPipeline", "AnalysisPipeline"]
