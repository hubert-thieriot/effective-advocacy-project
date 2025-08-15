"""
EFI Analyser - Pipeline implementations
"""

from .base import AbstractPipeline
from .linear import LinearPipeline

__all__ = ["AbstractPipeline", "LinearPipeline"]
