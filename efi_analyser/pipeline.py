"""
High-level pipeline wrapper (legacy alias for LinearPipeline)
"""

from .pipeline.linear import LinearPipeline as AnalysisPipeline
from efi_core.types import Document
from efi_data import Corpus

__all__ = ["AnalysisPipeline", "Document", "Corpus"]
