"""
EFI Analyser - Analysis tools for EFI corpora
"""

from .pipeline import AnalysisPipeline
from .filters import TextContainsFilter
from .processors import CosineSimilarityProcessor

# Re-export core types from efi_corpus for convenience
from efi_corpus.types import Document

__version__ = "0.1.0"
__all__ = [
    "AnalysisPipeline", 
    "TextContainsFilter", 
    "CosineSimilarityProcessor",
    "Document"
]
