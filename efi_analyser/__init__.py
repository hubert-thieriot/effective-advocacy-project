"""
EFI Analyser - Analysis tools for EFI corpora
"""

from .pipeline import AnalysisPipeline
from .filters import TextContainsFilter
from .processors import CosineSimilarityProcessor
from .aggregators import KeywordPresenceAggregator, DocumentCountAggregator
from .types import AnalysisPipelineResult

# Re-export core types from efi_corpus for convenience
from efi_corpus.types import Document

__version__ = "0.1.0"
__all__ = [
    "AnalysisPipeline", 
    "TextContainsFilter", 
    "CosineSimilarityProcessor",
    "KeywordPresenceAggregator",
    "DocumentCountAggregator",
    "AnalysisPipelineResult",
    "Document"
]
