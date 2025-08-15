"""
EFI Analyser - Analysis tools for EFI corpora
"""

# Pipeline classes
from .pipeline import AbstractPipeline, LinearPipeline

# Application classes  
from .apps import WordOccurrenceApp

# Legacy exports (for backward compatibility)
from .pipeline import LinearPipeline as AnalysisPipeline
from .filters import TextContainsFilter
from .processors import CosineSimilarityProcessor
from .aggregators import KeywordPresenceAggregator, DocumentCountAggregator
from .types import PipelineResult, AppResult, AnalysisPipelineResult

# Re-export core types from efi_corpus for convenience
from efi_corpus.types import Document

__version__ = "0.1.0"
__all__ = [
    # New pipeline architecture
    "AbstractPipeline", 
    "LinearPipeline",
    
    # Applications
    "WordOccurrenceApp",
    
    # Legacy compatibility
    "AnalysisPipeline",  # Alias for LinearPipeline
    
    # Existing components
    "TextContainsFilter", 
    "CosineSimilarityProcessor",
    "KeywordPresenceAggregator",
    "DocumentCountAggregator",
    
    # Types
    "PipelineResult",
    "AppResult", 
    "AnalysisPipelineResult",
    "Document"
]
