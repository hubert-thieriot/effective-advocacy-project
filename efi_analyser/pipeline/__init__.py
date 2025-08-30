"""
Pipeline implementations for different analysis workflows.
"""

from .base import AbstractPipeline
from .linear import LinearPipeline
from .claim_supporting import ClaimSupportingPipeline
from .finding_document_matching import DocumentMatchingPipeline
from .word_occurrence import WordOccurrencePipeline

__all__ = [
    "AbstractPipeline",
    "LinearPipeline",
    "ClaimSupportingPipeline",
    "DocumentMatchingPipeline",
    "WordOccurrencePipeline"
]
