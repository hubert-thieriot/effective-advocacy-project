"""
Report generators for different analysis results.
"""

from .base import BaseReportGenerator
from .document_matching import DocumentMatchingReportGenerator
from .claim_supporting import ClaimSupportingReportGenerator
from .word_occurrence import WordOccurrenceReportGenerator

__all__ = [
    "BaseReportGenerator",
    "DocumentMatchingReportGenerator",
    "ClaimSupportingReportGenerator",
    "WordOccurrenceReportGenerator"
]
