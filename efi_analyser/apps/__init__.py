"""
Application interfaces for different analysis workflows.
"""

from .finding_document_matcher import main as finding_document_matcher_main
from .claim_supporting import ClaimSupportingApp
from .word_occurrence import WordOccurrenceApp

__all__ = [
    "finding_document_matcher_main",
    "ClaimSupportingApp",
    "WordOccurrenceApp"
]
