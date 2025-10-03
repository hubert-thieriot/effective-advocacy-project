"""Deprecated thin wrappers around the new top-level ``apps`` package.

Existing imports still resolve but projects should migrate to ``apps.<name>``.
"""

from apps.finding_document_matching import main as finding_document_matcher_main
from apps.claim_supporting import ClaimSupportingApp
from apps.word_occurrence import WordOccurrenceApp

__all__ = [
    "finding_document_matcher_main",
    "ClaimSupportingApp",
    "WordOccurrenceApp",
]
