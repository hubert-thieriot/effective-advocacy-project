"""
Type definitions for the analysis system
"""

from typing import Protocol, Dict, Any
from dataclasses import dataclass
from efi_corpus.types import Document


@dataclass
class AnalysisResult:
    """Result of analyzing a single document"""
    doc_id: str
    url: str
    passed_filters: bool
    filter_results: Dict[str, bool]
    processing_results: Dict[str, Any]
    meta: Dict[str, Any]


class Filter(Protocol):
    """Protocol for document filters"""
    def apply(self, document: Document) -> bool:
        """Apply filter to document, return True if document passes"""
        ...


class Processor(Protocol):
    """Protocol for document processors"""
    def process(self, document: Document) -> Dict[str, Any]:
        """Process document and return results"""
        ...
