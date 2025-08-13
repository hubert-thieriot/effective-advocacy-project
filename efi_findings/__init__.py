"""
EFI Findings - Extract and store key findings from documents using LLM
"""

from .types import Finding, DocumentFindings, ExtractionConfig, StorageConfig
from .url_processor import URLProcessor
from .library_builder import LibraryBuilder
from .collectors import BaseURLCollector, CREAPublicationsCollector
from .extractors import BaseFindingExtractor, FindingExtractorFromText, FindingExtractorFromUrl
from .storers import BaseStorer, JSONStorer

__version__ = "0.1.0"
__all__ = [
    'Finding',
    'DocumentFindings', 
    'ExtractionConfig',
    'StorageConfig',
    'URLProcessor',
    'LibraryBuilder',
    'BaseURLCollector',
    'CREAPublicationsCollector',
    'BaseFindingExtractor',
    'FindingExtractorFromText',
    'FindingExtractorFromUrl',
    'BaseStorer',
    'JSONStorer'
]
