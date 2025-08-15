"""
EFI Findings - Library for extracting and managing findings from various sources
"""

from .types import Finding, DocumentFindings, ExtractionConfig, StorageConfig
from .collectors import BaseURLCollector, CREAPublicationsCollector
from .extractors import BaseFindingExtractor, FindingExtractorFromText, FindingExtractorFromUrl, FindingsExtractor
from .storers import BaseStorer, JSONStorer
from .library_builder import LibraryBuilder
from .library_reader import LibraryReader
from .url_processor import URLProcessor
from .rate_limiter import DomainRateLimiter, DomainRateLimit, RateLimitedSession

__version__ = "0.1.0"

__all__ = [
    # Types
    "Finding",
    "DocumentFindings", 
    "ExtractionConfig",
    "StorageConfig",
    
    # Collectors
    "BaseURLCollector",
    "CREAPublicationsCollector",
    
    # Extractors
    "BaseFindingExtractor",
    "FindingExtractorFromText",
    "FindingExtractorFromUrl",
    "FindingsExtractor",
    
    # Storers
    "BaseStorer",
    "JSONStorer",
    
    # Core components
    "LibraryBuilder",
    "LibraryReader",
    "URLProcessor",
    
    # Rate limiting
    "DomainRateLimiter",
    "DomainRateLimit", 
    "RateLimitedSession",
]
