"""
EFI Corpus - Local corpus building framework with MediaCloud integration
"""

from .corpus_handle import CorpusHandle
from .fetcher import Fetcher
from .text_extractor import TextExtractor
from .rate_limiter import RateLimiter, RateLimitConfig
from .types import BuilderParams, DiscoveryItem, Document
from .corpus_reader import CorpusReader

__version__ = "0.1.0"
__all__ = [
    "CorpusHandle", 
    "Fetcher", 
    "TextExtractor", 
    "RateLimiter", 
    "RateLimitConfig",
    "BuilderParams",
    "DiscoveryItem",
    "Document",
    "CorpusReader"
]
