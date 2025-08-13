"""
EFI Corpus - Local corpus building framework with MediaCloud integration
"""

__version__ = "0.1.0"

from .corpus_handle import CorpusHandle
from .fetcher import Fetcher
from .builders.base import BaseCorpusBuilder
from .builders.mediacloud import MediaCloudCorpusBuilder
from .text_extractor import TextExtractor

__all__ = ["CorpusHandle", "Fetcher", "BaseCorpusBuilder", "MediaCloudCorpusBuilder", "TextExtractor"]
