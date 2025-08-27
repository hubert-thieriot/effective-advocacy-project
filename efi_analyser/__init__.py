"""
EFI Analyser - Analysis tools for EFI corpora
"""

# Analysis pipeline components
from .pipeline.base import AbstractPipeline
from .pipeline.linear import LinearPipeline

# Filters
from .filters import TextContainsFilter, MetadataFilter, CompositeFilter, CREAFilter

# Processors
from .processors import CosineSimilarityProcessor, KeywordExtractorProcessor, TextStatisticsProcessor

# Aggregators
from .aggregators import KeywordPresenceAggregator, DocumentCountAggregator

# Chunkers and embedders
from .chunkers.sentence_chunker import SentenceChunker
from .embedders.sentence_transformer_embedder import SentenceTransformerEmbedder

# Retrieval system
from efi_core.retrieval.retriever import Retriever, SearchResult
from efi_core.retrieval.index_builder import IndexBuilder

__all__ = [
    # Pipeline
    "AbstractPipeline",
    "LinearPipeline",
    
    # Filters
    "TextContainsFilter",
    "MetadataFilter", 
    "CompositeFilter",
    "CREAFilter",
    
    # Processors
    "KeywordExtractorProcessor",
    "CosineSimilarityProcessor",
    "TextStatisticsProcessor",
    
    # Aggregators
    "KeywordPresenceAggregator",
    "DocumentCountAggregator",
    
    # Chunkers and embedders
    "SentenceChunker",
    "SentenceTransformerEmbedder",
    
    # Retrieval
    "Retriever",
    "SearchResult",
    "IndexBuilder"
]
