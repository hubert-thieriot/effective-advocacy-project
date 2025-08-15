"""
EFI Corpus - Corpus management and processing tools
"""

from .types import (
    BuilderParams, DiscoveryItem, Document,
    ChunkerSpec, EmbedderSpec, DocState,
    Chunker, Embedder, AnnIndex
)
from .layout import LocalFilesystemLayout, ChunkerSpec, EmbedderSpec
from .corpus_handle import CorpusHandle
from .stores import ChunkStore, EmbeddingStore, DocStateStore
from .build_controller import BuildController
from .indexes import NumpyScanIndex

__all__ = [
    # Core types
    "BuilderParams", "DiscoveryItem", "Document",
    
    # New chunking and embedding types
    "ChunkerSpec", "EmbedderSpec", "DocState",
    "Chunker", "Embedder", "AnnIndex",
    
    # Layout and interface
    "LocalFilesystemLayout", "CorpusHandle",
    
    # Stores
    "ChunkStore", "EmbeddingStore", "DocStateStore",
    
    # Build controller
    "BuildController",
    
    # Indexes
    "NumpyScanIndex",
]
