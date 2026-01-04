"""
EFI Corpus - Corpus management and processing tools
"""

from .types import (
    BuilderParams, DiscoveryItem, Document,
)
from efi_core.types import ChunkerSpec, EmbedderSpec, DocState
from efi_core.protocols import Chunker, Embedder, AnnIndex
from efi_core.layout import CorpusLayout, LibraryLayout, WorkspaceLayout, EmbeddedCorpusLayout, EmbeddedLibraryLayout
from .corpus_handle import CorpusHandle
from efi_core.stores import ChunkStore, EmbeddingStore, DocStateStore
from .embedded import EmbeddedCorpus

__all__ = [
    # Core types
    "BuilderParams", "DiscoveryItem", "Document",
    
    # New chunking and embedding types
    "ChunkerSpec", "EmbedderSpec", "DocState",
    "Chunker", "Embedder", "AnnIndex",
    
    # Layout classes
    "CorpusLayout", "LibraryLayout", "WorkspaceLayout", "EmbeddedCorpusLayout", "EmbeddedLibraryLayout",
    
    # Interface
    "CorpusHandle",
    "EmbeddedCorpus",

    # Stores
    "ChunkStore", "EmbeddingStore", "DocStateStore",
]
