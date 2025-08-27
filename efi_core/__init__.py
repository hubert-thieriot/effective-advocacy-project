"""
Core shared types and protocols for EFI projects.

This module centralizes domain types (e.g., Document, Finding) and
algorithmic protocols (e.g., Chunker, Embedder, AnnIndex) plus model
specifications (ChunkerSpec, EmbedderSpec).
"""

# Core EFI types and protocols
from .types import (
    Document,
    Finding,
    Chunk,
    ChunkerSpec,
    EmbedderSpec,
    DocState,
    FindingState
)

from .protocols import (
    Chunker,
    Embedder,
    AnnIndex,
    Corpus,
    Library
)

# Layout management
from .layout import (
    BaseLayout,
    CorpusLayout,
    LibraryLayout,
    WorkspaceLayout,
    EmbeddedCorpusLayout,
    EmbeddedLibraryLayout
)

# Data stores
from .stores import (
    ChunkStore,
    EmbeddingStore,
    DocStateStore
)

# Retrieval system
from .retrieval.retriever import (
    Retriever,
    SearchResult
)

from .retrieval.index_builder import (
    IndexBuilder
)

from .stores.indexes import (
    IndexStore
)

__all__ = [
    # Types
    "Document",
    "Finding", 
    "ChunkerSpec",
    "EmbedderSpec",
    "DocState",
    "FindingState",
    
    # Protocols
    "Chunker",
    "Embedder",
    "AnnIndex",
    "Corpus",
    "Library",
    
    # Layouts
    "BaseLayout",
    "CorpusLayout",
    "LibraryLayout", 
    "WorkspaceLayout",
    "EmbeddedCorpusLayout",
    "EmbeddedLibraryLayout",
    
    # Stores
    "ChunkStore",
    "EmbeddingStore",
    "DocStateStore",
    
    # Retrieval
    "Retriever",
    "SearchResult",
    "IndexBuilder",
    "IndexStore"
]


