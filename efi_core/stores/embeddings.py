from typing import List, Protocol, Optional
import numpy as np
from pathlib import Path
from efi_core.types import ChunkerSpec, EmbedderSpec
from efi_core.protocols import Embedder


class EmbeddingStorageLayout(Protocol):
    """Protocol for layouts that support embedding storage"""
    def emb_path(self, item_id: str, chunker: ChunkerSpec, embedder: EmbedderSpec) -> Path: ...


class EmbeddingStore:
    """Generic embedding store that works with any layout supporting embedding storage"""
    
    def __init__(self, layout: EmbeddingStorageLayout):
        self.layout = layout

    def read(self, item_id: str, chunker: ChunkerSpec, embedder: EmbedderSpec) -> Optional[np.ndarray]:
        """
        Read existing embeddings for an item.
        
        Args:
            item_id: Unique identifier for the item
            chunker: Chunker specification used to create the chunks
            embedder: Embedder specification
            
        Returns:
            Array of embeddings if they exist, None otherwise
        """
        p = self.layout.emb_path(item_id, chunker, embedder)
        if not p.exists():
            return None
        return np.load(p)

    def materialize(self, item_id: str, chunks: List[str], chunker: ChunkerSpec, embedder: Embedder) -> np.ndarray:
        """
        Get or create embeddings for text chunks
        
        Args:
            item_id: Unique identifier for the item being embedded
            chunks: List of text chunks to embed
            chunker: Chunker specification used to create the chunks
            embedder: Embedder implementation
            
        Returns:
            Array of embeddings with shape (len(chunks), embedding_dim)
        """
        p = self.layout.emb_path(item_id, chunker, embedder.spec)
        if p.exists():
            return np.load(p)
        
        p.parent.mkdir(parents=True, exist_ok=True)
        vecs = embedder.embed(chunks)
        arr = np.array(vecs)
        np.save(p, arr)
        return arr
    
    def has_embeddings(self, item_id: str, chunker: ChunkerSpec, embedder: EmbedderSpec) -> bool:
        """
        Check if embeddings exist for a given item
        
        Args:
            item_id: Unique identifier for the item
            chunker: Chunker specification used to create the chunks
            embedder: Embedder specification
            
        Returns:
            True if embeddings exist, False otherwise
        """
        p = self.layout.emb_path(item_id, chunker, embedder)
        return p.exists()


