"""
Storage classes for corpus processing pipeline
"""

import json
from typing import List
from pathlib import Path
import numpy as np

from efi_core.types import ChunkerSpec
from efi_core.protocols import Chunker, Embedder
from .layout import LocalFilesystemLayout


class ChunkStore:
    """Manages storage and retrieval of text chunks"""
    
    def __init__(self, layout: LocalFilesystemLayout):
        self.layout = layout

    def materialize(self, doc_id: str, text: str, chunker: ChunkerSpec, chunker_impl: Chunker) -> List[str]:
        """
        Get chunks for a document, creating them if they don't exist
        
        Args:
            doc_id: Document identifier
            text: Document text to chunk
            chunker: Chunking specification
            chunker_impl: Chunking implementation
            
        Returns:
            List of text chunks
        """
        chunks_path = self.layout.chunks_path(doc_id, chunker)
        
        # Check if chunks already exist
        if chunks_path.exists():
            # Fast path: read cached chunks
            chunks = []
            with open(chunks_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        chunk_data = json.loads(line)
                        chunks.append(chunk_data["text"])
            return chunks
        
        # Create chunks and cache them
        chunks_path.parent.mkdir(parents=True, exist_ok=True)
        chunks = chunker_impl.chunk(text)
        
        with open(chunks_path, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks):
                chunk_data = {"chunk_id": i, "text": chunk}
                f.write(json.dumps(chunk_data, ensure_ascii=False) + "\n")
        
        return chunks


class EmbeddingStore:
    """Manages storage and retrieval of text embeddings"""
    
    def __init__(self, layout: LocalFilesystemLayout):
        self.layout = layout

    def materialize(self, doc_id: str, chunks: List[str], chunker: ChunkerSpec, embedder: Embedder) -> np.ndarray:
        """
        Get embeddings for document chunks, creating them if they don't exist
        
        Args:
            doc_id: Document identifier
            chunks: List of text chunks
            chunker: Chunking specification
            embedder: Embedding implementation
            
        Returns:
            Array of embeddings with shape (n_chunks, embedding_dim)
        """
        emb_path = self.layout.emb_path(doc_id, chunker, embedder.spec)
        
        # Check if embeddings already exist
        if emb_path.exists():
            return np.load(emb_path)
        
        # Create embeddings and cache them
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        vectors = embedder.embed(chunks)
        
        # Convert to numpy array and save
        vectors_array = np.array(vectors)
        np.save(emb_path, vectors_array)
        
        return vectors_array


class LibraryEmbeddingStore:
    """Stores embeddings for findings (library items)."""

    def __init__(self, library_root: Path, embedder: Embedder):
        self.root = Path(library_root)
        self.embedder = embedder

    def emb_path(self, finding_id: str) -> Path:
        return self.root / "embeddings" / self.embedder.spec.key() / f"{finding_id}.npy"

    def materialize(self, finding_id: str, text: str) -> np.ndarray:
        p = self.emb_path(finding_id)
        if p.exists():
            return np.load(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        vecs = self.embedder.embed([text])
        arr = np.array(vecs)
        np.save(p, arr)
        return arr
