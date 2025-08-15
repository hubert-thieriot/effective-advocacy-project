"""
Storage classes for corpus processing pipeline
"""

import json
import hashlib
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np

from .types import ChunkerSpec, EmbedderSpec, DocState, Chunker, Embedder
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
        chunks_path = self.layout.chunks_path(doc_id, chunker.key())
        
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
        emb_path = self.layout.emb_path(doc_id, chunker.key(), embedder.spec.key())
        
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


class DocStateStore:
    """Manages storage and retrieval of document processing state"""
    
    def __init__(self, layout: LocalFilesystemLayout):
        self.layout = layout

    def read(self, doc_id: str) -> Optional[DocState]:
        """Read document state from disk"""
        state_path = self.layout.doc_state_path(doc_id)
        if not state_path.exists():
            return None
        
        try:
            with open(state_path, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            return DocState(**state_data)
        except (json.JSONDecodeError, KeyError):
            return None

    def write(self, state: DocState) -> None:
        """Write document state to disk"""
        state_path = self.layout.doc_state_path(state.doc_id)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(state.__dict__, f, indent=2, ensure_ascii=False)

    def needs_rebuild(self, doc_id: str, fingerprint: str, chunker: ChunkerSpec, embedder: EmbedderSpec) -> bool:
        """
        Check if a document needs to be rebuilt
        
        Args:
            doc_id: Document identifier
            fingerprint: Current document fingerprint
            chunker: Chunking specification
            embedder: Embedding specification
            
        Returns:
            True if document needs rebuilding
        """
        state = self.read(doc_id)
        if state is None:
            return True
        
        return (
            state.fingerprint != fingerprint or
            state.chunker_key != chunker.key() or
            state.embedder_key != embedder.key()
        )
