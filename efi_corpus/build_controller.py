"""
Main controller for building embedded corpora
"""

import time
from typing import List, Optional
import numpy as np
from pathlib import Path

from .types import ChunkerSpec, EmbedderSpec, DocState, Chunker, Embedder, AnnIndex
from .corpus_handle import CorpusHandle
from .stores import ChunkStore, EmbeddingStore, DocStateStore
from .indexes import NumpyScanIndex


class BuildController:
    """
    Orchestrates the corpus processing pipeline:
    1. Reads documents from corpus
    2. Chunks text based on specification
    3. Generates embeddings for chunks
    4. Builds search index
    5. Tracks state for incremental rebuilds
    """
    
    def __init__(self, corpus_path: str, workspace_path: str):
        self.corpus = CorpusHandle(corpus_path, read_only=True)
        self.workspace_path = Path(workspace_path)
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize stores
        self.chunk_store = ChunkStore(self.workspace_path)
        self.embedding_store = EmbeddingStore(self.workspace_path)
        self.doc_state_store = DocStateStore(self.workspace_path)
        
        # Initialize index
        self.index = None

    def build(self, chunker: ChunkerSpec, embedder: EmbedderSpec, 
              chunker_impl: Chunker, embedder_impl: Embedder,
              index_impl: Optional[AnnIndex] = None) -> AnnIndex:
        """
        Build the embedded corpus
        
        Args:
            chunker: Chunking specification
            embedder: Embedding specification  
            chunker_impl: Chunking implementation
            embedder_impl: Embedding implementation
            index_impl: Index implementation (optional, creates NumpyScanIndex if None)
            
        Returns:
            Built search index
        """
        print(f"Building embedded corpus with {self.corpus.get_document_count()} documents...")
        
        start_time = time.time()
        
        # Initialize index if not provided
        if index_impl is None:
            index_impl = NumpyScanIndex()
        
        # Process each document
        for i, doc in enumerate(self.corpus.read_documents()):
            if i % 100 == 0:
                print(f"Processing document {i+1}/{self.corpus.get_document_count()}")
            
            # Get or create chunks
            chunks = self.chunk_store.materialize(
                doc.doc_id, doc.text, chunker, chunker_impl
            )
            
            # Get or create embeddings
            embeddings = self.embedding_store.materialize(
                doc.doc_id, chunks, embedder, embedder_impl
            )
            
            # Add to index
            chunk_ids = list(range(len(chunks)))
            index_impl.add(doc.doc_id, chunk_ids, embeddings)
            
            # Update document state
            self.doc_state_store.update_state(
                doc.doc_id, chunker, embedder, doc.text
            )
        
        # Save index
        index_path = self.workspace_path / "index"
        index_path.mkdir(exist_ok=True)
        index_impl.save(index_path)
        
        elapsed = time.time() - start_time
        print(f"Build complete in {elapsed:.2f}s")
        
        self.index = index_impl
        return index_impl

    def query(self, query_vector: np.ndarray, k: int = 10) -> List[tuple]:
        """
        Query the built index
        
        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of (doc_id, chunk_id, score) tuples
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")
        
        return self.index.search(query_vector, k)

    def get_stats(self) -> dict:
        """Get build statistics"""
        return {
            "document_count": self.corpus.get_document_count(),
            "workspace_path": str(self.workspace_path),
            "index_built": self.index is not None
        }
