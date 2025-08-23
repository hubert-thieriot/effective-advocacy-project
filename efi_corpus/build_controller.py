"""
Build controller for corpus processing pipeline
"""

import time
from pathlib import Path
from typing import Optional

from efi_core.types import ChunkerSpec
from efi_core.protocols import Chunker, Embedder, AnnIndex
from .corpus_handle import CorpusHandle
from efi_core.stores import ChunkStore, EmbeddingStore, DocStateStore
from efi_core.layout import EmbeddedCorpusLayout


class BuildController:
    """
    Orchestrates the corpus processing pipeline:
    1. Reads documents from corpus
    2. Chunks text based on specification
    3. Generates embeddings for chunks
    4. Tracks state for incremental rebuilds
    """
    
    def __init__(self, corpus_path: str, workspace_path: str):
        self.corpus = CorpusHandle(corpus_path, read_only=True)
        self.workspace_path = Path(workspace_path)
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        self.layout = EmbeddedCorpusLayout(
            corpus_path=Path(corpus_path),
            workspace_root=self.workspace_path
        )

        # Initialize stores
        self.chunk_store = ChunkStore(self.layout)
        self.embedding_store = EmbeddingStore(self.layout)
        self.doc_state_store = DocStateStore(self.layout)

    def build(self, chunker: ChunkerSpec, embedder_impl: Embedder,
              chunker_impl: Chunker) -> None:
        """
        Build the embedded corpus
        
        Args:
            chunker: Chunking specification
            embedder: Embedding specification  
            chunker_impl: Chunking implementation
            embedder_impl: Embedding implementation
        """
        print(f"Building embedded corpus with {self.corpus.get_document_count()} documents...")
        
        start_time = time.time()
        
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
                doc.doc_id, chunks, chunker, embedder_impl
            )
            
            # Update document state
            fingerprint = self.corpus.fingerprint(doc.doc_id)
            state = {
                "document_id": doc.doc_id,
                "fingerprint": fingerprint,
                "last_built_ts": time.time(),
                "chunker_key": chunker.key(),
                "embedder_key": embedder_impl.spec.key(),
            }
            from efi_core.types import DocState
            self.doc_state_store.write(DocState(**state))
        
        elapsed = time.time() - start_time
        print(f"Build complete in {elapsed:.2f}s")

    def get_corpus_info(self) -> dict:
        """Get information about the corpus"""
        return {
            "corpus_path": str(self.corpus.corpus_path),
            "workspace_path": str(self.workspace_path),
            "document_count": self.corpus.get_document_count(),
            "read_only": self.corpus.read_only
        }
