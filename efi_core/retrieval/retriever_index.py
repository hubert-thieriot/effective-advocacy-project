"""
Index-based retriever using FAISS for fast similarity search.

This retriever uses FAISS indexes when available and can automatically
rebuild them if they don't exist.
"""

import numpy as np
from pathlib import Path
from typing import List, Union, Optional, Dict, Any
import logging

from efi_core.types import ChunkerSpec, EmbedderSpec
from efi_core.retrieval.retriever import SearchResult
from efi_core.retrieval.index_builder import IndexBuilder
from efi_core.stores.indexes import IndexStore

logger = logging.getLogger(__name__)


class RetrieverIndex:
    """
    Index-based retriever using FAISS for fast similarity search.
    
    Uses FAISS indexes when available and can automatically rebuild them
    if they don't exist. Falls back to brute-force search if needed.
    """
    
    def __init__(
        self,
        embedded_data_source,  # EmbeddedCorpus or EmbeddedLibrary
        workspace_path: Path,
        chunker_spec: ChunkerSpec,
        embedder_spec: EmbedderSpec,
        auto_rebuild: bool = True
    ):
        """
        Initialize index retriever.
        
        Args:
            embedded_data_source: EmbeddedCorpus or EmbeddedLibrary instance
            workspace_path: Path to workspace directory
            chunker_spec: Chunker specification
            embedder_spec: Embedder specification
            auto_rebuild: Whether to automatically rebuild missing indexes
        """
        self.embedded_data_source = embedded_data_source
        self.workspace_path = workspace_path
        self.chunker_spec = chunker_spec
        self.embedder_spec = embedder_spec
        self.auto_rebuild = auto_rebuild
        
        # Determine if this is a corpus or library
        self.is_corpus = hasattr(embedded_data_source, 'corpus')
        self.name = embedded_data_source.corpus.corpus_path.name if self.is_corpus else embedded_data_source.library.library_path.name
        
        # Initialize index store and builder
        self.index_store = IndexStore(
            workspace_path=workspace_path,
            corpus_or_library_name=self.name,
            is_corpus=self.is_corpus
        )
        
        self.index_builder = IndexBuilder(
            workspace_path=workspace_path,
            corpus_or_library_name=self.name,
            is_corpus=self.is_corpus
        )
        
        # Cache for loaded index
        self._cached_index = None
        self._cached_item_ids = None
        
        logger.info(f"Initialized FAISS retriever for {self.name}")
    
    def query(
        self,
        query_vector: Union[str, np.ndarray],
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Query for similar items using FAISS index.
        
        Args:
            query_vector: Either text string or pre-computed embedding vector
            top_k: Number of top results to return
            
        Returns:
            List of SearchResult objects sorted by score (highest first)
        """
        # Handle text queries by embedding them
        if isinstance(query_vector, str):
            query_text = query_vector
            query_vector = self._embed_text(query_vector)
        else:
            query_text = None
        
        if query_vector is None:
            logger.error("Failed to create query vector")
            raise ValueError("Failed to create query vector for query")
        
        # Try to use FAISS index
        if self._ensure_index_exists():
            try:
                results = self._query_with_index(query_vector, top_k)
                logger.debug(f"Used FAISS index for query, found {len(results)} results")
                return results
            except Exception as e:
                logger.error(f"FAISS index query failed: {e}")
                raise
        else:
            raise RuntimeError("No FAISS index available and auto-rebuild failed or disabled")
    
    def _ensure_index_exists(self) -> bool:
        """
        Ensure the FAISS index exists, rebuilding if necessary.
        
        Returns:
            True if index exists and is valid, False otherwise
        """
        if self.index_store.has_index(self.chunker_spec, self.embedder_spec):
            return True
        
        if not self.auto_rebuild:
            logger.info("No FAISS index found and auto-rebuild is disabled")
            return False
        
        logger.info("No FAISS index found, attempting to rebuild...")
        try:
            self._rebuild_index()
            return self.index_store.has_index(self.chunker_spec, self.embedder_spec)
        except Exception as e:
            logger.error(f"Failed to rebuild FAISS index: {e}")
            return False
    
    def _rebuild_index(self):
        """Rebuild the FAISS index for the current chunker and embedder specs."""
        logger.info(f"Building FAISS index for {self.name}...")
        
        success = self.index_builder.build_index(
            embedded_data_source=self.embedded_data_source,
            chunker_spec=self.chunker_spec,
            embedder_spec=self.embedder_spec,
            force_rebuild=True
        )
        
        if success:
            logger.info("Successfully built FAISS index")
        else:
            raise RuntimeError("Failed to build FAISS index")
    
    def _embed_text(self, text: str) -> Optional[np.ndarray]:
        """Embed text using the embedded data source's embedder."""
        try:
            # Use the embedder from the embedded data source
            embedder = self.embedded_data_source.embedder
            
            # Create a temporary chunk to get the embedding
            # We'll use the first chunk if text is long
            if len(text) > 1000:  # Arbitrary threshold
                text = text[:1000] + "..."
            
            # Get embedding for the text
            embedding = embedder.embed([text])
            if embedding is not None and len(embedding) > 0:
                return np.array(embedding[0])
            
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
        
        return None
    
    def _query_with_index(self, query_vector: np.ndarray, top_k: int) -> List[SearchResult]:
        """Query using FAISS index."""
        # Check if FAISS is available
        try:
            import faiss
        except ImportError:
            raise RuntimeError("FAISS not available for index-based search")
        
        # Load index if not cached
        if self._cached_index is None:
            index_data = self.index_store.load_index(self.chunker_spec, self.embedder_spec)
            if index_data is None:
                raise RuntimeError("Failed to load FAISS index")
            
            self._cached_index, self._cached_item_ids = index_data
        
        # Normalize query vector for cosine similarity
        query_vector = query_vector.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query_vector)
        
        # Search index
        scores, indices = self._cached_index.search(query_vector, min(top_k, len(self._cached_item_ids)))
        
        # Convert to results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self._cached_item_ids):
                item_id = self._cached_item_ids[idx]
                # Convert inner product score to cosine similarity (they're the same for normalized vectors)
                results.append(SearchResult(item_id=item_id, score=float(score)))
        
        return results
    
    
    def has_index(self) -> bool:
        """Check if FAISS index exists."""
        return self.index_store.has_index(self.chunker_spec, self.embedder_spec)
    
    def list_indexes(self) -> List[tuple]:
        """List all available indexes."""
        return self.index_store.list_indexes()
    
    def rebuild_index(self):
        """Manually rebuild the FAISS index."""
        logger.info("Manually rebuilding FAISS index...")
        self._rebuild_index()
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the retriever configuration."""
        return {
            "type": "index",
            "name": self.name,
            "is_corpus": self.is_corpus,
            "chunker_spec": str(self.chunker_spec),
            "embedder_spec": str(self.embedder_spec),
            "has_index": self.has_index(),
            "auto_rebuild": self.auto_rebuild
        }
