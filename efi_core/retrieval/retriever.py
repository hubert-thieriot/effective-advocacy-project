"""
Unified retriever for vector search over embedded corpora and libraries.

This module provides a Retriever class that can work with both EmbeddedCorpus
and EmbeddedLibrary to perform fast similarity search using either FAISS
indexes or brute-force cosine similarity.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Union, Optional, Dict, Any
import logging

from efi_core.stores.indexes import IndexStore
from efi_core.types import ChunkerSpec, EmbedderSpec
from efi_core.protocols import ReScorer

logger = logging.getLogger(__name__)


class SearchResult:
    """Represents a single search result with metadata."""
    
    def __init__(self, item_id: str, score: float, metadata: Optional[Dict[str, Any]] = None):
        self.item_id = item_id
        self.score = score
        self.metadata = metadata or {}
    
    def __repr__(self):
        return f"SearchResult(item_id='{self.item_id}', score={self.score:.4f})"


class Retriever:
    """
    Unified retriever for vector search over embedded data.
    
    Supports both FAISS-based approximate search and brute-force cosine similarity.
    Can work with either EmbeddedCorpus or EmbeddedLibrary.
    """
    
    def __init__(
        self,
        embedded_data_source,  # EmbeddedCorpus or EmbeddedLibrary
        workspace_path: Path,
        chunker_spec: ChunkerSpec,
        embedder_spec: EmbedderSpec,
        rescorer: Optional[ReScorer["SearchResult"]] = None
    ):
        """
        Initialize retriever.
        
        Args:
            embedded_data_source: EmbeddedCorpus or EmbeddedLibrary instance
            workspace_path: Path to workspace directory
            chunker_spec: Chunker specification
            embedder_spec: Embedder specification
            rescorer: Optional re-scorer for improving result quality
        """
        self.embedded_data_source = embedded_data_source
        self.workspace_path = workspace_path
        self.chunker_spec = chunker_spec
        self.embedder_spec = embedder_spec
        self.rescorer = rescorer
        
        # Determine if this is a corpus or library
        self.is_corpus = hasattr(embedded_data_source, 'corpus')
        self.name = embedded_data_source.corpus.corpus_path.name if self.is_corpus else embedded_data_source.library.library_path.name
        
        # Initialize index store
        self.index_store = IndexStore(
            workspace_path=workspace_path,
            corpus_or_library_name=self.name,
            is_corpus=self.is_corpus
        )
        
        # Cache for loaded index
        self._cached_index = None
        self._cached_item_ids = None
    
    def query(
        self,
        query_vector: Union[str, np.ndarray],
        top_k: int = 10,
        use_index: bool = True,
        rescore: bool = True
    ) -> List[SearchResult]:
        """
        Query for similar items.
        
        Args:
            query_vector: Either text string or pre-computed embedding vector
            top_k: Number of top results to return
            use_index: Whether to try using FAISS index first
            rescore: Whether to apply re-scoring if a rescorer is configured
            
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
            return []
        
        # Try to use index if requested and available
        if use_index and self.index_store.has_index(self.chunker_spec, self.embedder_spec):
            try:
                results = self._query_with_index(query_vector, top_k)
                if results:
                    logger.debug(f"Used FAISS index for query, found {len(results)} results")
                    if rescore and self.rescorer and query_text:
                        results = self._apply_rescoring(query_text, results)
                    return results
            except Exception as e:
                logger.warning(f"Index query failed, falling back to brute-force: {e}")
        
        # Fall back to brute-force search
        logger.debug("Using brute-force cosine similarity search")
        results = self._query_brute_force(query_vector, top_k)
        
        # Apply re-scoring if configured
        if rescore and self.rescorer and query_text:
            results = self._apply_rescoring(query_text, results)
        
        return results
    
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
                raise RuntimeError("Failed to load index")
            
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
    
    def _query_brute_force(self, query_vector: np.ndarray, top_k: int) -> List[SearchResult]:
        """Query using brute-force cosine similarity."""
        query_vector = query_vector.astype(np.float32)
        
        # Normalize query vector
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return []
        query_vector = query_vector / query_norm
        
        all_results = []
        
        try:
            # Stream through all embeddings
            if self.is_corpus:
                # For corpus: iterate through documents
                for doc in self.embedded_data_source.corpus.iter_documents():
                    try:
                        embeddings = self.embedded_data_source.embeddings(doc.doc_id)
                        if embeddings.size == 0:
                            continue
                        
                        # Handle both 1D and 2D arrays
                        if embeddings.ndim == 1:
                            embeddings = embeddings.reshape(1, -1)
                        
                        # Calculate cosine similarity for each chunk
                        for chunk_idx, chunk_embedding in enumerate(embeddings):
                            chunk_embedding = chunk_embedding.astype(np.float32)
                            chunk_norm = np.linalg.norm(chunk_embedding)
                            if chunk_norm == 0:
                                continue
                            
                            chunk_embedding = chunk_embedding / chunk_norm
                            similarity = np.dot(query_vector, chunk_embedding)
                            
                            # Create chunk ID
                            chunk_id = f"{doc.doc_id}_chunk_{chunk_idx}"
                            all_results.append(SearchResult(
                                item_id=chunk_id,
                                score=float(similarity),
                                metadata={"doc_id": doc.doc_id, "chunk_idx": chunk_idx}
                            ))
                    
                    except Exception as e:
                        logger.warning(f"Error processing document {doc.doc_id}: {e}")
                        continue
            
            else:
                # For library: iterate through findings
                for doc_findings in self.embedded_data_source.library.iter_findings():
                    for finding in doc_findings.findings:
                        try:
                            embeddings = self.embedded_data_source.embeddings(finding.finding_id)
                            if embeddings.size == 0:
                                continue
                            
                            # Handle both 1D and 2D arrays
                            if embeddings.ndim == 1:
                                embeddings = embeddings.reshape(1, -1)
                            
                            # Calculate cosine similarity for each chunk
                            for chunk_idx, chunk_embedding in enumerate(embeddings):
                                chunk_embedding = chunk_embedding.astype(np.float32)
                                chunk_norm = np.linalg.norm(chunk_embedding)
                                if chunk_norm == 0:
                                    continue
                                
                                chunk_embedding = chunk_embedding / chunk_norm
                                similarity = np.dot(query_vector, chunk_embedding)
                                
                                all_results.append(SearchResult(
                                    item_id=finding.finding_id,
                                    score=float(similarity),
                                    metadata={"finding_id": finding.finding_id, "chunk_idx": chunk_idx}
                                ))
                        
                        except Exception as e:
                            logger.warning(f"Error processing finding {finding.finding_id}: {e}")
                            continue
        
        except Exception as e:
            logger.error(f"Error in brute-force search: {e}")
            return []
        
        # Sort by score (highest first) and return top_k
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:top_k]
    
    def has_index(self) -> bool:
        """Check if FAISS index exists."""
        return self.index_store.has_index(self.chunker_spec, self.embedder_spec)
    
    def list_indexes(self) -> List[Tuple[str, str]]:
        """List all available indexes."""
        return self.index_store.list_indexes()
    
    def _apply_rescoring(self, query_text: str, results: List[SearchResult]) -> List[SearchResult]:
        """
        Apply re-scoring to search results.
        
        Args:
            query_text: The original query text
            results: List of SearchResult objects to re-score
            
        Returns:
            Re-scored and re-ranked results
        """
        if not self.rescorer or not results:
            return results
        
        try:
            logger.debug(f"Applying re-scoring to {len(results)} results")
            rescored_results = self.rescorer.rescore(query_text, results)
            logger.debug(f"Re-scoring completed, returned {len(rescored_results)} results")
            return rescored_results
        except Exception as e:
            logger.warning(f"Re-scoring failed: {e}, returning original results")
            return results
