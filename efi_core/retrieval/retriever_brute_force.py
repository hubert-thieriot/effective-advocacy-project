"""
Brute-force retriever using cosine similarity.

This retriever always performs exhaustive search over all embeddings
without using any indexes. It's simple, reliable, and works well for
small to medium-sized datasets.
"""

import numpy as np
from pathlib import Path
from typing import List, Union, Optional, Dict, Any
import logging

from efi_core.types import ChunkerSpec, EmbedderSpec
from efi_core.retrieval.retriever import SearchResult

logger = logging.getLogger(__name__)


class RetrieverBrute:
    """
    Brute-force retriever using cosine similarity.
    
    Always performs exhaustive search over all embeddings without
    using any indexes. Simple, reliable, and works well for
    small to medium-sized datasets.
    """
    
    def __init__(
        self,
        embedded_data_source,  # EmbeddedCorpus or EmbeddedLibrary
        chunker_spec: ChunkerSpec,
        embedder_spec: EmbedderSpec
    ):
        """
        Initialize brute-force retriever.
        
        Args:
            embedded_data_source: EmbeddedCorpus or EmbeddedLibrary instance
            chunker_spec: Chunker specification
            embedder_spec: Embedder specification
        """
        self.embedded_data_source = embedded_data_source
        self.chunker_spec = chunker_spec
        self.embedder_spec = embedder_spec
        
        # Determine if this is a corpus or library
        self.is_corpus = hasattr(embedded_data_source, 'corpus')
        self.name = embedded_data_source.corpus.corpus_path.name if self.is_corpus else embedded_data_source.library.library_path.name
        
        logger.info(f"Initialized brute-force retriever for {self.name}")
    
    def query(
        self,
        query_vector: Union[str, np.ndarray],
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Query for similar items using brute-force cosine similarity.
        
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
            return []
        
        # Perform brute-force search
        logger.debug("Using brute-force cosine similarity search")
        return self._query_brute_force(query_vector, top_k)
    
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
                for doc in self.embedded_data_source.corpus.read_documents():
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
                for doc_findings in self.embedded_data_source.library.read_findings():
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
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the retriever configuration."""
        return {
            "type": "brute_force",
            "name": self.name,
            "is_corpus": self.is_corpus,
            "chunker_spec": str(self.chunker_spec),
            "embedder_spec": str(self.embedder_spec)
        }
