"""
Index-based retriever using FAISS for fast similarity search.

This retriever uses FAISS indexes when available and can automatically
rebuild them if they don't exist.
"""

import numpy as np
from pathlib import Path
from typing import List, Union, Optional, Dict, Any
import logging

from efi_core.types import ChunkerSpec, EmbedderSpec, Candidate, Retriever
from efi_core.retrieval.index_builder import IndexBuilder
from efi_core.stores.indexes import IndexStore

logger = logging.getLogger(__name__)


class RetrieverIndex(Retriever):
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
        # Initialize instance variables
        self.embedded_data_source = embedded_data_source
        self.workspace_path = workspace_path
        self.chunker_spec = chunker_spec
        self.embedder_spec = embedder_spec
        self.auto_rebuild = auto_rebuild

        # Initialize cached index variables
        self._cached_index = None
        self._cached_item_ids = None

        # Initialize index store and builder
        self.index_store = IndexStore(
            workspace_path=workspace_path,
            corpus_or_library_name=self._get_data_source_name(),
            is_corpus=self._is_corpus()
        )

        # Initialize index builder
        self.index_builder = IndexBuilder(
            workspace_path=workspace_path,
            corpus_or_library_name=self._get_data_source_name(),
            is_corpus=self._is_corpus()
        )

        logger.info(f"Initialized FAISS retriever for {self._get_data_source_name()}")

    def _get_data_source_name(self) -> str:
        """Get the name of the data source."""
        if hasattr(self.embedded_data_source, 'corpus'):
            return self.embedded_data_source.corpus.corpus_path.name
        else:
            return self.embedded_data_source.library.library_path.name

    def _is_corpus(self) -> bool:
        """Check if the data source is a corpus."""
        return hasattr(self.embedded_data_source, 'corpus')
    
    def query(
        self,
        query_vector: Union[str, np.ndarray],
        top_k: int = 10
    ) -> List[Candidate]:
        """
        Query for similar items using FAISS index.

        Args:
            query_vector: Either text string or pre-computed embedding vector
            top_k: Number of top results to return

        Returns:
            List of Candidate objects sorted by score (highest first)
        """
        # Handle text queries by embedding them
        if isinstance(query_vector, str):
            query_vector = self._embed_text(query_vector)

        if query_vector is None:
            logger.error("Failed to create query vector")
            return []

        # Try to use FAISS index
        if self._ensure_index_exists():
            try:
                results = self._query_with_index(query_vector, top_k)
                logger.debug(f"Used FAISS index for query, found {len(results)} results")
                return results
            except Exception as e:
                logger.warning(f"Index query failed, falling back to brute-force: {e}")

        # Fall back to brute-force search
        logger.debug("Using brute-force cosine similarity search")
        return self._query_brute_force(query_vector, top_k)

    def _query_brute_force(self, query_vector: np.ndarray, top_k: int) -> List[Candidate]:
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
            if self._is_corpus():
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

                            # Get chunk text
                            chunk_text = ""
                            try:
                                chunk = self.embedded_data_source.get_chunk(
                                    chunk_id=chunk_id,
                                    materialize_if_necessary=False
                                )
                                chunk_text = chunk.text if chunk else ""
                            except Exception:
                                chunk_text = ""

                            all_results.append(Candidate(
                                item_id=chunk_id,
                                ann_score=float(similarity),
                                text=chunk_text,
                                meta={"doc_id": doc.doc_id, "chunk_idx": chunk_idx}
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

                                # Get chunk text for library
                                chunk_text = ""
                                try:
                                    chunks = self.embedded_data_source.get_chunks(
                                        finding_id=finding.finding_id,
                                        materialize_if_necessary=False
                                    )
                                    if chunks and chunk_idx < len(chunks):
                                        chunk_text = chunks[chunk_idx]
                                except Exception:
                                    chunk_text = ""

                                all_results.append(Candidate(
                                    item_id=finding.finding_id,
                                    ann_score=float(similarity),
                                    text=chunk_text,
                                    meta={"finding_id": finding.finding_id, "chunk_idx": chunk_idx}
                                ))

                        except Exception as e:
                            logger.warning(f"Error processing finding {finding.finding_id}: {e}")
                            continue

        except Exception as e:
            logger.error(f"Error in brute-force search: {e}")
            return []

        # Sort by score (highest first) and return top_k
        all_results.sort(key=lambda x: x.ann_score, reverse=True)
        return all_results[:top_k]

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
    
    def _query_with_index(self, query_vector: np.ndarray, top_k: int) -> List[Candidate]:
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

                # Get chunk text
                chunk_text = ""
                try:
                    if self._is_corpus():
                        # For corpus, item_id is chunk_id
                        chunk = self.embedded_data_source.get_chunk(
                            chunk_id=item_id,
                            materialize_if_necessary=False
                        )
                        chunk_text = chunk.text if chunk else ""
                    else:
                        # For library, item_id is finding_id, but we don't have chunk_idx
                        # Get the first chunk or the finding text itself
                        chunks = self.embedded_data_source.get_chunks(
                            finding_id=item_id,
                            materialize_if_necessary=False
                        )
                        if chunks:
                            chunk_text = chunks[0]  # Get first chunk
                        else:
                            # Fallback: get the finding text
                            finding = self.embedded_data_source.library.get_finding(item_id)
                            chunk_text = finding.text if finding else ""
                except Exception:
                    chunk_text = ""

                results.append(Candidate(item_id=item_id, ann_score=float(score), text=chunk_text))

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
