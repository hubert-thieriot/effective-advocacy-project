"""
Index builder for creating FAISS indexes from embedded data.

This module provides IndexBuilder for constructing FAISS indexes from
EmbeddedCorpus or EmbeddedLibrary instances.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np

from efi_core.stores.indexes import IndexStore
from efi_core.types import ChunkerSpec, EmbedderSpec

logger = logging.getLogger(__name__)


class IndexBuilder:
    """
    Builds FAISS indexes from embedded corpora and libraries.
    
    Handles the collection of embeddings and construction of indexes
    for fast vector search.
    """
    
    def __init__(
        self,
        workspace_path: Path,
        corpus_or_library_name: str,
        is_corpus: bool = True
    ):
        """
        Initialize IndexBuilder.
        
        Args:
            workspace_path: Path to workspace directory
            corpus_or_library_name: Name of corpus or library
            is_corpus: True if this is for a corpus, False for library
        """
        self.workspace_path = workspace_path
        self.corpus_or_library_name = corpus_or_library_name
        self.is_corpus = is_corpus
        
        # Initialize index store
        self.index_store = IndexStore(
            workspace_path=workspace_path,
            corpus_or_library_name=corpus_or_library_name,
            is_corpus=is_corpus
        )
    
    def build_index(
        self,
        embedded_data_source,  # EmbeddedCorpus or EmbeddedLibrary
        chunker_spec: ChunkerSpec,
        embedder_spec: EmbedderSpec,
        force_rebuild: bool = False
    ) -> bool:
        """
        Build FAISS index from embedded data source.
        
        Args:
            embedded_data_source: EmbeddedCorpus or EmbeddedLibrary instance
            chunker_spec: Chunker specification
            embedder_spec: Embedder specification
            force_rebuild: If True, rebuild index even if it exists
            
        Returns:
            True if index was built successfully
        """
        # Check if index already exists
        if not force_rebuild and self.index_store.has_index(chunker_spec, embedder_spec):
            logger.info(f"Index already exists for {chunker_spec.name}/{embedder_spec.model_name}")
            return True
        
        logger.info(f"Building index for {self.corpus_or_library_name}")
        
        # Collect all embeddings for the specific chunker and embedder specs
        embeddings_data = self._collect_embeddings(embedded_data_source, chunker_spec, embedder_spec)
        
        if not embeddings_data:
            logger.warning("No embeddings found for index building")
            return False
        
        logger.info(f"Collected {len(embeddings_data)} embeddings for index building")
        
        # Build index using index store
        success = self.index_store.build_index(
            chunker_spec=chunker_spec,
            embedder_spec=embedder_spec,
            embeddings_data=embeddings_data
        )
        
        if success:
            logger.info(f"Successfully built index for {self.corpus_or_library_name}")
        else:
            logger.error(f"Failed to build index for {self.corpus_or_library_name}")
        
        return success
    
    def _collect_embeddings(self, embedded_data_source, chunker_spec: ChunkerSpec, embedder_spec: EmbedderSpec) -> List[Tuple[str, np.ndarray]]:
        """
        Collect embeddings from embedded data source for specific specs.
        
        Args:
            embedded_data_source: EmbeddedCorpus or EmbeddedLibrary instance
            chunker_spec: Chunker specification
            embedder_spec: Embedder specification
            
        Returns:
            List of (item_id, embedding_array) tuples
        """
        embeddings_data = []
        
        try:
            # Use the iter_embeddings method - the embedded data source already knows its specs
            for item_id, embedding_vector in embedded_data_source.iter_embeddings():
                embeddings_data.append((item_id, embedding_vector))
        
        except Exception as e:
            logger.error(f"Error collecting embeddings: {e}")
            return []
        
        return embeddings_data
    
    def has_index(self, chunker_spec: ChunkerSpec, embedder_spec: EmbedderSpec) -> bool:
        """Check if index exists for given specs."""
        return self.index_store.has_index(chunker_spec, embedder_spec)
    
    def delete_index(self, chunker_spec: ChunkerSpec, embedder_spec: EmbedderSpec) -> bool:
        """Delete existing index for given specs."""
        return self.index_store.delete_index(chunker_spec, embedder_spec)
    
    def list_indexes(self) -> List[Tuple[str, str]]:
        """List all available indexes."""
        return self.index_store.list_indexes()
    
    def get_index_info(self, chunker_spec: ChunkerSpec, embedder_spec: EmbedderSpec) -> dict:
        """Get information about a specific index."""
        if not self.has_index(chunker_spec, embedder_spec):
            return {}
        
        try:
            metadata_file = self.index_store.get_metadata_file_path(chunker_spec, embedder_spec)
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            return {
                "chunker_spec": metadata.get("chunker_spec", {}),
                "embedder_spec": metadata.get("embedder_spec", {}),
                "dimension": metadata.get("dimension"),
                "total_vectors": metadata.get("total_vectors"),
                "index_path": str(self.index_store.get_index_path(chunker_spec, embedder_spec))
            }
        except Exception as e:
            logger.error(f"Error reading index info: {e}")
            return {}
