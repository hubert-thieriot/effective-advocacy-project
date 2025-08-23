"""
Index storage for vector search

This module provides IndexStore for managing FAISS indexes that enable
fast approximate nearest neighbor search over embeddings.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import logging

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

from efi_core.types import ChunkerSpec, EmbedderSpec

logger = logging.getLogger(__name__)


class IndexStore:
    """
    Manages FAISS indexes for fast vector search.
    
    Stores indexes under workspace/corpora/<name>/indexes/<chunker_key>/<embedder_key>/
    or workspace/libraries/<name>/indexes/<chunker_key>/<embedder_key>/
    """
    
    def __init__(self, workspace_path: Path, corpus_or_library_name: str, is_corpus: bool = True):
        """
        Initialize IndexStore.
        
        Args:
            workspace_path: Path to workspace directory
            corpus_or_library_name: Name of corpus or library
            is_corpus: True if this is for a corpus, False for library
        """
        self.workspace_path = workspace_path
        self.corpus_or_library_name = corpus_or_library_name
        self.is_corpus = is_corpus
        
        # Determine base path
        if is_corpus:
            self.base_path = workspace_path / "corpora" / corpus_or_library_name / "indexes"
        else:
            self.base_path = workspace_path / "libraries" / corpus_or_library_name / "indexes"
        
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def get_index_path(self, chunker_spec: ChunkerSpec, embedder_spec: EmbedderSpec) -> Path:
        """Get path to index directory for given specs."""
        chunker_key = chunker_spec.key()
        embedder_key = embedder_spec.key()
        return self.base_path / chunker_key / embedder_key
    
    def get_index_file_path(self, chunker_spec: ChunkerSpec, embedder_spec: EmbedderSpec) -> Path:
        """Get path to the actual index file."""
        index_dir = self.get_index_path(chunker_spec, embedder_spec)
        return index_dir / "index.faiss"
    
    def get_metadata_file_path(self, chunker_spec: ChunkerSpec, embedder_spec: EmbedderSpec) -> Path:
        """Get path to metadata file storing ID mappings."""
        index_dir = self.get_index_path(chunker_spec, embedder_spec)
        return index_dir / "metadata.json"
    
    def has_index(self, chunker_spec: ChunkerSpec, embedder_spec: EmbedderSpec) -> bool:
        """Check if index exists for given specs."""
        index_file = self.get_index_file_path(chunker_spec, embedder_spec)
        metadata_file = self.get_metadata_file_path(chunker_spec, embedder_spec)
        return index_file.exists() and metadata_file.exists()
    
    def build_index(
        self,
        chunker_spec: ChunkerSpec,
        embedder_spec: EmbedderSpec,
        embeddings_data: List[Tuple[str, np.ndarray]]
    ) -> bool:
        """
        Build FAISS index from embeddings data.
        
        Args:
            chunker_spec: Chunker specification
            embedder_spec: Embedder specification
            embeddings_data: List of (item_id, embedding_array) tuples
            
        Returns:
            True if index was built successfully
        """
        if not FAISS_AVAILABLE:
            logger.error("FAISS not available. Cannot build index.")
            return False
        
        if not embeddings_data:
            logger.warning("No embeddings data provided for index building.")
            return False
        
        try:
            # Get index directory
            index_dir = self.get_index_path(chunker_spec, embedder_spec)
            index_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare data
            item_ids = []
            all_vectors = []
            
            for item_id, embedding_array in embeddings_data:
                if embedding_array.size == 0:
                    continue
                
                # Handle both 1D and 2D arrays
                if embedding_array.ndim == 1:
                    vectors = embedding_array.reshape(1, -1)
                else:
                    vectors = embedding_array
                
                item_ids.extend([item_id] * len(vectors))
                all_vectors.append(vectors)
            
            if not all_vectors:
                logger.warning("No valid vectors found for index building.")
                return False
            
            # Concatenate all vectors
            vectors = np.vstack(all_vectors)
            dimension = vectors.shape[1]
            
            logger.info(f"Building FAISS index with {len(vectors)} vectors of dimension {dimension}")
            
            # Create FAISS index
            # Using IndexFlatIP (Inner Product) for cosine similarity
            # We'll normalize vectors before adding to index
            index = faiss.IndexFlatIP(dimension)
            
            # Normalize vectors for cosine similarity
            faiss.normalize_L2(vectors)
            
            # Add vectors to index
            index.add(vectors)
            
            # Save index
            index_file = self.get_index_file_path(chunker_spec, embedder_spec)
            faiss.write_index(index, str(index_file))
            
            # Save metadata
            metadata = {
                "item_ids": item_ids,
                "dimension": dimension,
                "total_vectors": len(vectors),
                "chunker_spec": {
                    "name": chunker_spec.name,
                    "params": chunker_spec.params
                },
                "embedder_spec": {
                    "model_name": embedder_spec.model_name,
                    "dim": embedder_spec.dim
                }
            }
            
            metadata_file = self.get_metadata_file_path(chunker_spec, embedder_spec)
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Successfully built index at {index_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error building index: {e}")
            return False
    
    def load_index(
        self, 
        chunker_spec: ChunkerSpec, 
        embedder_spec: EmbedderSpec
    ) -> Optional[Tuple[Any, List[str]]]:
        """
        Load existing FAISS index and metadata.
        
        Returns:
            Tuple of (faiss_index, item_ids) or None if loading fails
        """
        if not FAISS_AVAILABLE:
            logger.error("FAISS not available. Cannot load index.")
            return None
        
        try:
            index_file = self.get_index_file_path(chunker_spec, embedder_spec)
            metadata_file = self.get_metadata_file_path(chunker_spec, embedder_spec)
            
            if not index_file.exists() or not metadata_file.exists():
                return None
            
            # Load index
            index = faiss.read_index(str(index_file))
            
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            item_ids = metadata["item_ids"]
            
            logger.info(f"Loaded index with {len(item_ids)} items")
            return index, item_ids
            
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return None
    
    def delete_index(self, chunker_spec: ChunkerSpec, embedder_spec: EmbedderSpec) -> bool:
        """Delete index for given specs."""
        try:
            index_dir = self.get_index_path(chunker_spec, embedder_spec)
            if index_dir.exists():
                import shutil
                shutil.rmtree(index_dir)
                logger.info(f"Deleted index at {index_dir}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting index: {e}")
            return False
    
    def list_indexes(self) -> List[Tuple[str, str]]:
        """List all available indexes as (chunker_key, embedder_key) tuples."""
        indexes = []
        if self.base_path.exists():
            for chunker_dir in self.base_path.iterdir():
                if chunker_dir.is_dir():
                    for embedder_dir in chunker_dir.iterdir():
                        if embedder_dir.is_dir():
                            index_file = embedder_dir / "index.faiss"
                            metadata_file = embedder_dir / "metadata.json"
                            if index_file.exists() and metadata_file.exists():
                                indexes.append((chunker_dir.name, embedder_dir.name))
        return indexes
