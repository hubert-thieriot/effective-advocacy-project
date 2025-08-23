"""
Sentence Transformer-based embedder for text vectorization
"""

import numpy as np
from typing import List, Optional

from efi_core.protocols import Embedder
from efi_core.types import EmbedderSpec


class SentenceTransformerEmbedder(Embedder):
    """
    Embedder using Sentence Transformers for text vectorization.
    
    Uses the sentence-transformers library to generate embeddings.
    Falls back to random embeddings if the library is not available.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", revision: Optional[str] = None, lazy_load: bool = False):
        """
        Initialize the embedder
        
        Args:
            model_name: Name of the sentence transformer model
            revision: Model revision (optional)
            lazy_load: If True, don't load model until first use (faster initialization)
        """
        self.model_name = model_name
        self.revision = revision
        self.model = None
        self._dim = None
        self._lazy_load = lazy_load
        
        if not lazy_load:
            # Try to load the model immediately
            self._load_model()
        else:
            # Set default dimension for lazy loading
            self._dim = 384  # Default for all-MiniLM-L6-v2
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            # Get dimension from a test encoding
            test_embedding = self.model.encode(["test"])
            self._dim = test_embedding.shape[1]
        except ImportError:
            print(f"Warning: sentence-transformers not available. Using fallback embedder.")
            self.model = None
            self._dim = 384  # Default dimension
        except Exception as e:
            print(f"Warning: Failed to load model {self.model_name}: {e}. Using fallback embedder.")
            self.model = None
            self._dim = 384  # Default dimension
    
    @property
    def spec(self) -> EmbedderSpec:
        """Get the embedder specification"""
        return EmbedderSpec(
            model_name=self.model_name,
            dim=self.dim,
            revision=self.revision
        )
    
    @property
    def dim(self) -> int:
        """Get the embedding dimension"""
        return self._dim
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embeddings with shape (len(texts), dim)
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        
        # Load model if lazy loading and not yet loaded
        if self._lazy_load and self.model is None:
            self._load_model()
        
        if self.model is not None:
            # Use real sentence transformer
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.astype(np.float32)
        else:
            # Fallback to random embeddings (for testing/development)
            np.random.seed(hash(" ".join(texts)) % 2**32)  # Deterministic random
            embeddings = np.random.normal(0, 1, (len(texts), self.dim)).astype(np.float32)
            # Normalize to unit vectors
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)
            return embeddings
    
    def embed_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector with shape (dim,)
        """
        embeddings = self.embed([text])
        return embeddings[0]
