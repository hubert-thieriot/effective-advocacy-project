"""
Index implementations for approximate nearest neighbor search
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

from .types import AnnIndex


@dataclass
class NumpyScanIndex:
    """
    Simple exact scan index using numpy.
    
    This is a fallback implementation for development and testing.
    In production, use FAISS or other optimized ANN libraries.
    """
    
    _vectors: List[np.ndarray] = field(default_factory=list)
    _refs: List[Tuple[str, int]] = field(default_factory=list)
    _dim: Optional[int] = None

    def add(self, doc_id: str, chunk_ids: List[int], vectors: np.ndarray) -> None:
        """Add document chunks to the index"""
        if self._dim is None:
            self._dim = vectors.shape[1]
        elif vectors.shape[1] != self._dim:
            raise ValueError(f"Vector dimension mismatch: expected {self._dim}, got {vectors.shape[1]}")
        
        self._vectors.append(vectors)
        self._refs.extend([(doc_id, i) for i in chunk_ids])

    def query(self, q: List[float], top_k: int) -> List[Tuple[str, int, float]]:
        """
        Query the index for similar vectors
        
        Args:
            q: Query vector
            top_k: Number of top results to return
            
        Returns:
            List of (doc_id, chunk_id, similarity_score) tuples
        """
        if not self._vectors:
            return []
        
        # Convert query to numpy array
        query = np.array(q, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        # Stack all vectors
        all_vectors = np.vstack(self._vectors)
        
        # Normalize for cosine similarity
        query_norm = query / (np.linalg.norm(query) + 1e-9)
        vectors_norm = all_vectors / (np.linalg.norm(all_vectors, axis=1, keepdims=True) + 1e-9)
        
        # Compute similarities
        similarities = vectors_norm @ query_norm.T
        similarities = similarities.flatten()
        
        # Get top-k results
        top_indices = np.argpartition(-similarities, min(top_k, len(similarities)-1))[:top_k]
        top_indices = top_indices[np.argsort(-similarities[top_indices])]
        
        # Return results
        results = []
        for idx in top_indices:
            doc_id, chunk_id = self._refs[idx]
            score = float(similarities[idx])
            results.append((doc_id, chunk_id, score))
        
        return results

    def persist(self, path: Path) -> None:
        """Save index to disk"""
        path.mkdir(parents=True, exist_ok=True)
        
        # Save vectors
        if self._vectors:
            vectors_array = np.vstack(self._vectors)
            np.save(path / "vectors.npy", vectors_array)
        else:
            # Save empty array with correct shape
            empty_array = np.zeros((0, self._dim or 1))
            np.save(path / "vectors.npy", empty_array)
        
        # Save references
        with open(path / "refs.json", "w", encoding="utf-8") as f:
            json.dump(self._refs, f, ensure_ascii=False)
        
        # Save metadata
        metadata = {
            "dim": self._dim,
            "num_vectors": len(self._refs),
            "num_docs": len(set(ref[0] for ref in self._refs))
        }
        with open(path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def load(self, path: Path) -> None:
        """Load index from disk"""
        # Load vectors
        vectors_path = path / "vectors.npy"
        if vectors_path.exists():
            all_vectors = np.load(vectors_path)
            if all_vectors.size > 0:
                self._vectors = [all_vectors]
                self._dim = all_vectors.shape[1]
            else:
                self._vectors = []
                self._dim = None
        else:
            self._vectors = []
            self._dim = None
        
        # Load references
        refs_path = path / "refs.json"
        if refs_path.exists():
            with open(refs_path, "r", encoding="utf-8") as f:
                self._refs = [tuple(ref) for ref in json.load(f)]
        else:
            self._refs = []

    def get_stats(self) -> dict:
        """Get index statistics"""
        return {
            "num_vectors": len(self._refs),
            "num_documents": len(set(ref[0] for ref in self._refs)),
            "dimension": self._dim,
            "memory_mb": sum(v.nbytes for v in self._vectors) / (1024 * 1024) if self._vectors else 0
        }
