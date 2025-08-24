"""
Cross-encoder based re-scorer for improving retrieval quality.

Uses sentence-transformers cross-encoder models to re-score retrieved results
by jointly encoding query-document pairs for more accurate semantic alignment.
"""

import numpy as np
from typing import List, Optional
from pathlib import Path

from efi_core.protocols import ReScorer
from efi_core.retrieval.retriever import SearchResult


class CrossEncoderReScorer(ReScorer[SearchResult]):
    """
    Re-scorer using cross-encoder models for improved semantic alignment.
    
    Cross-encoders jointly encode query-document pairs, providing more
    accurate similarity scores than dual-encoder approaches. This is
    typically used as a second-stage ranking after fast retrieval.
    """
    
    def __init__(self, 
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 batch_size: int = 32,
                 use_gpu: bool = False,
                 normalize_scores: bool = True):
        """
        Initialize the cross-encoder re-scorer.
        
        Args:
            model_name: Name of the cross-encoder model
            batch_size: Batch size for processing (larger = faster but more memory)
            use_gpu: Whether to use GPU acceleration if available
            normalize_scores: Whether to normalize scores to [0, 1] range
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.normalize_scores = normalize_scores
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the cross-encoder model"""
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name, max_length=512)
            if self.use_gpu:
                self.model.to('cuda')
            print(f"✓ Loaded cross-encoder: {self.model_name}")
        except ImportError:
            print(f"⚠️ Warning: sentence-transformers not available. Using fallback re-scorer.")
            self.model = None
        except Exception as e:
            print(f"⚠️ Warning: Failed to load model {self.model_name}: {e}. Using fallback re-scorer.")
            self.model = None
    
    def rescore(self, query: str, matches: List[SearchResult]) -> List[SearchResult]:
        """
        Re-score search results using cross-encoder.
        
        Args:
            query: The original query text
            matches: List of SearchResult objects to re-score
            
        Returns:
            Re-ranked list of SearchResult objects with updated scores
        """
        if not matches:
            return matches
        
        if self.model is None:
            # Fallback: return original results with slight random variation
            return self._fallback_rescore(matches)
        
        try:
            # Prepare query-document pairs for cross-encoder
            pairs = [(query, match.item_id) for match in matches]
            
            # Get cross-encoder scores
            scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
            
            # Normalize scores if requested
            if self.normalize_scores:
                scores = self._normalize_scores(scores)
            
            # Create new SearchResult objects with updated scores
            rescored_matches = []
            for match, new_score in zip(matches, scores):
                # Create new SearchResult with updated score
                rescored_match = SearchResult(
                    item_id=match.item_id,
                    score=float(new_score),
                    metadata=match.metadata
                )
                rescored_matches.append(rescored_match)
            
            # Sort by new scores (descending)
            rescored_matches.sort(key=lambda x: x.score, reverse=True)
            
            return rescored_matches
            
        except Exception as e:
            print(f"⚠️ Error in cross-encoder re-scoring: {e}. Using fallback.")
            return self._fallback_rescore(matches)
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range"""
        if len(scores) == 0:
            return scores
        
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score == min_score:
            # All scores are the same, set to 0.5
            return np.full_like(scores, 0.5)
        
        # Normalize to [0, 1]
        normalized = (scores - min_score) / (max_score - min_score)
        return normalized
    
    def _fallback_rescore(self, matches: List[SearchResult]) -> List[SearchResult]:
        """Fallback re-scoring when cross-encoder is not available"""
        # Add small random variation to break ties
        np.random.seed(42)  # Deterministic for reproducibility
        variations = np.random.normal(0, 0.001, len(matches))
        
        rescored_matches = []
        for match, variation in zip(matches, variations):
            new_score = match.score + variation
            rescored_match = SearchResult(
                item_id=match.item_id,
                score=float(new_score),
                metadata=match.metadata
            )
            rescored_matches.append(rescored_match)
        
        # Sort by new scores
        rescored_matches.sort(key=lambda x: x.score, reverse=True)
        return rescored_matches
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "model_loaded": self.model is not None,
            "batch_size": self.batch_size,
            "use_gpu": self.use_gpu,
            "normalize_scores": self.normalize_scores
        }
