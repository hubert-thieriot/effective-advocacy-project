"""
Ensemble re-scorer that combines multiple scoring methods.

Combines original retrieval scores with cross-encoder scores using
configurable weights to get the best of both approaches.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from efi_core.protocols import ReScorer
from efi_core.retrieval.retriever import SearchResult


@dataclass
class EnsembleWeights:
    """Configuration for ensemble scoring weights"""
    original_score: float = 0.3
    cross_encoder_score: float = 0.7
    
    def __post_init__(self):
        """Validate weights sum to 1.0"""
        total = self.original_score + self.cross_encoder_score
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Ensemble weights must sum to 1.0, got {total}")


class EnsembleReScorer(ReScorer[SearchResult]):
    """
    Ensemble re-scorer that combines multiple scoring approaches.
    
    Combines original retrieval scores with cross-encoder scores
    using configurable weights. This provides a balance between
    speed (original scores) and accuracy (cross-encoder scores).
    """
    
    def __init__(self, 
                 cross_encoder_rescorer: ReScorer,
                 weights: Optional[EnsembleWeights] = None,
                 normalize_combined: bool = True):
        """
        Initialize the ensemble re-scorer.
        
        Args:
            cross_encoder_rescorer: Cross-encoder re-scorer to use
            weights: Weight configuration for combining scores
            normalize_combined: Whether to normalize final combined scores
        """
        self.cross_encoder_rescorer = cross_encoder_rescorer
        self.weights = weights or EnsembleWeights()
        self.normalize_combined = normalize_combined
    
    def rescore(self, query: str, matches: List[SearchResult]) -> List[SearchResult]:
        """
        Re-score using ensemble approach.
        
        Args:
            query: The original query text
            matches: List of SearchResult objects to re-score
            
        Returns:
            Re-ranked list with ensemble scores
        """
        if not matches:
            return matches
        
        # Store original scores for ensemble
        original_scores = {match.item_id: match.score for match in matches}
        
        # Get cross-encoder scores
        cross_encoder_results = self.cross_encoder_rescorer.rescore(query, matches)
        cross_encoder_scores = {result.item_id: result.score for result in cross_encoder_results}
        
        # Combine scores using weights
        combined_matches = []
        for match in matches:
            original_score = original_scores[match.item_id]
            cross_encoder_score = cross_encoder_scores[match.item_id]
            
            # Weighted combination
            combined_score = (
                self.weights.original_score * original_score +
                self.weights.cross_encoder_score * cross_encoder_score
            )
            
            # Create new result with combined score
            combined_match = SearchResult(
                item_id=match.item_id,
                score=combined_score,
                metadata={
                    **match.metadata,
                    'original_score': original_score,
                    'cross_encoder_score': cross_encoder_score,
                    'ensemble_score': combined_score
                }
            )
            combined_matches.append(combined_match)
        
        # Sort by combined scores
        combined_matches.sort(key=lambda x: x.score, reverse=True)
        
        # Normalize if requested
        if self.normalize_combined:
            combined_matches = self._normalize_scores(combined_matches)
        
        return combined_matches
    
    def _normalize_scores(self, matches: List[SearchResult]) -> List[SearchResult]:
        """Normalize scores to [0, 1] range"""
        if not matches:
            return matches
        
        scores = [match.score for match in matches]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            # All scores are the same, set to 0.5
            normalized_matches = []
            for match in matches:
                normalized_match = SearchResult(
                    item_id=match.item_id,
                    score=0.5,
                    metadata=match.metadata
                )
                normalized_matches.append(normalized_match)
            return normalized_matches
        
        # Normalize to [0, 1]
        normalized_matches = []
        for match in matches:
            normalized_score = (match.score - min_score) / (max_score - min_score)
            normalized_match = SearchResult(
                item_id=match.item_id,
                score=normalized_score,
                metadata=match.metadata
            )
            normalized_matches.append(normalized_match)
        
        return normalized_matches
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """Get information about the ensemble configuration"""
        return {
            "weights": {
                "original_score": self.weights.original_score,
                "cross_encoder_score": self.weights.cross_encoder_score
            },
            "normalize_combined": self.normalize_combined,
            "cross_encoder_info": getattr(self.cross_encoder_rescorer, 'get_model_info', lambda: {})()
        }
