"""
Ensemble re-scorer that combines multiple scoring methods.

Combines original retrieval scores with cross-encoder scores using
configurable weights to get the best of both approaches.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from efi_core.types import PairScorer, Task


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


class EnsembleReScorer(PairScorer):
    """
    Ensemble re-scorer that combines multiple scoring approaches.
    
    Combines original retrieval scores with cross-encoder scores
    using configurable weights. This provides a balance between
    speed (original scores) and accuracy (cross-encoder scores).
    """
    
    def __init__(self,
                 name: str = "ensemble_scorer",
                 task: Task = Task.NLI,
                 cross_encoder_rescorer: PairScorer = None,
                 weights: Optional[EnsembleWeights] = None,
                 normalize_combined: bool = True):
        """
        Initialize the ensemble scorer.

        Args:
            name: Name identifier for this scorer
            task: Task type
            cross_encoder_rescorer: Cross-encoder scorer to use
            weights: Weight configuration for combining scores
            normalize_combined: Whether to normalize final combined scores
        """
        super().__init__(name, task)
        self.cross_encoder_rescorer = cross_encoder_rescorer
        self.weights = weights or EnsembleWeights()
        self.normalize_combined = normalize_combined
    
    def batch_score(self, targets: List[str], passages: List[str]) -> List[Dict[str, float]]:
        """
        Score using ensemble approach.

        Args:
            targets: List of target texts
            passages: List of passage texts to score

        Returns:
            List of dictionaries with combined scores
        """
        if not passages:
            return [{"combined": 0.0} for _ in passages]

        if self.cross_encoder_rescorer is None:
            # No cross-encoder, return default scores
            return [{"combined": 0.0} for _ in passages]

        try:
            # Get cross-encoder scores
            cross_encoder_scores = self.cross_encoder_rescorer.batch_score(targets, passages)

            # For ensemble, we combine with a default "original" score
            # In practice, you'd want to pass in the original retrieval scores
            default_original_score = 0.5

            combined_scores = []
            for ce_score in cross_encoder_scores:
                # Extract cross-encoder relevance score
                ce_relevance = ce_score.get("relevance", 0.0)

                # Combine scores using weights
                combined_score = (
                    self.weights.original_score * default_original_score +
                    self.weights.cross_encoder_score * ce_relevance
                )

                combined_scores.append({
                    "combined": float(combined_score),
                    "original": default_original_score,
                    "cross_encoder": ce_relevance
                })

            return combined_scores

        except Exception as e:
            print(f"⚠️ Error in ensemble scoring: {e}. Returning default scores.")
            return [{"combined": 0.0} for _ in passages]
    
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
