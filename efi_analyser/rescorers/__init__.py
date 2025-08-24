"""
Re-scoring module for improving retrieval quality.

Provides pluggable components for re-scoring retrieved results using
cross-encoders and ensemble methods.
"""

from .cross_encoder_rescorer import CrossEncoderReScorer
from .ensemble_rescorer import EnsembleReScorer, EnsembleWeights
from .nli_rescorer import NLIReScorer, NLIReScorerConfig

__all__ = [
    'CrossEncoderReScorer',
    'EnsembleReScorer',
    'EnsembleWeights',
    'NLIReScorer',
    'NLIReScorerConfig'
]
