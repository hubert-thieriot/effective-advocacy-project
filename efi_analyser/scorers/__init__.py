"""
Scoring module for improving retrieval quality.

Provides pluggable components for scoring retrieved results using
cross-encoders and ensemble methods.
"""

from efi_core.types import PairScorer

# Task-specific abstract scorers
from .nli_scorer import NLIScorer
from .stance_scorer import StanceScorer

# Implementation-specific scorers
from .nli_hf_scorer import NLIHFScorer, NLIHFScorerConfig
from .nli_llm_scorer import NLILLMScorer, NLILLMScorerConfig
from .stance_hf_scorer import StanceHFScorer, StanceHFScorerConfig
from .stance_llm_scorer import StanceLLMScorer, StanceLLMScorerConfig

# Base implementations (used by task-specific scorers)
from .llm_scorer import LLMInterface, LLMScorerConfig

# Main NLI scorer implementation
from .nli_hf_scorer import NLIHFScorer, NLIHFScorerConfig

__all__ = [
    # Base classes
    'PairScorer',

    # Task-specific abstract scorers
    'NLIScorer',
    'StanceScorer',

    # NLI implementations
    'NLIHFScorer', 'NLIHFScorerConfig',
    'NLILLMScorer', 'NLILLMScorerConfig',

    # Stance implementations
    'StanceHFScorer', 'StanceHFScorerConfig',
    'StanceLLMScorer', 'StanceLLMScorerConfig',

    # Main NLI implementation
    'NLIHFScorer', 'NLIHFScorerConfig',

    # Base implementations
    'LLMInterface', 'LLMScorerConfig'
]
