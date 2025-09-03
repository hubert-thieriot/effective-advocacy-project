"""
Validation module for evaluating scorer performance.

Provides datasets, evaluation functions, and performance metrics for
different types of scorers (NLI, Stance, etc.).
"""

from .types import ValidationDataset, ValidationSample, TaskType, EvaluationResult
from .datasets import NLIDataset, StanceDataset
from .evaluator import ScorerEvaluator, EvaluationRunner
from .metrics import ClassificationMetrics

__all__ = [
    'ValidationDataset',
    'ValidationSample',
    'TaskType',
    'EvaluationResult',
    'NLIDataset',
    'StanceDataset',
    'ScorerEvaluator',
    'EvaluationRunner',
    'ClassificationMetrics'
]
