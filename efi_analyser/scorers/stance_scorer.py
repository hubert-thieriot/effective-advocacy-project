"""
Abstract Stance scorer base class.

This defines the interface that all Stance scorers must implement.
"""

from __future__ import annotations

from abc import ABC
from typing import List, Dict

from efi_core.types import PairScorer, Task


class StanceScorer(PairScorer, ABC):
    """Abstract base class for Stance detection scorers.

    All Stance scorers must output probabilities for the stance classes:
    - pro: expresses support for the target
    - anti: expresses opposition to the target
    - neutral: neutral or balanced position
    - uncertain: unclear or ambiguous stance

    Example output:
    [
        {"pro": 0.7, "anti": 0.2, "neutral": 0.05, "uncertain": 0.05},
        {"pro": 0.1, "anti": 0.8, "neutral": 0.05, "uncertain": 0.05}
    ]
    """

    def __init__(self, name: str, config=None):
        """Initialize Stance scorer.

        Args:
            name: Unique name for this scorer
            config: Configuration object (implementation-specific)
        """
        super().__init__(name, Task.STANCE)
        self.config = config

    def batch_score(self, targets: List[str], passages: List[str]) -> List[Dict[str, float]]:
        """Score target-passage pairs for stance relationship.

        This is the main interface that all Stance scorers must implement.

        Args:
            targets: List of target topics/claims
            passages: List of texts expressing stance toward targets

        Returns:
            List of dictionaries with stance class probabilities.
            Each dict must contain keys: "pro", "anti", "neutral", "uncertain"
            Probabilities should sum to approximately 1.0
        """
        raise NotImplementedError("Subclasses must implement batch_score")

    def validate_output_format(self, predictions: List[Dict[str, float]]) -> bool:
        """Validate that predictions follow the required Stance format.

        Args:
            predictions: List of prediction dictionaries

        Returns:
            True if format is valid, False otherwise
        """
        required_keys = {"pro", "anti", "neutral", "uncertain"}

        for pred in predictions:
            if not isinstance(pred, dict):
                return False
            if set(pred.keys()) != required_keys:
                return False
            if not all(isinstance(v, (int, float)) for v in pred.values()):
                return False
            if not all(0.0 <= v <= 1.0 for v in pred.values()):
                return False

        return True
