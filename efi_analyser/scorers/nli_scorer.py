"""
Abstract NLI scorer base class.
"""

from abc import ABC
from typing import List, Dict

from efi_core.types import PairScorer, Task


class NLIScorer(PairScorer, ABC):
    """Abstract base class for NLI (Natural Language Inference) scorers.

    All NLI scorers must output probabilities for the three NLI classes:
    - entails: premise logically implies hypothesis
    - contradicts: premise contradicts hypothesis
    - neutral: no clear entailment or contradiction

    Example output:
    [
        {"entails": 0.8, "contradicts": 0.15, "neutral": 0.05},
        {"entails": 0.2, "contradicts": 0.7, "neutral": 0.1}
    ]
    """

    def __init__(self, name: str, config=None):
        """Initialize NLI scorer.

        Args:
            name: Unique name for this scorer
            config: Configuration object (implementation-specific)
        """
        super().__init__(name, Task.NLI)
        self.config = config

    def batch_score(self, targets: List[str], passages: List[str]) -> List[Dict[str, float]]:
        """Score target-passage pairs for NLI relationship.

        This is the main interface that all NLI scorers must implement.

        Args:
            targets: List of premise texts
            passages: List of hypothesis texts

        Returns:
            List of dictionaries with NLI class probabilities.
            Each dict must contain keys: "entails", "contradicts", "neutral"
            Probabilities should sum to approximately 1.0
        """
        raise NotImplementedError("Subclasses must implement batch_score")

    def validate_output_format(self, predictions: List[Dict[str, float]]) -> bool:
        """Validate that predictions follow the required NLI format.

        Args:
            predictions: List of prediction dictionaries

        Returns:
            True if format is valid, False otherwise
        """
        required_keys = {"entails", "contradicts", "neutral"}

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