"""
Stance scorer using HuggingFace models.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel

from .stance_scorer import StanceScorer


class StanceHFScorerConfig(BaseModel):
    """Configuration for StanceHFScorer.

    This is a placeholder for future HuggingFace stance models.
    Currently, stance detection typically requires fine-tuned models.
    """

    model_name: str = "microsoft/DialoGPT-medium"  # Placeholder
    batch_size: int = 8
    device: int = -1
    max_length: int = 384
    local_files_only: bool = False


class StanceHFScorer(StanceScorer):
    """Stance scorer using HuggingFace models.

    NOTE: This is a placeholder implementation. Stance detection typically
    requires domain-specific fine-tuned models that are not readily available
    as general-purpose models like NLI models.

    For production use, you would need:
    1. A stance detection dataset
    2. Fine-tuned model on that dataset
    3. Proper label mapping for the task
    """

    def __init__(self, name: str = "stance_hf", config: Optional[StanceHFScorerConfig] = None) -> None:
        super().__init__(name, config)
        self.config = config or StanceHFScorerConfig()
        # Note: Implementation would require actual stance detection model
        print(f"⚠️ {name}: StanceHFScorer is a placeholder. Real implementation requires fine-tuned stance model.")

    def batch_score(self, targets: List[str], passages: List[str]) -> List[Dict[str, float]]:
        """Score target-passage pairs for stance.

        Placeholder implementation - returns neutral scores.
        """
        print("⚠️ StanceHFScorer: Using placeholder implementation. Returns neutral scores.")
        return [{"pro": 0.0, "anti": 0.0, "neutral": 1.0, "uncertain": 0.0} for _ in passages]
