"""Stance detection components."""

from efi_analyser.stance.types import (
    STANCE_LABELS,
    StanceAssignments,
    StanceResult,
    StanceLabelSet,
    StanceLabeledPassage,
    StancePaths,
)
from efi_analyser.stance.detector import StanceDetector
from efi_analyser.stance.zero_shot import ZeroShotStanceDetector
from efi_analyser.stance.trained import TrainedStanceDetector
from efi_analyser.stance.annotator import LLMStanceAnnotator
from efi_analyser.stance.trainer import StanceClassifierSpec, StanceClassifierModel, StanceClassifierTrainer

__all__ = [
    "STANCE_LABELS",
    "StanceAssignments",
    "StanceResult",
    "StanceLabelSet",
    "StanceLabeledPassage",
    "StancePaths",
    "StanceDetector",
    "ZeroShotStanceDetector",
    "TrainedStanceDetector",
    "LLMStanceAnnotator",
    "StanceClassifierSpec",
    "StanceClassifierModel",
    "StanceClassifierTrainer",
]
