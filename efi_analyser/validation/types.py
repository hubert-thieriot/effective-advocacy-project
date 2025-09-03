"""
Core types for validation datasets and evaluation results.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Union
from pathlib import Path


class TaskType(Enum):
    """Types of validation tasks."""
    NLI = "nli"
    STANCE = "stance"


@dataclass
class ValidationSample:
    """A single validation sample with text pairs and gold standard labels."""

    sample_id: str
    text_a: str  # premise/query/finding
    text_b: str  # hypothesis/passage/evidence
    gold_label: str  # expected label (e.g., "entails", "pro", etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate that gold_label is valid for the task type."""
        if hasattr(self, '_task_type'):
            valid_labels = self._get_valid_labels(self._task_type)
            if self.gold_label not in valid_labels:
                raise ValueError(f"Invalid gold_label '{self.gold_label}' for task {self._task_type}. "
                               f"Valid labels: {valid_labels}")

    @staticmethod
    def _get_valid_labels(task_type: TaskType) -> List[str]:
        """Get valid labels for a task type."""
        if task_type == TaskType.NLI:
            return ["entails", "contradicts", "neutral"]
        elif task_type == TaskType.STANCE:
            return ["pro", "anti", "neutral", "uncertain"]
        else:
            return []


@dataclass
class ValidationDataset(ABC):
    """Abstract base class for validation datasets."""

    name: str
    task_type: TaskType
    samples: List[ValidationSample] = field(default_factory=list)

    def __post_init__(self):
        """Set task type on all samples."""
        for sample in self.samples:
            sample._task_type = self.task_type

    @abstractmethod
    def load_from_file(self, file_path: Union[str, Path]) -> None:
        """Load dataset from file."""
        pass

    @abstractmethod
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save dataset to file."""
        pass

    def add_sample(self, sample: ValidationSample) -> None:
        """Add a sample to the dataset."""
        sample._task_type = self.task_type
        self.samples.append(sample)

    def get_texts(self) -> tuple[List[str], List[str]]:
        """Get all text pairs for batch scoring."""
        texts_a = [s.text_a for s in self.samples]
        texts_b = [s.text_b for s in self.samples]
        return texts_a, texts_b

    def get_gold_labels(self) -> List[str]:
        """Get all gold standard labels."""
        return [s.gold_label for s in self.samples]


@dataclass
class EvaluationResult:
    """Results from evaluating a scorer on a dataset."""

    scorer_name: str
    dataset_name: str
    task_type: TaskType
    predictions: List[Dict[str, float]]  # List of label -> prob dicts
    gold_labels: List[str]
    metrics: Dict[str, float] = field(default_factory=dict)
    sample_results: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Validate predictions match gold labels."""
        if len(self.predictions) != len(self.gold_labels):
            raise ValueError(f"Predictions ({len(self.predictions)}) and gold labels "
                           f"({len(self.gold_labels)}) must have same length")

    @property
    def accuracy(self) -> float:
        """Get accuracy from metrics."""
        return self.metrics.get('accuracy', 0.0)

    @property
    def macro_f1(self) -> float:
        """Get macro F1 from metrics."""
        return self.metrics.get('macro_f1', 0.0)
