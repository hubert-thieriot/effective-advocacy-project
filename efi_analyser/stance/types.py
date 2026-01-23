"""Data structures for stance detection workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json

import numpy as np


STANCE_LABELS: Tuple[str, str, str] = ("supports", "opposes", "neutral")


def build_stance_input(target: str, text: str) -> str:
    """Build model input text that binds a target to a passage."""
    return f"Target: {target}\nText: {text}"


@dataclass
class StanceResult:
    """Stance result for a single (chunk, target) pair."""

    chunk_id: str
    target: str
    text: str
    scores: Dict[str, float]
    label: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StancePaths:
    """Filesystem paths for stance artifacts."""

    annotation_path: Path
    classifier_dir: Path
    classifications_dir: Path


class StanceAssignments(List[StanceResult]):
    """Collection of stance results with JSON helpers."""

    def save(self, path: Path) -> None:
        serialized = [
            {
                "chunk_id": item.chunk_id,
                "target": item.target,
                "text": item.text,
                "scores": item.scores,
                "label": item.label,
                "metadata": item.metadata,
            }
            for item in self
        ]
        path.write_text(json.dumps(serialized, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "StanceAssignments":
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = cls()
        for item in payload:
            items.append(
                StanceResult(
                    chunk_id=item.get("chunk_id", ""),
                    target=item.get("target", ""),
                    text=item.get("text", ""),
                    scores=item.get("scores", {}) or {},
                    label=item.get("label", ""),
                    metadata=item.get("metadata", {}) or {},
                )
            )
        return items

    def to_label_set(
        self,
        *,
        labels: Sequence[str] = STANCE_LABELS,
    ) -> "StanceLabelSet":
        passages: List[StanceLabeledPassage] = []
        for item in self:
            passages.append(
                StanceLabeledPassage(
                    passage_id=item.chunk_id,
                    text=build_stance_input(item.target, item.text),
                    label=item.label,
                    target=item.target,
                    metadata=item.metadata,
                )
            )
        return StanceLabelSet(label_order=list(labels), passages=passages)


@dataclass
class StanceLabeledPassage:
    """Labeled stance training example."""

    passage_id: str
    text: str
    label: str
    target: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StanceLabelSet:
    """Collection of labeled stance examples."""

    label_order: List[str]
    passages: List[StanceLabeledPassage]

    def to_numpy(self) -> np.ndarray:
        label_to_id = {label: idx for idx, label in enumerate(self.label_order)}
        labels = [label_to_id.get(p.label, label_to_id.get("neutral", 0)) for p in self.passages]
        return np.asarray(labels, dtype=np.int64)

    def texts(self) -> List[str]:
        return [p.text for p in self.passages]


__all__ = [
    "STANCE_LABELS",
    "StanceResult",
    "StanceAssignments",
    "StanceLabeledPassage",
    "StanceLabelSet",
    "build_stance_input",
    "StancePaths",
]
