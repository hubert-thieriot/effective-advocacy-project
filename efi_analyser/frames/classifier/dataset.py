"""Data structures and utilities for frame classification datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # Torch is optional when only manipulating raw data.
    import torch
    from torch.utils.data import Dataset as TorchDataset
except ModuleNotFoundError:  # pragma: no cover - torch missing in some environments.
    torch = None
    TorchDataset = object  # type: ignore

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - pandas optional
    pd = None  # type: ignore

from efi_analyser.frames.types import FrameAssignment, FrameSchema


@dataclass
class FrameLabeledPassage:
    """A single passage paired with soft frame labels."""

    passage_id: str
    text: str
    soft_labels: Dict[str, float]
    hard_labels: List[str] = field(default_factory=list)
    rationale: str = ""
    evidence_spans: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameLabelSet:
    """Collection of labelled passages aligned to a frame schema."""

    schema: FrameSchema
    passages: List[FrameLabeledPassage]
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def label_order(self) -> List[str]:
        return [frame.frame_id for frame in self.schema.frames]

    @property
    def num_frames(self) -> int:
        return len(self.schema.frames)

    def to_numpy(self) -> Tuple[np.ndarray, List[str]]:
        """Return numpy arrays of soft labels ordered by the schema."""
        label_matrix = []
        for passage in self.passages:
            label_matrix.append([passage.soft_labels.get(fid, 0.0) for fid in self.label_order])
        return np.asarray(label_matrix, dtype=np.float32), self.label_order

    def to_dataframe(self) -> "pd.DataFrame":  # type: ignore[override]
        if pd is None:  # pragma: no cover - pandas optional
            raise ModuleNotFoundError("pandas is required for `to_dataframe` but is not installed")
        records: List[Dict[str, Any]] = []
        for passage in self.passages:
            row = {
                "passage_id": passage.passage_id,
                "text": passage.text,
                "rationale": passage.rationale,
                "evidence_spans": passage.evidence_spans,
                "hard_labels": passage.hard_labels,
            }
            for fid in self.label_order:
                row[f"prob_{fid}"] = passage.soft_labels.get(fid, 0.0)
            row.update({f"meta_{k}": v for k, v in passage.meta.items()})
            records.append(row)
        return pd.DataFrame.from_records(records)

    def split(
        self,
        train_ratio: float = 0.8,
        dev_ratio: float = 0.1,
        seed: int = 13,
    ) -> Tuple["FrameLabelSet", "FrameLabelSet", "FrameLabelSet"]:
        rng = np.random.default_rng(seed)
        indices = np.arange(len(self.passages))
        rng.shuffle(indices)

        n_train = int(len(indices) * train_ratio)
        n_dev = int(len(indices) * dev_ratio)
        train_idx = indices[:n_train]
        dev_idx = indices[n_train : n_train + n_dev]
        test_idx = indices[n_train + n_dev :]

        def subset(idxs: Iterable[int]) -> FrameLabelSet:
            return FrameLabelSet(
                schema=self.schema,
                passages=[self.passages[i] for i in idxs],
                source=self.source,
                metadata=self.metadata.copy(),
            )

        return subset(train_idx), subset(dev_idx), subset(test_idx)

    @classmethod
    def from_assignments(
        cls,
        schema: FrameSchema,
        assignments: Sequence[FrameAssignment],
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "FrameLabelSet":
        passages: List[FrameLabeledPassage] = []
        for assignment in assignments:
            passages.append(
                FrameLabeledPassage(
                    passage_id=assignment.passage_id,
                    text=assignment.passage_text,
                    soft_labels=assignment.probabilities,
                    hard_labels=assignment.top_frames,
                    rationale=assignment.rationale,
                    evidence_spans=assignment.evidence_spans,
                )
            )
        return cls(schema=schema, passages=passages, source=source, metadata=metadata or {})


class FrameTorchDataset(TorchDataset):
    """PyTorch dataset for frame classification."""

    def __init__(
        self,
        label_set: FrameLabelSet,
        return_text: bool = True,
    ) -> None:
        if torch is None:  # pragma: no cover - torch not available
            raise ModuleNotFoundError("torch must be installed to use FrameTorchDataset")

        self.label_set = label_set
        self.return_text = return_text
        labels, _ = label_set.to_numpy()
        self.tensor_labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return len(self.label_set.passages)

    def __getitem__(self, idx: int):
        passage = self.label_set.passages[idx]
        if self.return_text:
            return passage.text, self.tensor_labels[idx]
        return self.tensor_labels[idx]


def build_label_matrix(
    label_set: FrameLabelSet,
    label_order: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Utility to produce a label matrix using a custom order."""
    order = list(label_order) if label_order is not None else label_set.label_order
    matrix = []
    for passage in label_set.passages:
        matrix.append([passage.soft_labels.get(fid, 0.0) for fid in order])
    return np.asarray(matrix, dtype=np.float32)


__all__ = [
    "FrameLabeledPassage",
    "FrameLabelSet",
    "FrameTorchDataset",
    "build_label_matrix",
]
