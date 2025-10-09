from __future__ import annotations

import pytest

from efi_analyser.frames.types import Frame, FrameSchema
from efi_analyser.frames.types import FrameAssignment
from efi_analyser.frames.classifier.dataset import (
    FrameLabelSet,
    FrameLabeledPassage,
    FrameTorchDataset,
    build_label_matrix,
)


def make_schema() -> FrameSchema:
    return FrameSchema(
        domain="demo",
        notes="",
        frames=[
            Frame(frame_id="F1", name="Economic", description="", keywords=[], examples=[]),
            Frame(frame_id="F2", name="Environmental", description="", keywords=[], examples=[]),
        ],
    )


def make_assignments() -> list[FrameAssignment]:
    return [
        FrameAssignment(
            passage_id="p1",
            passage_text="Coal brings jobs",
            probabilities={"F1": 0.9, "F2": 0.1},
            top_frames=["F1"],
            rationale="jobs",
            evidence_spans=["jobs"],
        ),
        FrameAssignment(
            passage_id="p2",
            passage_text="Coal hurts health",
            probabilities={"F1": 0.2, "F2": 0.8},
            top_frames=["F2"],
            rationale="health",
            evidence_spans=["health"],
        ),
    ]


def test_frame_label_set_from_assignments():
    schema = make_schema()
    assignments = make_assignments()
    label_set = FrameLabelSet.from_assignments(schema, assignments, source="llm")

    assert label_set.label_order == ["F1", "F2"]
    matrix, order = label_set.to_numpy()
    assert matrix.shape == (2, 2)
    assert list(order) == ["F1", "F2"]
    assert matrix[0, 0] == pytest.approx(0.9, rel=1e-5)
    assert matrix[1, 1] == pytest.approx(0.8, rel=1e-5)

    # Pandas optional
    try:
        df = label_set.to_dataframe()
        assert "prob_F1" in df.columns
    except ModuleNotFoundError:
        pass  # pandas not available


def test_build_label_matrix_custom_order():
    schema = make_schema()
    assignments = make_assignments()
    label_set = FrameLabelSet.from_assignments(schema, assignments, source="llm")
    matrix = build_label_matrix(label_set, label_order=["F2", "F1"])
    assert matrix.shape == (2, 2)
    assert matrix[0, 0] == pytest.approx(0.1, rel=1e-5)
    assert matrix[1, 1] == pytest.approx(0.2, rel=1e-5)


def test_frame_torch_dataset():
    schema = make_schema()
    assignments = make_assignments()
    label_set = FrameLabelSet.from_assignments(schema, assignments, source="llm")
    try:
        dataset = FrameTorchDataset(label_set)
    except ModuleNotFoundError:
        return  # torch not installed
    text, labels = dataset[0]
    assert isinstance(text, str)
    assert labels.shape[-1] == label_set.num_frames
