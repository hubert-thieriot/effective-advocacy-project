from __future__ import annotations

import pytest

from efi_analyser.frames.types import Frame, FrameSchema, FrameAssignment
from efi_analyser.frames.classifier.dataset import FrameLabelSet
from efi_analyser.frames.classifier.model import FrameClassifierSpec
from efi_analyser.frames.classifier.trainer import FrameClassifierTrainer


@pytest.mark.parametrize("model_name", [
    "hf-internal-testing/tiny-random-distilbert"
])
def test_train_frame_classifier_smoke(model_name: str):
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")  # noqa: F841
    utils = pytest.importorskip("transformers.utils")
    if not getattr(utils, "is_accelerate_available", lambda: False)():
        pytest.skip("accelerate backend unavailable")

    schema = FrameSchema(
        domain="demo",
        notes="",
        frames=[
            Frame(frame_id="F1", name="Economic", description="", keywords=[], examples=[]),
            Frame(frame_id="F2", name="Environmental", description="", keywords=[], examples=[]),
        ],
    )
    assignments = [
        FrameAssignment(
            passage_id="p1",
            passage_text="Coal brings jobs and investment",
            probabilities={"F1": 0.9, "F2": 0.1},
            top_frames=["F1"],
        ),
        FrameAssignment(
            passage_id="p2",
            passage_text="Coal pollution hurts health",
            probabilities={"F1": 0.2, "F2": 0.8},
            top_frames=["F2"],
        ),
    ]
    label_set = FrameLabelSet.from_assignments(schema, assignments, source="llm")

    spec = FrameClassifierSpec(
        model_name=model_name,
        max_length=64,
        num_train_epochs=0.1,
        batch_size=2,
        learning_rate=5e-4,
        output_dir="tmp/frame_classifier_test",
        fp16=False,
    )

    trainer = FrameClassifierTrainer(spec)
    try:
        model = trainer.train(label_set)
    except OSError as exc:  # huggingface model download failed (no internet)
        pytest.skip(f"Could not download model {model_name}: {exc}")

    probs = model.predict_proba_batch([assignments[0].passage_text])
    assert set(probs[0].keys()) == {"F1", "F2"}
