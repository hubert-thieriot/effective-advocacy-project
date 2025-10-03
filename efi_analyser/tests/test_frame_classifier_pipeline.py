from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from efi_analyser.frames.types import Frame, FrameAssignment, FrameSchema
from efi_analyser.frames.classifier.model import FrameClassifierSpec
from efi_analyser.frames.classifier.pipeline import (
    ApplicationConfig,
    FrameClassifierArtifacts,
    FrameClassifierPipeline,
    InductionConfig,
    SplitConfig,
)


class StubLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def infer(self, messages, timeout=None):
        self.calls.append({"messages": messages, "timeout": timeout})
        if not self.responses:
            raise AssertionError("StubLLM exhausted responses")
        return self.responses.pop(0)

    def spec_key(self):
        return "stub-llm"


class FakeCorpus:
    def __init__(self) -> None:
        self._chunks = {
            "doc1": [SimpleNamespace(chunk_id=0, text="Coal brings jobs"), SimpleNamespace(chunk_id=1, text="Coal costs rise")],
            "doc2": [SimpleNamespace(chunk_id=0, text="Pollution increases asthma"), SimpleNamespace(chunk_id=1, text="Coal alternatives grow")],
        }

    def list_ids(self):
        return list(self._chunks.keys())

    def get_chunks(self, doc_id, materialize_if_necessary=False):
        return self._chunks.get(doc_id, [])


class FakeEmbeddedCorpus:
    def __init__(self) -> None:
        self.corpus = FakeCorpus()

    def get_chunks(self, doc_id, materialize_if_necessary=False):
        return self.corpus.get_chunks(doc_id, materialize_if_necessary=materialize_if_necessary)


def make_schema() -> FrameSchema:
    return FrameSchema(
        domain="demo",
        frames=[
            Frame(frame_id="F1", name="Economic", description="", keywords=["jobs"], examples=[]),
            Frame(frame_id="F2", name="Health", description="", keywords=["asthma"], examples=[]),
        ],
        notes="",
        schema_id="demo_schema",
    )


class DummyTrainer:
    def __init__(self):
        self.train_calls = []

    def train(self, label_set, eval_set=None, compute_metrics=None):
        self.train_calls.append((label_set, eval_set))

        class Model:
            def __init__(self, schema, label_order):
                self.schema = schema
                self.label_order = label_order

        return Model(label_set.schema, label_set.label_order)


class FakeSampler:
    def __init__(self, passages):
        self.passages = passages
        self.calls = []

    def collect(self, config):
        self.calls.append(config)
        return list(self.passages)


def test_induce_schema_from_corpus_uses_llm():
    embedded = FakeEmbeddedCorpus()
    induction_passages = [
        ("doc1:chunk000", "Coal brings jobs"),
        ("doc2:chunk000", "Coal hurts health"),
    ]
    sampler = FakeSampler(induction_passages)
    schema_response = json.dumps(
        {
            "domain": "demo",
            "frames": [
                {"frame_id": "F1", "name": "Economic", "description": "", "keywords": ["jobs"], "examples": []},
                {"frame_id": "F2", "name": "Health", "description": "", "keywords": ["asthma"], "examples": []},
            ],
        }
    )
    llm = StubLLM([schema_response])

    pipeline = FrameClassifierPipeline(
        embedded_corpus=embedded,
        domain="demo",
        inducer_client=llm,
        applicator_client=StubLLM([]),
        classifier_spec=FrameClassifierSpec(model_name="dummy"),
        sampler=sampler,
        trainer=DummyTrainer(),
        induction_config=InductionConfig(sample_size=2, seed=1),
        application_config=ApplicationConfig(sample_size=0),
    )

    schema, collected = pipeline._build_schema(schema_override=None, passages_override=None)
    assert isinstance(schema, FrameSchema)
    assert len(collected) == 2
    assert len(schema.frames) == 2
    assert llm.calls


def test_apply_frames_to_passages_returns_assignments():
    schema = make_schema()
    passages = [("doc1:chunk000", "Coal brings jobs"), ("doc2:chunk000", "Pollution increases asthma")]
    response = json.dumps(
        [
            {
                "passage_id": passages[0][0],
                "probs": {"F1": 0.9, "F2": 0.1},
                "top_frames": ["F1"],
            },
            {
                "passage_id": passages[1][0],
                "probs": {"F1": 0.2, "F2": 0.8},
                "top_frames": ["F2"],
            },
        ]
    )
    llm = StubLLM([response])
    pipeline = FrameClassifierPipeline(
        embedded_corpus=FakeEmbeddedCorpus(),
        domain="demo",
        inducer_client=StubLLM([]),
        applicator_client=llm,
        classifier_spec=FrameClassifierSpec(model_name="dummy"),
        sampler=FakeSampler(passages),
        trainer=DummyTrainer(),
        application_config=ApplicationConfig(sample_size=len(passages), batch_size=2, seed=2),
        induction_config=InductionConfig(sample_size=0),
    )

    collected, assignments = pipeline._label_passages(
        schema,
        passages,
        passages_override=passages,
        assignments_override=None,
    )
    assert len(assignments) == len(passages)
    assert collected == passages
    assert all(isinstance(item, FrameAssignment) for item in assignments)


def test_run_pipeline_with_stub_training():
    embedded = FakeEmbeddedCorpus()
    schema = make_schema()

    passages = [
        ("doc1:chunk000", "Coal brings jobs"),
        ("doc2:chunk000", "Pollution increases asthma"),
    ]
    response = json.dumps(
        [
            {
                "passage_id": passages[0][0],
                "probs": {"F1": 0.85, "F2": 0.15},
                "top_frames": ["F1"],
            },
            {
                "passage_id": passages[1][0],
                "probs": {"F1": 0.25, "F2": 0.75},
                "top_frames": ["F2"],
            },
        ]
    )
    applicator_llm = StubLLM([response])
    dummy_trainer = DummyTrainer()

    pipeline = FrameClassifierPipeline(
        embedded_corpus=embedded,
        domain="demo",
        inducer_client=None,
        applicator_client=applicator_llm,
        classifier_spec=FrameClassifierSpec(model_name="dummy"),
        induction_config=InductionConfig(sample_size=0),
        application_config=ApplicationConfig(
            sample_size=len(passages),
            seed=3,
            batch_size=2,
            exclude_induction_passages=False,
        ),
        split_config=SplitConfig(train_ratio=0.5, dev_ratio=0.0, seed=5),
        trainer=dummy_trainer,
    )

    artifacts = pipeline.run(
        schema_override=schema,
        induction_passages_override=passages,
        application_passages_override=passages,
    )

    assert isinstance(artifacts, FrameClassifierArtifacts)
    assert dummy_trainer.train_calls
    assert artifacts.dev_set is None
    assert len(artifacts.assignments) == len(passages)
    assert artifacts.train_set.schema == schema
