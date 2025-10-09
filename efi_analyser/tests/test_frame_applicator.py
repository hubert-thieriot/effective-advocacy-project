"""Tests for the LLM-based frame applicator."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from efi_analyser.frames.applicator import LLMFrameApplicator
from efi_analyser.frames.types import Frame, FrameSchema


class StubLLM:
    def __init__(self, responses: List[str], spec_value: str = "stub-llm") -> None:
        self.responses = responses
        self.calls: List[Dict[str, Any]] = []
        self._spec_value = spec_value

    def infer(self, messages, timeout=None):
        self.calls.append({"messages": messages, "timeout": timeout})
        if not self.responses:
            raise AssertionError("StubLLM exhausted responses")
        return self.responses.pop(0)

    def spec_key(self) -> str:
        return self._spec_value


def make_schema() -> FrameSchema:
    return FrameSchema(
        domain="india coal",
        frames=[
            Frame(
                frame_id="finance",
                name="Financial Risk",
                description="Focus on cost, investment, and stranded assets",
                keywords=["investment", "debt"],
                examples=["Analysts warn of stranded coal plants."],
            ),
            Frame(
                frame_id="security",
                name="Energy Security",
                description="Reliability and self-sufficiency arguments",
                keywords=["grid", "reliability"],
                examples=["Coal keeps the grid stable."],
            ),
        ],
        notes="",
        schema_id="coal_india_demo",
    )


def test_batch_assignment_with_normalization_and_defaults():
    batch_one = json.dumps(
        [
            {
                "passage_id": "p0",
                "probs": {"finance": 0.6, "security": 0.3},
                "top_frames": ["finance"],
                "rationale": "Discusses financial burdens",
                "evidence_spans": ["coal loans", "investor risk"],
            },
            {
                "passage_id": "p1",
                "probs": {"finance": 0.1},  # missing frame + sum != 1
            },
        ]
    )
    batch_two = json.dumps(
        [
            {
                "passage_id": "p2",
                "probs": {"security": 0.9, "finance": 0.9},  # will be renormalized
                "top_frames": [],
                "rationale": 42,  # invalid type handled
                "evidence_spans": ["   "],  # stripped away
            }
        ]
    )

    llm = StubLLM([batch_one, batch_two])
    applicator = LLMFrameApplicator(llm_client=llm, batch_size=2, max_chars_per_passage=500)

    passages = [
        ("p0", "Banks fear new coal loans."),
        ("p1", "Village debates cost overruns."),
        ("p2", "Officials tout grid reliability."),
    ]

    assignments = applicator.batch_assign(make_schema(), passages, top_k=2)

    assert len(assignments) == 3
    assert assignments[0].passage_id == "p0"
    assert assignments[0].probabilities["finance"] == pytest.approx(0.6666666, rel=1e-6)
    assert assignments[0].probabilities["security"] == pytest.approx(0.3333333, rel=1e-6)
    assert assignments[0].top_frames == ["finance"]
    assert assignments[0].rationale.startswith("Discusses")
    assert assignments[0].evidence_spans == ["coal loans", "investor risk"]

    # Second entry fills missing frame and normalizes
    assert assignments[1].passage_id == "p1"
    probs_p1 = assignments[1].probabilities
    assert probs_p1["finance"] == pytest.approx(1.0)
    assert probs_p1["security"] == pytest.approx(0.0)
    assert assignments[1].top_frames == ["finance"]

    # Third entry renormalized and strips invalid fields
    probs_p2 = assignments[2].probabilities
    assert pytest.approx(sum(probs_p2.values()), rel=1e-6) == 1.0
    assert set(assignments[2].top_frames) == {"finance", "security"}
    assert assignments[2].rationale == ""
    assert assignments[2].evidence_spans == []

    # Two batches executed due to size=2
    assert len(llm.calls) == 2
    assert all(len(call["messages"][1]["content"].split("- passage_id")) >= 2 for call in llm.calls)


def test_cache_prevents_repeat_calls():
    batch = json.dumps(
        [
            {
                "passage_id": "p0",
                "probs": {"finance": 0.5, "security": 0.5},
                "top_frames": ["finance", "security"],
            }
        ]
    )

    llm = StubLLM([batch])
    applicator = LLMFrameApplicator(llm_client=llm, batch_size=4, max_chars_per_passage=500)
    schema = make_schema()
    passages = [("p0", "Coal debate in parliament.")]

    first = applicator.batch_assign(schema, passages)
    second = applicator.batch_assign(schema, passages)

    assert len(first) == 1 and len(second) == 1
    assert llm.calls[0]["timeout"] == pytest.approx(600.0)
    assert len(llm.calls) == 1  # cached prevents second call
    assert second[0].probabilities == first[0].probabilities


def test_shared_cache_respects_model_spec():
    shared_cache: Dict[str, Any] = {}
    schema = make_schema()
    passages = [("p0", "Coal debate in parliament.")]

    batch = json.dumps(
        [
            {
                "passage_id": "p0",
                "probs": {"finance": 0.5, "security": 0.5},
                "top_frames": ["finance", "security"],
            }
        ]
    )

    llm_a = StubLLM([batch], spec_value="model-A")
    applicator_a = LLMFrameApplicator(
        llm_client=llm_a,
        batch_size=4,
        cache=shared_cache,
        max_chars_per_passage=500,
    )
    applicator_a.batch_assign(schema, passages)

    llm_b = StubLLM([batch], spec_value="model-B")
    applicator_b = LLMFrameApplicator(
        llm_client=llm_b,
        batch_size=4,
        cache=shared_cache,
        max_chars_per_passage=500,
    )

    applicator_b.batch_assign(schema, passages)

    assert len(llm_a.calls) == 1
    assert len(llm_b.calls) == 1  # cache miss because spec differs
    assert any(key.startswith("model-A") for key in shared_cache)
    assert any(key.startswith("model-B") for key in shared_cache)


def test_long_passages_are_chunked_with_overlap():
    long_text = "Coal demand " * 80
    llm = StubLLM(["[]"])
    applicator = LLMFrameApplicator(
        llm_client=llm,
        batch_size=16,
        max_chars_per_passage=120,
        chunk_overlap_chars=20,
    )
    segments = applicator._split_passage("p0", long_text)
    payload = []
    for index, (pid, _) in enumerate(segments):
        finance = max(0.1, 0.7 - index * 0.1)
        security = 1.0 - finance
        payload.append(
            {
                "passage_id": pid,
                "probs": {"finance": finance, "security": security},
                "top_frames": ["finance", "security"],
            }
        )
    llm.responses = [json.dumps(payload)]

    assignments = applicator.batch_assign(make_schema(), [("p0", long_text)])

    assert len(assignments) == len(segments)
    assert assignments[0].passage_id == segments[0][0]
    assert all(len(a.passage_text) <= 120 for a in assignments)


def test_missing_passage_response_raises():
    batch = json.dumps(
        [
            {"passage_id": "unexpected", "probs": {"finance": 1.0, "security": 0.0}}
        ]
    )
    llm = StubLLM([batch])
    applicator = LLMFrameApplicator(llm_client=llm, batch_size=4)

    with pytest.raises(ValueError):
        applicator.batch_assign(make_schema(), [("p0", "text")])
