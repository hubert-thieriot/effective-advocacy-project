"""Tests for the LLM-based frame annotator."""

from __future__ import annotations

import os
import json
from typing import Any, Dict, List

import pytest

from efi_analyser.frames.annotator import LLMFrameAnnotator
from efi_analyser.frames.types import Frame, FrameSchema

SYSTEM_TEMPLATE = """You are an expert analyst that assigns frame probabilities to passages.
Always respond with valid JSON (array of objects)."""

USER_TEMPLATE = """FRAME SCHEMA
================
{{ frames_text }}

PASSAGES
========
{{ passages_text }}

RESPONSE FORMAT
===============
Return a JSON array. Each entry must look like:
{
  "passage_id": "...",
  "probs": {"<frame_id>": <float between 0 and 1>},
  "top_frames": ["<frame_id>", ...],
  "rationale": "<why>",
  "evidence_spans": ["<snippet>", ...]
}

Only include frame_ids present in the schema. Provide probabilities for every frame_id.
"""

# Prefer slightly smarter models for better frame assignment
SMART_OLLAMA_MODELS = ("llama3.2:3b", "phi3:3.8b", "llama3.2:1b")
FAST_OLLAMA_MODELS = ("llama3.2:1b", "llama3.2:3b", "phi3:3.8b")
RUN_OLLAMA_TESTS = os.getenv("RUN_OLLAMA_TESTS") == "1"


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


def build_annotator(llm_client: Any, **kwargs: Any) -> LLMFrameAnnotator:
    """Helper to construct annotators with deterministic prompt templates."""
    return LLMFrameAnnotator(
        llm_client=llm_client,
        system_template=SYSTEM_TEMPLATE,
        user_template=USER_TEMPLATE,
        **kwargs,
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
    applicator = build_annotator(llm, batch_size=2, max_chars_per_passage=500)

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
    applicator = build_annotator(llm, batch_size=4, max_chars_per_passage=500)
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
    applicator_a = build_annotator(
        llm_client=llm_a,
        batch_size=4,
        cache=shared_cache,
        max_chars_per_passage=500,
    )
    applicator_a.batch_assign(schema, passages)

    llm_b = StubLLM([batch], spec_value="model-B")
    applicator_b = build_annotator(
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
    applicator = build_annotator(
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


def test_missing_passage_response_is_reported(capsys: pytest.CaptureFixture[str]):
    batch = json.dumps(
        [
            {"passage_id": "unexpected", "probs": {"finance": 1.0, "security": 0.0}}
        ]
    )
    llm = StubLLM([batch])
    applicator = build_annotator(llm, batch_size=4)

    assignments = applicator.batch_assign(make_schema(), [("p0", "text")])
    assert assignments == []
    out, err = capsys.readouterr()
    assert "missing 1 passages" in out


@pytest.mark.llm
def test_batch_assignment_with_ollama_smoke():
    """Integration test: uses Ollama locally, StubLLM on CI."""
    # Detect CI environment (GitHub Actions, GitLab CI, etc.)
    is_ci = os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true"
    
    if is_ci:
        # On CI, use stub LLM with realistic response
        response = json.dumps([
            {
                "passage_id": "ollama_p0",
                "probs": {"security": 0.85, "finance": 0.15},
                "top_frames": ["security"],
                "rationale": "Passage emphasizes grid reliability and preventing blackouts, which aligns with energy security frame.",
                "evidence_spans": ["grid reliability", "preventing blackouts"],
            }
        ])
        client = StubLLM([response], spec_value="ci-stub-llm")
    else:
        # Local: use Ollama if enabled
        if not RUN_OLLAMA_TESTS:
            pytest.skip("Set RUN_OLLAMA_TESTS=1 to enable Ollama integration test.")

        from efi_analyser.scorers.ollama_interface import OllamaConfig, OllamaInterface

        base_url = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        if not OllamaInterface.is_available(base_url):
            pytest.skip(f"Ollama daemon not reachable at {base_url}")

        # Use a slightly smarter model for better frame assignment
        available_models = OllamaInterface.list_models(base_url)
        chosen_model = None
        for candidate in SMART_OLLAMA_MODELS:
            if candidate in available_models:
                chosen_model = candidate
                break
        if not chosen_model:
            pytest.skip(f"No suitable models found. Available: {available_models}")

        config = OllamaConfig(
            model=chosen_model,
            base_url=base_url,
            timeout=60.0,  # Allow more time for slightly larger models
            preferred_models=SMART_OLLAMA_MODELS,
            verbose=bool(os.getenv("OLLAMA_VERBOSE")),
        )
        client = OllamaInterface(name="frame_annotator_test", config=config)

    annotator = build_annotator(
        client,
        batch_size=1,
        max_chars_per_passage=None,
        chunk_overlap_chars=0,
    )

    schema = make_schema()
    # Use a passage that clearly matches one of the frames (security - grid/reliability)
    passages = [
        ("ollama_p0", "Coal power plants ensure grid reliability during peak demand periods, preventing blackouts."),
    ]

    assignments = annotator.batch_assign(schema, passages, top_k=2, show_progress=False)
    assert len(assignments) == len(passages), "Expected to get assignments back"
    assert assignments[0].passage_id == "ollama_p0"
    
    # Check that we got probabilities (should be non-zero for at least one frame)
    probs = assignments[0].probabilities
    assert probs is not None, "Expected probabilities dict"
    assert len(probs) == 2, f"Expected probabilities for both frames, got {probs}"
    
    # For a passage about grid reliability, we expect security frame to have higher probability
    security_prob = probs.get("security", 0.0)
    finance_prob = probs.get("finance", 0.0)
    assert security_prob > 0.0, f"Expected security frame to have non-zero probability for grid reliability passage, got {probs}"
    assert security_prob > finance_prob, f"Expected security probability ({security_prob}) > finance probability ({finance_prob}) for grid reliability passage"
