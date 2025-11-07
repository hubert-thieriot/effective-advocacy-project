"""Tests for frame induction pipeline."""

from __future__ import annotations

import json

import pytest

from efi_analyser.frames.induction import FrameInducer


class StubLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.messages = None

    def infer(self, messages):
        self.messages = messages
        return self.response


class SequencedLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []
        self.timeouts: list[float | None] = []

    def infer(self, messages, timeout=None):
        self.calls.append(messages)
        self.timeouts.append(timeout)
        if not self.responses:
            raise AssertionError("SequencedLLM exhausted responses")
        return self.responses.pop(0)


class TimeoutAwareLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.messages = None
        self.timeout = None

    def infer(self, messages, timeout=None):
        self.messages = messages
        self.timeout = timeout
        return self.response


@pytest.fixture
def sample_response():
    payload = {
        "domain": "energy transition",
        "frames": [
            {
                "frame_id": "F1",
                "name": "Economic Opportunity",
                "description": "Focus on jobs and investment potential",
                "keywords": ["jobs", "investment"],
                "examples": ["The energy shift will create 1M jobs."],
            },
            {
                "frame_id": "F2",
                "name": "Security Risk",
                "description": "Highlights dependence on foreign actors",
                "keywords": ["security"],
                "examples": ["Reliance on imports weakens autonomy."],
            },
        ],
        "notes": "Ensure regional diversity.",
    }
    return json.dumps(payload)


def test_induce_parses_and_returns_schema(sample_response):
    llm = StubLLM(response=sample_response)
    inducer = FrameInducer(llm_client=llm, domain="energy transition", frame_target=6)

    passages = [
        "   First passage about green jobs   ",
        "Second passage about national security",
        "First passage about green jobs",  # duplicate
        "Third passage" * 300,  # overly long
    ]

    schema = inducer.induce(passages)

    # LLM called with system + user messages
    assert llm.messages is not None
    assert llm.messages[0]["role"] == "system"
    assert "Discover distinct media frames" in llm.messages[0]["content"]

    user_content = llm.messages[1]["content"]
    # Deduplicated and enumerated
    assert "1. First passage about green jobs" in user_content
    assert "2. Second passage about national security" in user_content
    assert "Frame target: 6 frames" in user_content
    # Long passage should be truncated with ellipsis marker
    assert "3. Third passage" in user_content
    assert "..." in user_content

    assert schema.domain == "energy transition"
    assert len(schema.frames) == 2
    assert schema.frames[0].frame_id == "F1"
    assert schema.notes == "Ensure regional diversity."


def test_induce_accepts_textual_frame_target(sample_response):
    llm = StubLLM(response=sample_response)
    inducer = FrameInducer(
        llm_client=llm,
        domain="energy transition",
        frame_target="between 5 and 20",
        induction_guidance="Include a frame on overcapacity",
    )

    inducer.induce(["Passage about policy targets"])

    user_content = llm.messages[1]["content"]
    assert "Frame target: between 5 and 20" in user_content
    assert "GUIDANCE: Include a frame on overcapacity" in user_content


def test_timeout_forwarded_when_supported(sample_response):
    llm = TimeoutAwareLLM(response=sample_response)
    inducer = FrameInducer(
        llm_client=llm,
        domain="energy transition",
        frame_target=4,
        infer_timeout=900,
    )

    inducer.induce(["Passage about resilience"])

    assert llm.timeout == 900
    assert llm.messages[0]["role"] == "system"


def test_induce_raises_on_invalid_json():
    llm = StubLLM(response="not json")
    inducer = FrameInducer(llm_client=llm, domain="energy transition")

    with pytest.raises(ValueError):
        inducer.induce(["passage"])


def test_induce_requires_passages():
    llm = StubLLM(response="{}")
    inducer = FrameInducer(llm_client=llm, domain="energy transition")

    with pytest.raises(ValueError):
        inducer.induce([])


def test_induce_handles_multiple_batches(sample_response):
    partial_response_one = json.dumps(
        {
            "domain": "energy transition",
            "frames": [
                {
                    "frame_id": "A1",
                    "name": "Economic Benefits",
                    "description": "Highlights job growth and investments.",
                    "keywords": ["jobs", "investment"],
                    "examples": ["Passage on jobs."],
                }
            ],
            "notes": "batch one",
        }
    )
    partial_response_two = json.dumps(
        {
            "domain": "energy transition",
            "frames": [
                {
                    "frame_id": "B1",
                    "name": "Environmental Harm",
                    "description": "Focus on pollution and health impacts.",
                    "keywords": ["pollution", "health"],
                    "examples": ["Passage on pollution."],
                }
            ],
            "notes": "batch two",
        }
    )
    merged_response = json.dumps(
        {
            "domain": "energy transition",
            "frames": [
                {
                    "frame_id": "F1",
                    "name": "Economic Benefits",
                    "description": "Jobs and investment narratives.",
                    "keywords": ["jobs", "investment"],
                    "examples": ["Passage on jobs."],
                },
                {
                    "frame_id": "F2",
                    "name": "Environmental Harm",
                    "description": "Pollution and health narratives.",
                    "keywords": ["pollution", "health"],
                    "examples": ["Passage on pollution."],
                },
            ],
            "notes": "merged",
        }
    )

    llm = SequencedLLM([partial_response_one, partial_response_two, merged_response])
    inducer = FrameInducer(
        llm_client=llm,
        domain="energy transition",
        frame_target="between 5 and 10",
        max_passages_per_call=2,
        max_total_passages=10,
    )

    passages = [
        "Economic passage one",
        "Economic passage two",
        "Environmental passage one",
        "Environmental passage two",
    ]

    schema = inducer.induce(passages)

    assert len(llm.calls) == 3  # two batches + merge
    assert schema.domain == "energy transition"
    assert {frame.name for frame in schema.frames} == {
        "Economic Benefits",
        "Environmental Harm",
    }
