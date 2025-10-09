"""End-to-end frame induction tests against live LLMs."""

from __future__ import annotations

import os
import re
import pytest

from efi_analyser.frames.induction import FrameInducer
from efi_analyser.scorers.openai_interface import OpenAIConfig, OpenAIInterface


# @pytest.mark.llm
def test_frame_induction_with_openai() -> None:
    
    # Optional local alternative for slower runs:
    # model = os.getenv("OLLAMA_MODEL", "mistral:7b")
    # client = OllamaChatClient(model=model)
    # schema = FrameInducer(client, "India coal narratives", frame_target="roughly 10").induce(passages)
    
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not configured; skipping live LLM test")

    model = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
    client = OpenAIInterface(
        name="frame_induction_e2e",
        config=OpenAIConfig(model=model, temperature=0.0, timeout=600.0),
    )

    passages = [
        "Coal keeps India's grid stable during peak summer demand and supports energy security.",
        "Rapid coal expansion threatens India's climate commitments due to rising emissions.",
        "Coal mining communities rely on the industry for jobs, wages, and local economic activity.",
        "Coal-fired power plants contribute to severe air pollution and respiratory disease in Indian cities.",
        "New solar investments challenge coal's dominance while promising cleaner growth and investment.",
        "Financial analysts warn that stranded coal assets could burden public banks and taxpayers.",
        "Local protests highlight land displacement and safety risks tied to new coal mining projects.",
        "Innovations in battery storage raise doubts about the need for additional coal capacity.",
        "Industrial users argue reliable coal supply keeps manufacturing competitive internationally.",
        "Environmental groups push for stricter enforcement of emissions norms at legacy coal plants.",
    ]

    inducer = FrameInducer(
        llm_client=client,
        domain="India coal narratives",
        frame_target="roughly 4",
        infer_timeout=600.0,
    )
    schema = inducer.induce(passages)

    assert schema.domain == "India coal narratives"
    assert abs(len(schema.frames)-4) <= 1
    for frame in schema.frames:
        assert frame.frame_id
        assert frame.name
        assert isinstance(frame.examples, list)
        assert isinstance(frame.keywords, list)
        
    # Check content
    # At least these three frames: Economy, Security, Environment|Pollution
    economy_frame = next((frame for frame in schema.frames if "econom" in frame.name.lower()), None)
    security_frame = next((frame for frame in schema.frames if "security" in frame.name.lower()), None)
    environment_frame = next((frame for frame in schema.frames if re.search(r"environment|pollution", frame.name.lower())), None)
    
    assert economy_frame is not None
    assert security_frame is not None
    assert environment_frame is not None
    
    # Check they have different ids
    assert len(set([economy_frame.frame_id, security_frame.frame_id, environment_frame.frame_id])) == 3

   
