from __future__ import annotations

import os
import pytest

from efi_analyser.rescorers.llm_rescorer import LLMReScorerConfig
from efi_analyser.rescorers.ollama_rescorer import OllamaReScorer
from efi_core.retrieval.retriever import SearchResult


def _env_or_default(var: str, default: str) -> str:
    v = os.getenv(var)
    return v if v else default


@pytest.mark.integration
@pytest.mark.local_llm
def test_ollama_rescorer_integration_if_available(tmp_path):
    try:
        import openai  # noqa: F401
    except Exception:
        pytest.skip("openai package not installed; skipping OllamaReScorer integration test")

    base_url = _env_or_default("LLM_BASE_URL", "http://localhost:11434/v1")
    model = _env_or_default("LLM_MODEL", "gemma3:4b")

    config = LLMReScorerConfig(model=model, ignore_cache=True)
    # Use a temporary cache location
    config.cache_dir = tmp_path / "cache" / "rescorer"

    try:
        r = OllamaReScorer(config=config, base_url=base_url, api_key=os.getenv("LLM_API_KEY", "ollama"))
        query = "Coal consumption in India increased by 8% in 2023."
        matches = [
            SearchResult("doc1_chunk_0", 0.2, metadata={"text": "Government data shows coal consumption rose 8% in India in 2023."}),
            SearchResult("doc2_chunk_1", 0.9, metadata={"text": "Unrelated statement about different country and year."}),
        ]
        rescored = r.rescore(query, matches)
    except Exception as exc:
        pytest.skip(f"Local LLM server not reachable or model not available: {exc}")

    assert len(rescored) == 2
    for m in rescored:
        assert 0.0 <= m.score <= 1.0
        assert m.metadata.get("llm_model") == model
    
    assert rescored[0].score > rescored[1].score


if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    tmp_path = Path(tempfile.mkdtemp())
    test_ollama_rescorer_integration_if_available(tmp_path)