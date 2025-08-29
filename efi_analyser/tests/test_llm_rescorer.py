from __future__ import annotations

import json
from typing import Dict, List
import re

from efi_analyser.rescorers.llm_rescorer import LLMReScorer, LLMReScorerConfig
from efi_core.retrieval.retriever import SearchResult


class _MockLLMRescorer(LLMReScorer):
    """Mocked LLM rescoring that returns deterministic JSON based on simple rules."""

    def _inference(self, messages: List[Dict[str, str]]) -> str:
        # Inspect only the chunk text section to avoid matching prompt examples
        user = next(m for m in messages if m["role"] == "user")
        content = user["content"]
        m = re.search(r'chunk:\n\"([\s\S]*?)\"', content)
        chunk_text = m.group(1) if m else content
        if "rose 8%" in chunk_text and "India in 2023" in chunk_text:
            return json.dumps({"score": 0.95, "rationale": "Strong match."})
        return json.dumps({"score": 0.1, "rationale": "Weak or no support."})


def test_llm_rescorer_basic(tmp_path):
    config = LLMReScorerConfig()
    config.cache_dir = tmp_path / "cache" / "rescorer"
    r = _MockLLMRescorer(config)

    query = "Coal consumption in India increased by 8% in 2023."
    matches = [
        SearchResult("doc1_chunk_0", 0.2, metadata={"text": "Government data shows coal consumption rose 8% in India in 2023."}),
        SearchResult("doc2_chunk_1", 0.9, metadata={"text": "Unrelated statement about different country and year."}),
    ]

    rescored = r.rescore(query, matches)

    # Expect doc1 to rank first with a high score from LLM
    assert len(rescored) == 2
    assert rescored[0].item_id == "doc1_chunk_0"
    assert 0.9 <= rescored[0].score <= 1.0
    assert rescored[1].item_id == "doc2_chunk_1"
    assert 0.0 <= rescored[1].score <= 0.2

    # Verify cache materialization
    # Re-run and ensure identical outputs (cache hit path works)
    rescored_again = r.rescore(query, matches)
    assert [m.item_id for m in rescored_again] == [m.item_id for m in rescored]
    assert [round(m.score, 3) for m in rescored_again] == [round(m.score, 3) for m in rescored]


def test_llm_rescorer_batch_processing(tmp_path):
    """Test that batch processing works when configured."""
    config = LLMReScorerConfig(batch_size=2, max_workers=2)  # Small batches for testing
    config.cache_dir = tmp_path / "cache" / "rescorer"
    r = _MockLLMRescorer(config)

    query = "Coal consumption in India increased by 8% in 2023."
    matches = [
        SearchResult("doc1_chunk_0", 0.2, metadata={"text": "Government data shows coal consumption rose 8% in India in 2023."}),
        SearchResult("doc2_chunk_1", 0.9, metadata={"text": "Unrelated statement about different country and year."}),
        SearchResult("doc3_chunk_2", 0.3, metadata={"text": "Another unrelated statement."}),
        SearchResult("doc4_chunk_3", 0.4, metadata={"text": "Yet another unrelated statement."}),
    ]

    rescored = r.rescore(query, matches)

    # Should process in batches of 2
    assert len(rescored) == 4
    # First should be the high-scoring match
    assert rescored[0].item_id == "doc1_chunk_0"
    assert 0.9 <= rescored[0].score <= 1.0


def test_llm_rescorer_timing_metadata(tmp_path):
    """Test that timing metadata is included in rescored results."""
    config = LLMReScorerConfig()
    config.cache_dir = tmp_path / "cache" / "rescorer"
    r = _MockLLMRescorer(config)

    query = "Test finding."
    matches = [
        SearchResult("doc1", 0.5, metadata={"text": "Test chunk text."}),
    ]

    rescored = r.rescore(query, matches)

    assert len(rescored) == 1
    result = rescored[0]
    
    # Check that timing metadata is present
    assert "llm_scoring_time" in result.metadata
    assert isinstance(result.metadata["llm_scoring_time"], (int, float))
    assert result.metadata["llm_scoring_time"] >= 0.0
    
    # Check other metadata
    assert result.metadata["llm_model"] == config.model
    assert result.metadata["llm_score"] > 0.0
    assert "llm_rationale" in result.metadata


if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    tmp_path = Path(tempfile.mkdtemp())
    test_llm_rescorer_basic(tmp_path)