import pytest

from efi_analyser.rescorers import NLIReScorer
from efi_core.retrieval.retriever import SearchResult


def test_nli_rescorer_rescores_with_mocked_pipeline():
    rescorer = NLIReScorer()

    def fake_pipeline(inputs, batch_size, truncation, return_all_scores):
        outputs = []
        for inp in inputs:
            if "cat" in inp["text"].lower():
                outputs.append([
                    {"label": "ENTAILMENT", "score": 0.9},
                    {"label": "CONTRADICTION", "score": 0.1},
                ])
            else:
                outputs.append([
                    {"label": "ENTAILMENT", "score": 0.2},
                    {"label": "CONTRADICTION", "score": 0.8},
                ])
        return outputs

    rescorer._pipeline = fake_pipeline
    matches = [
        SearchResult("a", 0.1, {"text": "A cat is on the mat."}),
        SearchResult("b", 0.2, {"text": "Dogs are everywhere."}),
    ]

    rescored = rescorer.rescore("A cat sits on the mat", matches)

    assert [r.item_id for r in rescored] == ["a", "b"]
    assert rescored[0].score == pytest.approx(0.9)
    assert rescored[1].score == pytest.approx(0.2)
    assert rescored[0].metadata["nli_score"] == pytest.approx(0.9)
    assert rescored[1].metadata["nli_score"] == pytest.approx(0.2)


def test_nli_rescorer_without_pipeline_returns_original():
    rescorer = NLIReScorer()
    rescorer._pipeline = None

    matches = [
        SearchResult("a", 0.5, {"text": "foo"}),
        SearchResult("b", 0.4, {"text": "bar"}),
    ]

    rescored = rescorer.rescore("query", matches)
    assert rescored is matches
