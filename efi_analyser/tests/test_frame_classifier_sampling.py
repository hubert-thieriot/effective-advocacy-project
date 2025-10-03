from __future__ import annotations

from types import SimpleNamespace

import pytest

from efi_analyser.frames.classifier.sampler import CorpusSampler, SamplerConfig


class FakeCorpus:
    def __init__(self) -> None:
        self._chunks = {
            "doc1": [SimpleNamespace(chunk_id=0, text="Coal boosts jobs"), SimpleNamespace(chunk_id=1, text="Coal hurts health")],
            "doc2": [SimpleNamespace(chunk_id=0, text="Renewables grow rapidly")],
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


@pytest.mark.parametrize("exclude_ids", [None, ["doc1:chunk000"]])
def test_collect_random_passages_returns_expected_size(exclude_ids):
    embedded = FakeEmbeddedCorpus()
    sampler = CorpusSampler(embedded)
    passages = sampler.collect(
        SamplerConfig(
            sample_size=2,
            seed=7,
            exclude_passage_ids=exclude_ids,
        )
    )
    assert len(passages) == 2
    assert all(len(text) > 0 for _, text in passages)
    if exclude_ids:
        ids = {pid for pid, _ in passages}
        assert not ids.intersection(set(exclude_ids))


def test_collect_random_passages_with_keywords_filters():
    embedded = FakeEmbeddedCorpus()
    sampler = CorpusSampler(embedded)
    passages = sampler.collect(
        SamplerConfig(
            sample_size=1,
            seed=3,
            keywords=["renewables"],
        )
    )
    assert passages[0][0].startswith("doc2")
