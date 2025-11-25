import json
from pathlib import Path
from dataclasses import dataclass
from typing import List

from efi_corpus.corpus_handle import CorpusHandle
from efi_corpus.embedded.embedded_corpus import EmbeddedCorpus
from efi_analyser.frames.classifier.sampler import CompositeCorpusSampler, SamplerConfig
from efi_core.types import ChunkerSpec, EmbedderSpec


@dataclass
class DummyChunker:
    name: str = "dummy"

    @property
    def spec(self) -> ChunkerSpec:
        return ChunkerSpec(name=self.name, params={})

    def chunk(self, text: str) -> List[str]:
        # Simple chunker: split by period, ensure at least one chunk
        parts = [p.strip() for p in text.split(".") if p.strip()]
        return parts or [text]


@dataclass
class DummyEmbedder:
    model_name: str = "dummy"

    @property
    def spec(self) -> EmbedderSpec:
        return EmbedderSpec(model_name=self.model_name, dim=3)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [[0.0, 0.0, 0.0] for _ in texts]


def _write_doc(corpus: CorpusHandle, stable_id: str, title: str, published_at: str, text: str = "Hello. World.") -> None:
    meta = {
        "id": stable_id,
        "title": title,
        "published_at": published_at,
        "uri": f"https://example.org/{stable_id}",
        "language": "en",
    }
    fetch = {"status": "ok"}
    corpus.write_document(
        stable_id=stable_id,
        meta=meta,
        text=text,
        raw_bytes=b"",
        raw_ext="html",
        fetch_info=fetch,
    )
    corpus.append_index({
        "id": stable_id,
        "url": meta["uri"],
        "title": title,
        "published_at": published_at,
    })


def test_composite_sampler_date_from_filters(tmp_path: Path):
    # Create two tiny corpora under a temp directory
    corpora_root = tmp_path / "corpora"
    corpora_root.mkdir(parents=True, exist_ok=True)

    c1_path = corpora_root / "test_corpus_a"
    c2_path = corpora_root / "test_corpus_b"
    c1 = CorpusHandle(c1_path, read_only=False)
    c2 = CorpusHandle(c2_path, read_only=False)

    # Write documents: two before 2024, two after
    _write_doc(c1, "a1", "Old A", "2023-05-01")
    _write_doc(c1, "a2", "New A", "2024-06-15")
    _write_doc(c2, "b1", "Old B", "2023-12-31")
    _write_doc(c2, "b2", "New B", "2024-01-01")

    # Build EmbeddedCorpus with dummy chunker/embedder
    chunker = DummyChunker()
    embedder = DummyEmbedder()
    ec1 = EmbeddedCorpus(c1_path, tmp_path / "ws", chunker, embedder)
    ec2 = EmbeddedCorpus(c2_path, tmp_path / "ws", chunker, embedder)

    sampler = CompositeCorpusSampler({"test_corpus_a": ec1, "test_corpus_b": ec2}, policy="equal")

    # Filter on/after 2024-01-01 → should exclude a1 (2023-05-01) and b1 (2023-12-31)
    results = sampler.collect_chunks(
        SamplerConfig(
            sample_size=4,
            seed=42,
            keywords=None,
            exclude_passage_ids=None,
            date_from="2024-01-01",
        )
    )
    # Each doc has 2 chunks; after filter, 2 docs remain → up to 4 passages available
    assert len(results) <= 4
    assert len(results) > 0
    # Ensure sampled passage IDs belong to new docs only
    ids = [pid for pid, _ in results]
    assert not any("a1:" in pid or "b1:" in pid for pid in ids)


def test_composite_sampler_date_from_too_strict(tmp_path: Path):
    corpora_root = tmp_path / "corpora"
    corpora_root.mkdir(parents=True, exist_ok=True)

    c1_path = corpora_root / "test_corpus_c"
    c1 = CorpusHandle(c1_path, read_only=False)
    _write_doc(c1, "c1", "Old C", "2020-01-01")

    chunker = DummyChunker()
    embedder = DummyEmbedder()
    ec1 = EmbeddedCorpus(c1_path, tmp_path / "ws", chunker, embedder)
    sampler = CompositeCorpusSampler({"test_corpus_c": ec1}, policy="equal")

    # Too strict date_from eliminates all docs
    results = sampler.collect_chunks(
        SamplerConfig(
            sample_size=2,
            seed=1,
            date_from="2100-01-01",
        )
    )
    assert results == []
