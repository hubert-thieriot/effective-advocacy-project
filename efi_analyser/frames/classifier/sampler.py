"""Corpus sampling utilities for frame classifier workflows."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from efi_core.types import Chunk
from efi_corpus.embedded.embedded_corpus import EmbeddedCorpus


@dataclass
class SamplerConfig:
    """Configuration controlling how passages are sampled from a corpus."""

    sample_size: int
    seed: int
    keywords: Optional[Sequence[str]] = None
    exclude_passage_ids: Optional[Sequence[str]] = None


class CorpusSampler:
    """Collect distinct passages from an embedded corpus with optional filtering."""

    def __init__(self, embedded_corpus: EmbeddedCorpus) -> None:
        self.embedded_corpus = embedded_corpus

    def collect(self, config: SamplerConfig) -> List[Tuple[str, str]]:
        if config.sample_size <= 0:
            raise ValueError("Sampler config sample_size must be positive.")

        doc_ids = self.embedded_corpus.corpus.list_ids()
        if not doc_ids:
            raise ValueError("Corpus is empty; no documents available for sampling.")

        rng = random.Random(config.seed)
        rng.shuffle(doc_ids)

        normalized_keywords = self._normalize_keywords(config.keywords)
        sampled: List[Tuple[str, str]] = []
        excluded: Set[str] = set(config.exclude_passage_ids or [])

        for doc_id in doc_ids:
            chunks = self.embedded_corpus.get_chunks(doc_id, materialize_if_necessary=True) or []
            sampled.extend(
                self._collect_from_doc(
                    doc_id,
                    chunks,
                    normalized_keywords,
                    config.sample_size - len(sampled),
                    excluded,
                )
            )
            if len(sampled) >= config.sample_size:
                return sampled[: config.sample_size]

        if len(sampled) < config.sample_size:
            raise ValueError(
                f"Requested {config.sample_size} passages but only gathered {len(sampled)} after scanning the corpus."
            )
        return sampled[: config.sample_size]

    # ------------------------------------------------------------------ helpers
    def _collect_from_doc(
        self,
        doc_id: str,
        chunks: Iterable[Chunk],
        keywords: Optional[Sequence[str]],
        remaining: int,
        excluded: Set[str],
    ) -> List[Tuple[str, str]]:
        collected: List[Tuple[str, str]] = []
        if remaining <= 0:
            return collected

        for chunk in chunks:
            text = chunk.text.strip()
            if not text:
                continue
            passage_id = f"{doc_id}:chunk{int(chunk.chunk_id):03d}"
            if passage_id in excluded:
                continue
            if keywords and not self._matches_keywords(text, keywords):
                continue
            collected.append((passage_id, text))
            excluded.add(passage_id)
            if len(collected) >= remaining:
                break
        return collected

    def _normalize_keywords(self, keywords: Optional[Sequence[str]]) -> Optional[List[str]]:
        if not keywords:
            return None
        normalized = [kw.strip().lower() for kw in keywords if kw and kw.strip()]
        return normalized or None

    def _matches_keywords(self, text: str, keywords: Sequence[str]) -> bool:
        lowered = text.lower()
        return any(keyword in lowered for keyword in keywords)


__all__ = ["SamplerConfig", "CorpusSampler"]
