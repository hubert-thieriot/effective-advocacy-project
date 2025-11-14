"""Corpus sampling utilities for frame classifier workflows."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
import re

from efi_core.types import Chunk
from efi_corpus.embedded.embedded_corpus import EmbeddedCorpus
from efi_core.utils import normalize_date

from ..identifiers import make_global_passage_id, split_passage_id


@dataclass
class SamplerConfig:
    """Configuration controlling how passages are sampled from a corpus."""

    sample_size: int
    seed: int
    keywords: Optional[Sequence[str]] = None
    exclude_passage_ids: Optional[Sequence[str]] = None
    # Optional content-based exclusion/cleanup rules
    exclude_regex: Optional[Sequence[str]] = None
    exclude_min_hits: Optional[Dict[str, int]] = None  # e.g., {"share price": 3}
    trim_after_markers: Optional[Sequence[str]] = None
    # Optional publication date lower bound (YYYY-MM-DD). Only docs on/after this date are considered.
    date_from: Optional[str] = None


class CorpusSampler:
    """Collect distinct passages from an embedded corpus with optional filtering."""

    def __init__(self, embedded_corpus: EmbeddedCorpus) -> None:
        self.embedded_corpus = embedded_corpus

    def collect(self, config: SamplerConfig, *, allow_partial: bool = False) -> List[Tuple[str, str]]:
        if config.sample_size <= 0:
            raise ValueError("Sampler config sample_size must be positive.")

        doc_ids = self.embedded_corpus.corpus.list_ids()
        if not doc_ids:
            raise ValueError("Corpus is empty; no documents available for sampling.")

        # Optional filtering by publication date
        if config.date_from:
            try:
                df_norm = str(config.date_from).strip()
            except Exception:
                df_norm = None
            if df_norm:
                original_count = len(doc_ids)
                filtered: List[str] = []
                for _doc_id in doc_ids:
                    try:
                        meta = self.embedded_corpus.corpus.get_metadata(_doc_id)
                        pub = meta['published_at']
                        dt = normalize_date(pub)
                        if not dt:
                            continue  # skip docs without parseable date
                        if dt.date().isoformat() >= df_norm:
                            filtered.append(_doc_id)
                    except Exception:
                        continue
                doc_ids = filtered
                if not doc_ids:
                    print(f"ℹ️  Date filter {df_norm} removed all documents (was {original_count}).")
        
        rng = random.Random(config.seed)
        rng.shuffle(doc_ids)

        normalized_keywords = self._normalize_keywords(config.keywords)
        exclude_regex = self._compile_patterns(config.exclude_regex)
        exclude_min_hits = self._normalize_min_hits(config.exclude_min_hits)
        trim_markers = self._normalize_list(config.trim_after_markers)
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
                    exclude_regex,
                    exclude_min_hits,
                    trim_markers,
                )
            )
            if len(sampled) >= config.sample_size:
                return sampled[: config.sample_size]

        if len(sampled) < config.sample_size:
            if allow_partial:
                return sampled
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
        exclude_regex: Optional[List[re.Pattern]],
        exclude_min_hits: Optional[Dict[str, int]],
        trim_markers: Optional[Sequence[str]],
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
            # Apply preprocessing: trim after markers if configured
            text_processed = self._trim_after_markers(text, trim_markers)
            # Exclude based on configured rules
            if self._is_excluded(text_processed, exclude_regex, exclude_min_hits):
                continue
            if keywords and not self._matches_keywords(text_processed, keywords):
                continue
            collected.append((passage_id, text_processed))
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

    # ------------------------- new helpers for filtering ---------------------
    def _normalize_list(self, items: Optional[Sequence[str]]) -> Optional[List[str]]:
        if not items:
            return None
        normalized = [s for s in (str(v).strip() for v in items) if s]
        return normalized or None

    def _trim_after_markers(self, text: str, markers: Optional[Sequence[str]]) -> str:
        if not markers:
            return text
        lowered = text.lower()
        cut = len(text)
        for marker in markers:
            m = marker.lower()
            idx = lowered.find(m)
            if idx != -1:
                cut = min(cut, idx)
        return text[:cut].strip() if cut < len(text) else text

    def _is_excluded(
        self,
        text: str,
        patterns: Optional[List[re.Pattern]],
        min_hits: Optional[Dict[str, int]],
    ) -> bool:
        if patterns:
            for pat in patterns:
                if pat.search(text):
                    return True
        if min_hits:
            lowered = text.lower()
            for phrase, threshold in min_hits.items():
                if lowered.count(phrase) >= threshold:
                    return True
        return False

    def _compile_patterns(self, patterns: Optional[Sequence[str]]) -> Optional[List[re.Pattern]]:
        if not patterns:
            return None
        compiled: List[re.Pattern] = []
        for p in patterns:
            try:
                compiled.append(re.compile(str(p), flags=re.IGNORECASE | re.MULTILINE))
            except re.error:
                continue
        return compiled or None

    def _normalize_min_hits(self, mapping: Optional[Dict[str, int]]) -> Optional[Dict[str, int]]:
        if not mapping:
            return None
        result: Dict[str, int] = {}
        for k, v in mapping.items():
            key = str(k).strip().lower()
            try:
                val = int(v)
            except Exception:
                continue
            if key and val >= 1:
                result[key] = val
        return result or None

    # Caches configured per-instance via setters on CompositeCorpusSampler


class CompositeCorpusSampler:
    """Coordinate sampling across multiple corpora with configurable policy."""

    VALID_POLICIES = {"equal", "proportional"}

    def __init__(
        self,
        corpora: Mapping[str, EmbeddedCorpus],
        policy: str = "equal",
    ) -> None:
        if not corpora:
            raise ValueError("CompositeCorpusSampler requires at least one corpus.")
        canonical_policy = policy.strip().lower()
        if canonical_policy not in self.VALID_POLICIES:
            raise ValueError(
                f"Unsupported sampling policy '{policy}'. Valid policies: {sorted(self.VALID_POLICIES)}"
            )
        self._corpora: Dict[str, EmbeddedCorpus] = dict(corpora)
        self._samplers: Dict[str, CorpusSampler] = {
            name: CorpusSampler(corpus) for name, corpus in self._corpora.items()
        }
        self._policy = canonical_policy

    # ------------------------------------------------------------------ public
    def collect(self, config: SamplerConfig) -> List[Tuple[str, str]]:
        if not self._corpora:
            return []

        if len(self._corpora) == 1:
            name, sampler = next(iter(self._samplers.items()))
            local_exclude = self._resolve_local_excludes(config.exclude_passage_ids, name)
            local_samples = sampler.collect(
                SamplerConfig(
                    sample_size=config.sample_size,
                    seed=config.seed,
                    keywords=config.keywords,
                    exclude_passage_ids=local_exclude or None,
                    exclude_regex=config.exclude_regex,
                    exclude_min_hits=config.exclude_min_hits,
                    trim_after_markers=config.trim_after_markers,
                    date_from=config.date_from,
                ),
                allow_partial=True,
            )
            if len(local_samples) < config.sample_size:
                print(
                    f"⚠️ Requested {config.sample_size} passages but only gathered {len(local_samples)} from corpus '{name}'."
                )
            return local_samples

        allocations = self._allocate_counts(config.sample_size, config.seed)

        exclude_map = self._build_exclude_map(config.exclude_passage_ids)
        results: List[Tuple[str, str]] = []

        for index, (name, target) in enumerate(allocations.items()):
            if target <= 0:
                continue
            sampler = self._samplers[name]
            local_excludes = exclude_map.get(name)
            local_samples = sampler.collect(
                SamplerConfig(
                    sample_size=target,
                    seed=config.seed + index,
                    keywords=config.keywords,
                    exclude_passage_ids=local_excludes or None,
                    exclude_regex=config.exclude_regex,
                    exclude_min_hits=config.exclude_min_hits,
                    trim_after_markers=config.trim_after_markers,
                    date_from=config.date_from,
                ),
                allow_partial=True,
            )
            results.extend(
                (make_global_passage_id(name, passage_id), text)
                for passage_id, text in local_samples
            )

        if len(results) < config.sample_size:
            print(
                f"⚠️ Requested {config.sample_size} passages but only gathered {len(results)} across corpora."
            )

        rng = random.Random(config.seed)
        rng.shuffle(results)
        return results[: config.sample_size]

    # ----------------------------------------------------------------- helpers
    def _allocate_counts(self, total: int, seed: int) -> Dict[str, int]:
        counts = {
            name: max(0, corpus.corpus.get_document_count())
            for name, corpus in self._corpora.items()
        }
        available = {name: count for name, count in counts.items() if count > 0}
        if not available:
            return {name: 0 for name in self._corpora}

        if self._policy == "equal":
            per_corpus = total // len(available)
            allocations = {name: per_corpus for name in available}
            remainder = total - per_corpus * len(available)
        else:  # proportional
            total_docs = sum(available.values())
            if total_docs <= 0:
                return {name: 0 for name in self._corpora}
            allocations = {
                name: int(total * count / total_docs)
                for name, count in available.items()
            }
            remainder = total - sum(allocations.values())

        if remainder > 0:
            rng = random.Random(seed)
            ordered = list(available.keys())
            rng.shuffle(ordered)
            for name in ordered:
                if remainder <= 0:
                    break
                allocations[name] = allocations.get(name, 0) + 1
                remainder -= 1

        baseline = {name: 0 for name in self._corpora}
        baseline.update(allocations)
        return baseline

    def _build_exclude_map(
        self, exclude_ids: Optional[Sequence[str]]
    ) -> Dict[str, List[str]]:
        if not exclude_ids:
            return {}
        mapping: Dict[str, List[str]] = {name: [] for name in self._corpora}
        ambiguous: List[str] = []
        for passage_id in exclude_ids:
            corpus_name, _doc_id, local_passage_id = split_passage_id(str(passage_id))
            if corpus_name is None:
                ambiguous.append(local_passage_id)
                continue
            if corpus_name in mapping:
                mapping[corpus_name].append(local_passage_id)
        if ambiguous:
            for entries in mapping.values():
                entries.extend(ambiguous)
        return {name: entries for name, entries in mapping.items() if entries}

    def _resolve_local_excludes(
        self, exclude_ids: Optional[Sequence[str]], corpus_name: str
    ) -> List[str]:
        if not exclude_ids:
            return []
        excludes: List[str] = []
        for passage_id in exclude_ids:
            candidate_corpus, _doc_id, local_passage_id = split_passage_id(str(passage_id))
            if candidate_corpus in (None, corpus_name):
                excludes.append(local_passage_id)
        return excludes


__all__ = ["SamplerConfig", "CorpusSampler", "CompositeCorpusSampler"]
