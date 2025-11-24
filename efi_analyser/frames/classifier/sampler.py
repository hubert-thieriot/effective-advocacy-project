"""Corpus sampling utilities for frame classifier workflows."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
import re

from efi_core.types import Chunk
from efi_corpus.embedded.embedded_corpus import EmbeddedCorpus
from efi_core.utils import normalize_date

from ..identifiers import (
    make_global_doc_id,
    make_global_passage_id,
    split_passage_id,
    split_global_doc_id,
)
from ..types import Candidate


@dataclass
class SamplerConfig:
    """Configuration controlling how passages are sampled from a corpus."""

    sample_size: int
    seed: int
    keywords: Optional[Sequence[str]] = None  # Chunk-level keywords (passages must contain these)
    require_document_keywords: Optional[Sequence[str]] = None  # Document-level keywords (document text must contain these)
    exclude_passage_ids: Optional[Sequence[str]] = None
    # Optional content-based exclusion/cleanup rules
    exclude_regex: Optional[Sequence[str]] = None
    exclude_min_hits: Optional[Dict[str, int]] = None  # e.g., {"share price": 3}
    trim_after_markers: Optional[Sequence[str]] = None
    # Optional publication date lower bound (YYYY-MM-DD). Only docs on/after this date are considered.
    date_from: Optional[str] = None
    # Optional domain whitelist: only include documents from these domains (extracted from URL)
    domain_whitelist: Optional[Sequence[str]] = None


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
        
        # Optional document-level keyword filter: documents must contain at least one keyword in their text
        if config.require_document_keywords:
            normalized_doc_keywords = self._normalize_keywords(config.require_document_keywords)
            if normalized_doc_keywords:
                original_count = len(doc_ids)
                doc_keyword_filtered: List[str] = []
                for _doc_id in doc_ids:
                    try:
                        # Get document text and check for keywords
                        doc_text = self.embedded_corpus.corpus.get_text(_doc_id)
                        doc_text_lower = doc_text.lower()
                        # Check if any keyword appears in the document text
                        if any(kw in doc_text_lower for kw in normalized_doc_keywords):
                            doc_keyword_filtered.append(_doc_id)
                    except (FileNotFoundError, Exception):
                        # Skip documents where text cannot be read
                        continue
                doc_ids = doc_keyword_filtered
                if not doc_ids and original_count > 0:
                    print(f"ℹ️  Document keyword filter removed all documents (was {original_count}).")
        
        # Optional domain whitelist filter: only include documents from whitelisted domains
        if config.domain_whitelist:
            normalized_domains = [d.lower().strip() for d in config.domain_whitelist if d and d.strip()]
            if normalized_domains:
                original_count = len(doc_ids)
                domain_filtered: List[str] = []
                for _doc_id in doc_ids:
                    try:
                        meta = self.embedded_corpus.corpus.get_metadata(_doc_id)
                        url = meta.get("uri") or meta.get("url")
                        if not url:
                            continue
                        # Extract domain from URL (same logic as aggregation_document._extract_domain)
                        from urllib.parse import urlparse
                        parsed = urlparse(url)
                        netloc = parsed.netloc or parsed.path
                        if not netloc:
                            continue
                        domain = netloc.lower()
                        if domain.startswith("www."):
                            domain = domain[4:]
                        # Extract base domain (ignore subdomains)
                        parts = domain.split('.')
                        if len(parts) >= 2:
                            if len(parts) >= 3 and parts[-2] in ('co', 'com', 'org', 'net', 'ac', 'gov'):
                                doc_domain = '.'.join(parts[-3:])
                            else:
                                doc_domain = '.'.join(parts[-2:])
                        else:
                            doc_domain = domain
                        # Check if document domain is in whitelist
                        if doc_domain in normalized_domains:
                            domain_filtered.append(_doc_id)
                    except (FileNotFoundError, Exception):
                        # Skip documents where metadata cannot be read
                        continue
                doc_ids = domain_filtered
                if not doc_ids and original_count > 0:
                    print(f"ℹ️  Domain whitelist filter removed all documents (was {original_count}).")
        
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

    # ------------------------------------------------------------------ helpers
    def _normalize_keywords(self, keywords: Optional[Sequence[str]]) -> Optional[List[str]]:
        """Normalize keywords for case-insensitive matching."""
        if not keywords:
            return None
        normalized = [kw.strip().lower() for kw in keywords if kw and kw.strip()]
        return normalized or None

    # ------------------------------------------------------------------ public
    def collect_chunks(self, config: SamplerConfig) -> List[Tuple[str, str]]:
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
                    require_document_keywords=config.require_document_keywords,
                    exclude_passage_ids=local_exclude or None,
                    exclude_regex=config.exclude_regex,
                    exclude_min_hits=config.exclude_min_hits,
                    trim_after_markers=config.trim_after_markers,
                    date_from=config.date_from,
                    domain_whitelist=config.domain_whitelist,
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
                    require_document_keywords=config.require_document_keywords,
                    exclude_passage_ids=local_excludes or None,
                    exclude_regex=config.exclude_regex,
                    exclude_min_hits=config.exclude_min_hits,
                    trim_after_markers=config.trim_after_markers,
                    date_from=config.date_from,
                    domain_whitelist=config.domain_whitelist,
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

    def collect_docs(
        self,
        *,
        sample_size: int,
        seed: int,
        date_from: Optional[str] = None,
        exclude_doc_ids: Optional[Sequence[str]] = None,
        require_keywords: Optional[Sequence[str]] = None,
        domain_whitelist: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """Sample global document ids across corpora using the configured policy.

        This mirrors the cross-corpus allocation logic used for passage sampling
        but operates at the document level, with optional date, exclusion, keyword, and domain
        filters.
        
        Args:
            sample_size: Number of documents to sample
            seed: Random seed for reproducible sampling
            date_from: Only include documents published on or after this date (YYYY-MM-DD)
            exclude_doc_ids: Document IDs to exclude from sampling
            require_keywords: Documents must contain at least one of these keywords in their text
            domain_whitelist: Only include documents from these domains (extracted from URL)
        """
        if sample_size <= 0 or not self._corpora:
            return []

        # Normalize keywords for case-insensitive matching
        normalized_keywords: Optional[List[str]] = None
        if require_keywords:
            normalized_keywords = self._normalize_keywords(require_keywords)
        
        # Normalize domain whitelist for case-insensitive matching
        normalized_domains: Optional[List[str]] = None
        if domain_whitelist:
            normalized_domains = [d.lower().strip() for d in domain_whitelist if d and d.strip()]

        # Build per-corpus exclusion map from global doc ids
        exclude_map: Dict[str, set[str]] = {}
        if exclude_doc_ids:
            for gid in exclude_doc_ids:
                corpus_name, local_id = split_global_doc_id(str(gid))
                if corpus_name is None:
                    # Single-corpus case: apply to all corpora
                    for name in self._corpora.keys():
                        exclude_map.setdefault(name, set()).add(local_id)
                else:
                    exclude_map.setdefault(corpus_name, set()).add(local_id)

        # Collect available local doc ids per corpus after filters
        df_norm: Optional[str] = None
        if date_from:
            df_norm = str(date_from).strip() or None

        per_corpus_ids: Dict[str, List[str]] = {}
        for name, embedded in self._corpora.items():
            ids = list(embedded.corpus.list_ids())

            # Optional date filter
            if df_norm:
                filtered: List[str] = []
                for local_id in ids:
                    try:
                        meta = embedded.corpus.get_metadata(local_id)
                        pub = meta.get("published_at")
                        dt = normalize_date(pub)
                        if dt and dt.date().isoformat() >= df_norm:
                            filtered.append(local_id)
                    except Exception:
                        continue
                ids = filtered

            # Optional keyword filter: documents must contain at least one keyword in their text
            if normalized_keywords:
                keyword_filtered: List[str] = []
                for local_id in ids:
                    try:
                        # Get document text and check for keywords
                        doc_text = embedded.corpus.get_text(local_id)
                        doc_text_lower = doc_text.lower()
                        # Check if any keyword appears in the document text
                        if any(kw in doc_text_lower for kw in normalized_keywords):
                            keyword_filtered.append(local_id)
                    except (FileNotFoundError, Exception):
                        # Skip documents where text cannot be read
                        continue
                ids = keyword_filtered

            # Optional domain whitelist filter: only include documents from whitelisted domains
            if normalized_domains:
                domain_filtered: List[str] = []
                for local_id in ids:
                    try:
                        meta = embedded.corpus.get_metadata(local_id)
                        url = meta.get("uri") or meta.get("url")
                        if not url:
                            continue
                        # Extract domain from URL (same logic as aggregation_document._extract_domain)
                        from urllib.parse import urlparse
                        parsed = urlparse(url)
                        netloc = parsed.netloc or parsed.path
                        if not netloc:
                            continue
                        domain = netloc.lower()
                        if domain.startswith("www."):
                            domain = domain[4:]
                        # Extract base domain (ignore subdomains)
                        parts = domain.split('.')
                        if len(parts) >= 2:
                            if len(parts) >= 3 and parts[-2] in ('co', 'com', 'org', 'net', 'ac', 'gov'):
                                doc_domain = '.'.join(parts[-3:])
                            else:
                                doc_domain = '.'.join(parts[-2:])
                        else:
                            doc_domain = domain
                        # Check if document domain is in whitelist
                        if doc_domain in normalized_domains:
                            domain_filtered.append(local_id)
                    except (FileNotFoundError, Exception):
                        # Skip documents where metadata cannot be read
                        continue
                ids = domain_filtered

            # Exclude already-selected documents (by local id)
            excluded_locals = exclude_map.get(name, set())
            if excluded_locals:
                ids = [doc_id for doc_id in ids if doc_id not in excluded_locals]

            per_corpus_ids[name] = ids

        available_counts: Dict[str, int] = {
            name: len(ids) for name, ids in per_corpus_ids.items() if ids
        }
        if not available_counts:
            return []

        # Allocate sample counts per corpus according to policy
        total = sample_size
        if self._policy == "equal":
            per = total // len(available_counts)
            allocations: Dict[str, int] = {name: per for name in available_counts}
            remainder = total - per * len(available_counts)
        else:  # proportional
            total_docs = sum(available_counts.values())
            if total_docs <= 0:
                return []
            allocations = {
                name: int(total * count / total_docs)
                for name, count in available_counts.items()
            }
            remainder = total - sum(allocations.values())

        if remainder > 0:
            rng = random.Random(seed)
            ordered = list(available_counts.keys())
            rng.shuffle(ordered)
            for name in ordered:
                if remainder <= 0:
                    break
                allocations[name] = allocations.get(name, 0) + 1
                remainder -= 1

        # Sample local ids per corpus according to allocations
        sampled: List[str] = []
        rng = random.Random(seed + 17)
        multi = len(self._corpora) > 1
        for name, ids in per_corpus_ids.items():
            target = allocations.get(name, 0)
            if target <= 0 or not ids:
                continue
            local_ids = list(ids)
            rng.shuffle(local_ids)
            chosen = local_ids[: min(target, len(local_ids))]
            for local_id in chosen:
                sampled.append(
                    make_global_doc_id(name if multi else None, local_id)
                )

        if len(sampled) < sample_size:
            print(
                f"⚠️ Requested {sample_size} documents but only sampled {len(sampled)} across corpora."
            )

        rng = random.Random(seed + 33)
        rng.shuffle(sampled)
        return sampled[:sample_size]

    def collect_candidates(self, config: SamplerConfig) -> List[Candidate]:
        """Collect passages along with document-level metadata.

        This wraps :meth:`collect` to preserve the existing sampling behaviour
        while enriching each sampled passage with metadata needed downstream
        (URL, title, published_at, doc_folder_path, corpus, doc ids, etc.).
        """
        raw = self.collect_chunks(config)
        if not raw:
            return []

        # Cache doc-level metadata to avoid repeated corpus lookups
        doc_meta_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        results: List[Candidate] = []

        # Helper to resolve the EmbeddedCorpus for a given corpus name
        def _get_embedded(corpus_name: Optional[str]) -> Optional[EmbeddedCorpus]:
            if corpus_name:
                return self._corpora.get(corpus_name)
            if len(self._corpora) == 1:
                return next(iter(self._corpora.values()))
            return None

        for passage_id, text in raw:
            corpus_name, local_doc_id, _ = split_passage_id(passage_id)
            embedded = _get_embedded(corpus_name)

            metadata: Dict[str, Any] = {}
            # Core identifiers
            metadata["doc_id"] = local_doc_id
            metadata["global_doc_id"] = make_global_doc_id(corpus_name, local_doc_id) if corpus_name else local_doc_id
            if corpus_name:
                metadata["corpus"] = corpus_name

            if embedded is not None:
                cache_key = (embedded.corpus.corpus_path.name, local_doc_id)
                if cache_key not in doc_meta_cache:
                    corpus = embedded.corpus
                    index_entry = corpus.get_index_entry(local_doc_id) or {}
                    meta = corpus.get_metadata(local_doc_id)
                    fetch_info = corpus.get_fetch_info(local_doc_id)
                    merged_meta: Dict[str, str] = {}
                    for source in (index_entry, meta, fetch_info):
                        if not isinstance(source, dict):
                            continue
                        for key, value in source.items():
                            if key not in merged_meta and isinstance(value, str):
                                merged_meta[key] = value

                    try:
                        doc_folder_path = corpus.layout.doc_dir(local_doc_id)
                        doc_folder_path_abs = (
                            doc_folder_path.resolve() if doc_folder_path.exists() else doc_folder_path
                        )
                        doc_folder_str = str(doc_folder_path_abs)
                    except Exception:
                        doc_folder_str = ""

                    doc_meta_cache[cache_key] = {
                        "url": merged_meta.get("url", ""),
                        "title": merged_meta.get("title", ""),
                        "published_at": merged_meta.get("published_at", ""),
                        "doc_folder_path": doc_folder_str,
                    }

                extra = doc_meta_cache.get(cache_key) or {}
                for k, v in extra.items():
                    # Only fill when value is non-empty
                    if isinstance(v, str) and v.strip():
                        metadata[k] = v

            results.append(Candidate(item_id=passage_id, text=text, meta=metadata))

        return results

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


EmbeddedCorporaSampler = CompositeCorpusSampler

__all__ = ["SamplerConfig", "CorpusSampler", "CompositeCorpusSampler", "EmbeddedCorporaSampler"]
