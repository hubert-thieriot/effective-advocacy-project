"""Corpus-level frame classification helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import json
import random

from tqdm import tqdm

from apps.narrative_framing.filtering import make_filter_spec, filter_text as nf_filter_text
from efi_analyser.frames.corpora import EmbeddedCorpora
from efi_analyser.frames.classifier.model import FrameClassifierModel
from efi_analyser.frames.identifiers import make_global_passage_id, split_global_doc_id


@dataclass
class DocumentClassification:
    """Single document's chunk-level classification payload."""

    payload: Dict[str, object]

    @property
    def doc_id(self) -> str:
        return str(self.payload.get("doc_id", "")).strip()

    def to_payload(self) -> Dict[str, object]:
        return self.payload


class DocumentClassifications(List[DocumentClassification]):
    """Collection of document classifications with folder I/O helpers."""

    @classmethod
    def from_folder(
        cls,
        directory: Path,
        doc_ids: Optional[Iterable[str]] = None,
    ) -> "DocumentClassifications":
        items = cls()
        if not directory.exists():
            return items
        doc_id_filter = set(doc_ids) if doc_ids is not None else None
        for child in sorted(directory.glob("*.json")):
            try:
                payload = json.loads(child.read_text(encoding="utf-8"))
            except Exception:
                continue
            doc = DocumentClassification(payload=payload)
            if not doc.doc_id:
                continue
            if doc_id_filter and doc.doc_id not in doc_id_filter:
                continue
            items.append(doc)
        return items

    def to_folder(self, directory: Path) -> None:
        """Write all classifications to ``directory`` and remove stale files."""
        directory.mkdir(parents=True, exist_ok=True)
        keep_doc_ids: set[str] = set()
        for doc in self:
            doc_id = doc.doc_id
            if not doc_id:
                continue
            keep_doc_ids.add(doc_id)
            path = directory / f"{doc_id}.json"
            path.write_text(json.dumps(doc.to_payload(), indent=2, ensure_ascii=False), encoding="utf-8")

        for existing in directory.glob("*.json"):
            if existing.stem not in keep_doc_ids:
                existing.unlink(missing_ok=True)

    @property
    def n_docs(self) -> int:
        """Number of distinct documents represented in this collection."""
        return len({doc.doc_id for doc in self if doc.doc_id})

    @property
    def n_chunks(self) -> int:
        """Total number of classified chunks across all documents."""
        total = 0
        for doc in self:
            chunks = doc.payload.get("chunks", [])
            if isinstance(chunks, list):
                total += len(chunks)
        return total


@dataclass
class FrameClassifier:
    """Corpus-level frame classifier built on a trained model."""

    model: FrameClassifierModel
    corpora: EmbeddedCorpora
    batch_size: int
    seed: Optional[int] = None
    require_keywords: Optional[Sequence[str]] = None
    exclude_regex: Optional[Sequence[str]] = None
    exclude_min_hits: Optional[Dict[str, int]] = None
    trim_after_markers: Optional[Sequence[str]] = None

    def run(
        self,
        sample_size: Optional[int] = None,
        *,
        doc_ids: Optional[Sequence[str]] = None,
        output_dir: Optional[Path] = None,
    ) -> DocumentClassifications:
        """Classify corpus documents into per-chunk frame scores."""
        if doc_ids is not None:
            doc_id_list = list(doc_ids)
        else:
            doc_id_list = self.corpora.list_global_doc_ids()
        if not doc_id_list:
            return DocumentClassifications()

        if doc_ids is None and sample_size is not None:
            limited = max(0, min(sample_size, len(doc_id_list)))
            rng = random.Random(self.seed)
            rng.shuffle(doc_id_list)
            doc_id_list = doc_id_list[:limited]
        elif doc_ids is None:
            # When no explicit sample size and doc_ids not provided, respect original ordering.
            doc_id_list = list(doc_id_list)

        classified_docs = DocumentClassifications()

        spec = make_filter_spec(
            exclude_regex=self.exclude_regex,
            exclude_min_hits=self.exclude_min_hits,
            trim_after_markers=self.trim_after_markers,
            keywords=self.require_keywords,
        )

        iterator = tqdm(
            doc_id_list,
            desc="Classifying documents",
            unit="doc",
            leave=False,
        )

        for doc_id in iterator:
            corpus_name, local_doc_id = split_global_doc_id(doc_id)
            embedded_corpus = self.corpora.get_embedded(corpus_name)

            doc = embedded_corpus.corpus.get_document(local_doc_id)
            if doc is None:
                continue

            published_at = doc.published_at
            if not published_at:
                metadata = embedded_corpus.corpus.get_metadata(local_doc_id)
                if metadata and isinstance(metadata, dict):
                    published_at = metadata.get("published_at")

            chunks = embedded_corpus.get_chunks(local_doc_id, materialize_if_necessary=True) or []
            texts: List[str] = []
            chunk_text_pairs: List[Tuple[str, str]] = []
            for chunk in chunks:
                text = nf_filter_text((chunk.text or ""), spec)
                if not text:
                    continue
                local_passage_id = f"{local_doc_id}:chunk{int(chunk.chunk_id):03d}"
                chunk_id = make_global_passage_id(
                    corpus_name if len(self.corpora) > 1 else None,
                    local_passage_id,
                )
                texts.append(text)
                chunk_text_pairs.append((chunk_id, text))

            if not texts:
                continue

            # Apply optional keyword gate: skip classification if none of the chunks
            # contain any of the required keywords (case-insensitive substring).
            if spec.keywords is not None:
                lowered_texts = [t.lower() for t in texts]
                if not any(any(kw in t for kw in spec.keywords) for t in lowered_texts):
                    continue

            probabilities = self.model.predict_proba_batch(texts, batch_size=self.batch_size)
            chunk_records: List[Dict[str, object]] = []
            for (chunk_id, passage_text), probs in zip(chunk_text_pairs, probabilities):
                ordered = sorted(probs.items(), key=lambda item: item[1], reverse=True)
                chunk_records.append(
                    {
                        "chunk_id": chunk_id,
                        "text": passage_text,
                        "probabilities": {frame_id: float(score) for frame_id, score in probs.items()},
                        "top_frames": [fid for fid, _ in ordered[:3]],
                    }
                )

            payload = {
                "doc_id": doc_id,
                "corpus": corpus_name,
                "local_doc_id": local_doc_id,
                "title": doc.title,
                "url": doc.url,
                "published_at": published_at,
                "chunks": chunk_records,
            }

            classified_docs.append(DocumentClassification(payload=payload))

        iterator.close()

        if output_dir:
            classified_docs.to_folder(output_dir)

        return classified_docs


__all__ = [
    "DocumentClassification",
    "DocumentClassifications",
    "FrameClassifier",
]
