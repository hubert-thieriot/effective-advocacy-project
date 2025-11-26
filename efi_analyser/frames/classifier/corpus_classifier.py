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
        doc_id_list = self._resolve_doc_ids(doc_ids, sample_size)
        if not doc_id_list:
            return DocumentClassifications()

        spec = make_filter_spec(
            exclude_regex=self.exclude_regex,
            exclude_min_hits=self.exclude_min_hits,
            trim_after_markers=self.trim_after_markers,
            keywords=self.require_keywords,
        )

        classified_docs = DocumentClassifications()
        iterator = tqdm(doc_id_list, desc="Classifying documents", unit="doc", leave=False)

        for doc_id in iterator:
            classification = self._classify_document(doc_id, spec)
            if classification is not None:
                classified_docs.append(classification)

        iterator.close()

        if output_dir:
            classified_docs.to_folder(output_dir)

        return classified_docs

    def _resolve_doc_ids(
        self,
        doc_ids: Optional[Sequence[str]],
        sample_size: Optional[int],
    ) -> List[str]:
        """Resolve and optionally sample document IDs."""
        if doc_ids is not None:
            return list(doc_ids)
        
        doc_id_list = self.corpora.list_global_doc_ids()
        if sample_size is not None:
            limited = max(0, min(sample_size, len(doc_id_list)))
            rng = random.Random(self.seed)
            rng.shuffle(doc_id_list)
            return doc_id_list[:limited]
        return list(doc_id_list)

    def _extract_chunks(
        self,
        doc_id: str,
        spec,
    ) -> Tuple[Optional[str], Optional[str], List[Tuple[str, str]]]:
        """Extract and filter chunks from a document.
        
        Returns: (corpus_name, published_at, list of (chunk_id, text) pairs)
        """
        corpus_name, local_doc_id = split_global_doc_id(doc_id)
        embedded_corpus = self.corpora.get_embedded(corpus_name)

        doc = embedded_corpus.corpus.get_document(local_doc_id)
        if doc is None:
            return None, None, []

        published_at = doc.published_at
        if not published_at:
            metadata = embedded_corpus.corpus.get_metadata(local_doc_id)
            if metadata and isinstance(metadata, dict):
                published_at = metadata.get("published_at")

        chunks = embedded_corpus.get_chunks(local_doc_id, materialize_if_necessary=True) or []
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
            chunk_text_pairs.append((chunk_id, text))

        return corpus_name, published_at, chunk_text_pairs

    def _classify_document(
        self,
        doc_id: str,
        spec,
    ) -> Optional[DocumentClassification]:
        """Classify a single document's chunks."""
        corpus_name, local_doc_id = split_global_doc_id(doc_id)
        embedded_corpus = self.corpora.get_embedded(corpus_name)
        doc = embedded_corpus.corpus.get_document(local_doc_id)
        if doc is None:
            return None

        corpus_name, published_at, chunk_text_pairs = self._extract_chunks(doc_id, spec)
        if not chunk_text_pairs:
            return None

        texts = [text for _, text in chunk_text_pairs]

        # Apply optional keyword gate
        if spec.keywords is not None:
            lowered_texts = [t.lower() for t in texts]
            if not any(any(kw in t for kw in spec.keywords) for t in lowered_texts):
                return None

        # Classify all chunks
        probabilities = self.model.predict_proba_batch(texts, batch_size=self.batch_size)
        
        chunk_records = self._build_chunk_records(chunk_text_pairs, probabilities)

        return DocumentClassification(payload={
            "doc_id": doc_id,
            "corpus": corpus_name,
            "local_doc_id": local_doc_id,
            "title": doc.title,
            "url": doc.url,
            "published_at": published_at,
            "chunks": chunk_records,
        })

    def _build_chunk_records(
        self,
        chunk_text_pairs: List[Tuple[str, str]],
        probabilities: List[Dict[str, float]],
    ) -> List[Dict[str, object]]:
        """Build chunk records from text pairs and their probabilities."""
        chunk_records: List[Dict[str, object]] = []
        for (chunk_id, passage_text), probs in zip(chunk_text_pairs, probabilities):
            ordered = sorted(probs.items(), key=lambda item: item[1], reverse=True)
            chunk_records.append({
                "chunk_id": chunk_id,
                "text": passage_text,
                "probabilities": {frame_id: float(score) for frame_id, score in probs.items()},
                "top_frames": [fid for fid, _ in ordered[:3]],
            })
        return chunk_records


__all__ = [
    "DocumentClassification",
    "DocumentClassifications",
    "FrameClassifier",
]
