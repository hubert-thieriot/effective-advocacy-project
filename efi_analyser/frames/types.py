"""Data structures for frame induction and assignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import random


@dataclass
class Frame:
    """Definition of a single media frame within a schema."""

    frame_id: str
    name: str
    description: str
    keywords: List[str]
    examples: List[str]
    short_name: str = ""
    anti_triggers: List[str] = field(default_factory=list)  # Explicit exclusion patterns
    boundary_notes: List[str] = field(default_factory=list)  # Human-readable disambiguation notes


@dataclass
class FrameSchema:
    """Domain-specific collection of frames produced by the inducer."""

    domain: str
    frames: List[Frame]
    notes: str = ""
    schema_id: str = ""


@dataclass
class FrameAssignment:
    """Probabilistic assignment of a candidate text to frames in a schema."""

    passage_id: str
    passage_text: str
    probabilities: Dict[str, float]
    top_frames: List[str]
    rationale: str = ""
    evidence_spans: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class FrameAssignments(List[FrameAssignment]):
    """Convenience collection for working with groups of frame assignments."""

    @property
    def count(self) -> int:
        return len(self)

    def select_random(self, n: int, seed: Optional[int] = None) -> "FrameAssignments":
        """Return a new FrameAssignments with up to n randomly selected items."""
        if n <= 0 or not self:
            return FrameAssignments()
        n = min(n, len(self))
        indices = list(range(len(self)))
        rng = random.Random(seed)
        rng.shuffle(indices)
        return FrameAssignments(self[i] for i in indices[:n])

    # ------------------------------------------------------------------ I/O helpers
    @classmethod
    def load(cls, path: Path) -> "FrameAssignments":
        """Load frame assignments from a JSON file on disk."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = cls()
        for item in payload:
            items.append(
                FrameAssignment(
                    passage_id=item["passage_id"],
                    passage_text=item.get("passage_text", ""),
                    probabilities=item.get("probabilities", {}),
                    top_frames=item.get("top_frames", []),
                    rationale=item.get("rationale", ""),
                    evidence_spans=item.get("evidence_spans", []),
                    metadata=item.get("metadata", {}),
                )
            )
        return items

    def save(self, path: Path) -> None:
        """Persist assignments to a JSON file."""
        serialized = [
            {
                "passage_id": assignment.passage_id,
                "passage_text": assignment.passage_text,
                "probabilities": assignment.probabilities,
                "top_frames": assignment.top_frames,
                "rationale": assignment.rationale,
                "evidence_spans": assignment.evidence_spans,
                "metadata": assignment.metadata,
            }
            for assignment in self
        ]
        path.write_text(json.dumps(serialized, indent=2, ensure_ascii=False), encoding="utf-8")

    def to_classifications(self) -> "DocumentClassifications":
        """Convert LLM frame assignments to document classifications format.
        
        This allows using LLM annotations for aggregation when classifier is disabled.
        Groups assignments by document and creates classification payloads.
        """
        from collections import defaultdict
        from efi_analyser.frames.classifier import DocumentClassification, DocumentClassifications
        from efi_analyser.frames.identifiers import split_passage_id
        
        # Group assignments by doc_id
        doc_chunks: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        doc_meta: Dict[str, Dict[str, Any]] = {}
        
        for assignment in self:
            # Parse passage_id to extract corpus and doc_id
            corpus_name, local_doc_id, _ = split_passage_id(assignment.passage_id)
            
            # Reconstruct global doc_id
            if corpus_name:
                doc_id = f"{corpus_name}@@{local_doc_id}"
            else:
                doc_id = local_doc_id
            
            # Build chunk record
            chunk: Dict[str, Any] = {
                "text": assignment.passage_text,
                "probabilities": assignment.probabilities,
            }
            doc_chunks[doc_id].append(chunk)
            
            # Store metadata from assignment if available
            if doc_id not in doc_meta:
                doc_meta[doc_id] = {}
        
        # Build DocumentClassifications
        classifications = DocumentClassifications()
        for doc_id, chunks in doc_chunks.items():
            payload: Dict[str, Any] = {
                "doc_id": doc_id,
                "chunks": chunks,
            }
            # Add any metadata we have
            meta = doc_meta.get(doc_id, {})
            if meta.get("published_at"):
                payload["published_at"] = meta["published_at"]
            if meta.get("title"):
                payload["title"] = meta["title"]
            if meta.get("url"):
                payload["url"] = meta["url"]
                
            classifications.append(DocumentClassification(payload=payload))
        
        return classifications


@dataclass
class Candidate:
    """Candidate passage with metadata used across induction/application."""

    item_id: str
    text: str
    ann_score: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)
    scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    utilities: Dict[str, float] = field(default_factory=dict)
