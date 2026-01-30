"""Data structures for NER results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json


@dataclass
class EntityMention:
    """A single entity mention extracted from text."""

    text: str
    type: str  # PERSON, ORG, GPE, LOC, etc.
    start_char: int
    end_char: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "type": self.type,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntityMention":
        return cls(
            text=data["text"],
            type=data["type"],
            start_char=data["start_char"],
            end_char=data["end_char"],
        )


@dataclass
class ChunkEntities:
    """Entities extracted from a single chunk with significant framing."""

    chunk_id: str
    text: str
    frame_id: str
    frame_score: float
    entities: List[EntityMention] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "frame_id": self.frame_id,
            "frame_score": self.frame_score,
            "entities": [e.to_dict() for e in self.entities],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkEntities":
        return cls(
            chunk_id=data["chunk_id"],
            text=data["text"],
            frame_id=data["frame_id"],
            frame_score=data["frame_score"],
            entities=[EntityMention.from_dict(e) for e in data.get("entities", [])],
        )


@dataclass
class DocumentEntities:
    """All entity extractions for a single document."""

    doc_id: str
    chunks: List[ChunkEntities] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "chunks": [c.to_dict() for c in self.chunks],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentEntities":
        return cls(
            doc_id=data["doc_id"],
            chunks=[ChunkEntities.from_dict(c) for c in data.get("chunks", [])],
        )


@dataclass
class NERResult:
    """Complete NER extraction result for the pipeline."""

    documents: List[DocumentEntities] = field(default_factory=list)
    language: str = "en"
    frame_threshold: float = 0.5

    @property
    def n_documents(self) -> int:
        return len(self.documents)

    @property
    def n_chunks(self) -> int:
        return sum(len(doc.chunks) for doc in self.documents)

    @property
    def n_entities(self) -> int:
        return sum(
            len(chunk.entities)
            for doc in self.documents
            for chunk in doc.chunks
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "frame_threshold": self.frame_threshold,
            "documents": [d.to_dict() for d in self.documents],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NERResult":
        return cls(
            language=data.get("language", "en"),
            frame_threshold=data.get("frame_threshold", 0.5),
            documents=[DocumentEntities.from_dict(d) for d in data.get("documents", [])],
        )

    def save(self, path: Path) -> None:
        """Save NER results to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "NERResult":
        """Load NER results from JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)


__all__ = [
    "EntityMention",
    "ChunkEntities",
    "DocumentEntities",
    "NERResult",
]
