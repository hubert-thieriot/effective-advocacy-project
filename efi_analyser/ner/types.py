"""Data structures for NER results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
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
    language: Union[str, List[str]] = "en"
    frame_threshold: float = 0.5
    consolidated: Optional["ConsolidatedNERResult"] = None

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
        result = {
            "language": self.language,
            "frame_threshold": self.frame_threshold,
            "documents": [d.to_dict() for d in self.documents],
        }
        if self.consolidated is not None:
            result["consolidated"] = self.consolidated.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NERResult":
        consolidated_data = data.get("consolidated")
        return cls(
            language=data.get("language", "en"),
            frame_threshold=data.get("frame_threshold", 0.5),
            documents=[DocumentEntities.from_dict(d) for d in data.get("documents", [])],
            consolidated=ConsolidatedNERResult.from_dict(consolidated_data) if consolidated_data else None,
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


@dataclass
class ConsolidatedEntity:
    """A consolidated entity grouping all variations."""

    entity_id: str
    canonical_name: str
    entity_type: str  # PERSON, ORG, EVENT, FAC, PRODUCT
    aliases: List[str] = field(default_factory=list)
    total_count: int = 0

    # Enrichment fields (extensible dict for future additions)
    attributes: Dict[str, Any] = field(default_factory=dict)
    # Current attributes:
    #   For PERSON: {"organization": "..."}
    #   For ORG: {"organization_type": "company|ngo|govt|media|research|other"}

    # Provenance
    original_types: Dict[str, int] = field(default_factory=dict)  # type -> count before correction
    frames: Dict[str, int] = field(default_factory=dict)  # frame_id -> count
    confidence: float = 1.0  # LLM confidence in consolidation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "canonical_name": self.canonical_name,
            "entity_type": self.entity_type,
            "aliases": self.aliases,
            "total_count": self.total_count,
            "attributes": self.attributes,
            "original_types": self.original_types,
            "frames": self.frames,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConsolidatedEntity":
        return cls(
            entity_id=data["entity_id"],
            canonical_name=data["canonical_name"],
            entity_type=data["entity_type"],
            aliases=data.get("aliases", []),
            total_count=data.get("total_count", 0),
            attributes=data.get("attributes", {}),
            original_types=data.get("original_types", {}),
            frames=data.get("frames", {}),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class ConsolidatedNERResult:
    """Consolidated NER output with grouped entities."""

    entities: List[ConsolidatedEntity] = field(default_factory=list)
    language: Union[str, List[str]] = "en"
    frame_threshold: float = 0.5

    # Consolidation metadata
    consolidation_model: str = "gpt-4o-mini"
    raw_entity_count: int = 0
    consolidated_entity_count: int = 0
    type_corrections: Dict[str, Dict[str, int]] = field(default_factory=dict)  # original -> corrected -> count

    @property
    def n_entities(self) -> int:
        return len(self.entities)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "language": self.language,
            "frame_threshold": self.frame_threshold,
            "consolidation_model": self.consolidation_model,
            "raw_entity_count": self.raw_entity_count,
            "consolidated_entity_count": self.consolidated_entity_count,
            "type_corrections": self.type_corrections,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConsolidatedNERResult":
        return cls(
            entities=[ConsolidatedEntity.from_dict(e) for e in data.get("entities", [])],
            language=data.get("language", "en"),
            frame_threshold=data.get("frame_threshold", 0.5),
            consolidation_model=data.get("consolidation_model", "gpt-4o-mini"),
            raw_entity_count=data.get("raw_entity_count", 0),
            consolidated_entity_count=data.get("consolidated_entity_count", 0),
            type_corrections=data.get("type_corrections", {}),
        )

    def save(self, path: Path) -> None:
        """Save consolidated NER results to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "ConsolidatedNERResult":
        """Load consolidated NER results from JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)


__all__ = [
    "EntityMention",
    "ChunkEntities",
    "DocumentEntities",
    "NERResult",
    "ConsolidatedEntity",
    "ConsolidatedNERResult",
]
