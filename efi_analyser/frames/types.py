"""Data structures for frame induction and assignment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Frame:
    """Definition of a single media frame within a schema."""

    frame_id: str
    name: str
    description: str
    keywords: List[str]
    examples: List[str]
    short_name: str = ""


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


@dataclass
class Candidate:
    """Candidate passage with metadata used across induction/application."""

    item_id: str
    text: str
    ann_score: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)
    scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    utilities: Dict[str, float] = field(default_factory=dict)
