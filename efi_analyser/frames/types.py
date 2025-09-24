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


@dataclass
class FrameSchema:
    """Domain-specific collection of frames produced by the inducer."""

    domain: str
    frames: List[Frame]
    notes: str = ""


@dataclass
class FrameAssignment:
    """Probabilistic assignment of a candidate text to frames in a schema."""

    candidate: str
    frame_probabilities: Dict[str, float]
    candidate_id: Optional[str] = None
    top_frames: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
