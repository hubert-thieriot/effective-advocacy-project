"""Frame induction toolkit for EFI analyser."""

from .types import Candidate, Frame, FrameSchema, FrameAssignment, FrameAssignments
from .induction import FrameInducer
from .applicator import LLMFrameAnnotator, LLMFrameApplicator
from .corpora import EmbeddedCorpora
from . import classifier

__all__ = [
    "Candidate",
    "Frame",
    "FrameSchema",
    "FrameAssignment",
    "FrameAssignments",
    "FrameInducer",
    "LLMFrameAnnotator",
    "LLMFrameApplicator",  # backwards-compatible alias
    "EmbeddedCorpora",
    "classifier",
]
