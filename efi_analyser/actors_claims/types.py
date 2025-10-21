from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class MentionType(str, Enum):
    PERSON = "PERSON"
    ORG = "ORG"
    ROLE = "ROLE"
    PRONOUN = "PRONOUN"


class ClaimMode(str, Enum):
    DIRECT_QUOTE = "direct_quote"
    REPORTED_SPEECH = "reported_speech"


@dataclass
class Document:
    doc_id: str
    language: str
    text: str


@dataclass
class Mention:
    mention_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    type: MentionType
    sentence_id: Optional[int] = None
    confidence: Optional[float] = None


@dataclass
class Claim:
    claim_id: str
    doc_id: str
    claim_text: str
    evidence_text: str
    start_char: int
    end_char: int
    mode: ClaimMode
    sentence_id: Optional[int] = None
    confidence: Optional[float] = None


@dataclass
class Attribution:
    attribution_id: str
    doc_id: str
    claim_id: str
    speaker_text: str
    speaker_start_char: int
    speaker_end_char: int
    mode: ClaimMode
    confidence: float
    # Optional linkage to a specific mention_id if available
    mention_id: Optional[str] = None


@dataclass
class CombinedResult:
    doc: Document
    mentions: List[Mention] = field(default_factory=list)
    claims: List[Claim] = field(default_factory=list)
    attributions: List[Attribution] = field(default_factory=list)

