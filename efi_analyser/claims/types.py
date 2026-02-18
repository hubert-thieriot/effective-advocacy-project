"""Data structures for claims analysis.

This module implements Discourse Network Analysis (DNA) methodology where:
- Statements: What actors say in text chunks
- Claims: Specific policy positions/beliefs (NOT abstract themes)
- Agreements: How statements align with claims (support/oppose/neutral)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json


@dataclass
class Statement:
    """A statement extracted from a chunk with an identified speaker.

    Represents what an actor says in a text chunk. Links back to the
    NER-identified entities for speaker attribution.
    """

    statement_id: str
    chunk_id: str
    doc_id: str
    text: str
    speaker: Optional[str] = None  # Actor name from NER
    speaker_entity_id: Optional[str] = None  # Link to consolidated entity
    context: str = ""  # Surrounding chunk text for context

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statement_id": self.statement_id,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
            "speaker": self.speaker,
            "speaker_entity_id": self.speaker_entity_id,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Statement":
        return cls(
            statement_id=data["statement_id"],
            chunk_id=data["chunk_id"],
            doc_id=data["doc_id"],
            text=data["text"],
            speaker=data.get("speaker"),
            speaker_entity_id=data.get("speaker_entity_id"),
            context=data.get("context", ""),
        )


@dataclass
class Claim:
    """A specific policy position/belief for measuring alignment.

    Claims should be concrete, actionable policy positions, NOT abstract themes.

    Good examples:
    - "DNA testing should be mandatory for racehorses"
    - "The Grand National course should be shortened"
    - "Factory farming should be banned"
    - "We should adopt cap and trade for carbon emissions"

    Bad examples (too abstract):
    - "Animal welfare" (not a position)
    - "Environmental impact" (not actionable)
    - "Climate change" (too vague)
    """

    claim_id: str
    text: str  # The specific policy position
    description: str = ""  # Clarification/context
    keywords: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)  # Supporting statement examples
    counter_examples: List[str] = field(default_factory=list)  # Opposing statement examples
    source: str = "induced"  # "induced" or "manual"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "description": self.description,
            "keywords": self.keywords,
            "examples": self.examples,
            "counter_examples": self.counter_examples,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Claim":
        return cls(
            claim_id=data["claim_id"],
            text=data["text"],
            description=data.get("description", ""),
            keywords=data.get("keywords", []),
            examples=data.get("examples", []),
            counter_examples=data.get("counter_examples", []),
            source=data.get("source", "induced"),
        )


@dataclass
class ClaimSchema:
    """Domain-specific collection of claims for analysis."""

    domain: str
    claims: List[Claim] = field(default_factory=list)
    notes: str = ""
    schema_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "domain": self.domain,
            "claims": [c.to_dict() for c in self.claims],
            "notes": self.notes,
            "schema_id": self.schema_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClaimSchema":
        return cls(
            domain=data.get("domain", ""),
            claims=[Claim.from_dict(c) for c in data.get("claims", [])],
            notes=data.get("notes", ""),
            schema_id=data.get("schema_id", ""),
        )

    def save(self, path: Path) -> None:
        """Save claim schema to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "ClaimSchema":
        """Load claim schema from JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)


@dataclass
class Agreement:
    """Agreement score between a statement and a claim.

    Represents how a statement aligns with a specific policy claim.
    Score ranges from -1.0 (strongly opposes) to 1.0 (strongly supports).
    """

    statement_id: str
    claim_id: str
    score: float  # -1.0 (opposes) to 1.0 (supports)
    label: str  # "supports", "opposes", "neutral", "irrelevant"
    rationale: str = ""
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statement_id": self.statement_id,
            "claim_id": self.claim_id,
            "score": self.score,
            "label": self.label,
            "rationale": self.rationale,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Agreement":
        return cls(
            statement_id=data["statement_id"],
            claim_id=data["claim_id"],
            score=data["score"],
            label=data["label"],
            rationale=data.get("rationale", ""),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class StatementsResult:
    """Collection of extracted statements."""

    statements: List[Statement] = field(default_factory=list)
    extraction_model: str = ""

    @property
    def n_statements(self) -> int:
        return len(self.statements)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statements": [s.to_dict() for s in self.statements],
            "extraction_model": self.extraction_model,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StatementsResult":
        return cls(
            statements=[Statement.from_dict(s) for s in data.get("statements", [])],
            extraction_model=data.get("extraction_model", ""),
        )

    def save(self, path: Path) -> None:
        """Save statements to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "StatementsResult":
        """Load statements from JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)


@dataclass
class AgreementsResult:
    """Collection of statement-claim agreements."""

    agreements: List[Agreement] = field(default_factory=list)
    scoring_model: str = ""

    @property
    def n_agreements(self) -> int:
        return len(self.agreements)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agreements": [a.to_dict() for a in self.agreements],
            "scoring_model": self.scoring_model,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgreementsResult":
        return cls(
            agreements=[Agreement.from_dict(a) for a in data.get("agreements", [])],
            scoring_model=data.get("scoring_model", ""),
        )

    def save(self, path: Path) -> None:
        """Save agreements to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "AgreementsResult":
        """Load agreements from JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def by_statement(self) -> Dict[str, List[Agreement]]:
        """Group agreements by statement_id."""
        result: Dict[str, List[Agreement]] = {}
        for agreement in self.agreements:
            if agreement.statement_id not in result:
                result[agreement.statement_id] = []
            result[agreement.statement_id].append(agreement)
        return result

    def by_claim(self) -> Dict[str, List[Agreement]]:
        """Group agreements by claim_id."""
        result: Dict[str, List[Agreement]] = {}
        for agreement in self.agreements:
            if agreement.claim_id not in result:
                result[agreement.claim_id] = []
            result[agreement.claim_id].append(agreement)
        return result


@dataclass
class ClaimsAnalysisResult:
    """Complete result from the claims analysis stage."""

    statements: StatementsResult
    claims: ClaimSchema
    agreements: AgreementsResult

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statements": self.statements.to_dict(),
            "claims": self.claims.to_dict(),
            "agreements": self.agreements.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClaimsAnalysisResult":
        return cls(
            statements=StatementsResult.from_dict(data.get("statements", {})),
            claims=ClaimSchema.from_dict(data.get("claims", {})),
            agreements=AgreementsResult.from_dict(data.get("agreements", {})),
        )


__all__ = [
    "Statement",
    "Claim",
    "ClaimSchema",
    "Agreement",
    "StatementsResult",
    "AgreementsResult",
    "ClaimsAnalysisResult",
]
