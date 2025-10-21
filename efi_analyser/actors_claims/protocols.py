from __future__ import annotations

from typing import Protocol, List

from .types import Document, Mention, Claim, Attribution, CombinedResult


class ActorExtractor(Protocol):
    def extract(self, doc: Document) -> List[Mention]:
        """Return entity mentions with 0-based [start,end) offsets.

        Implementations may return PERSON/ORG/ROLE mentions without canonicalization.
        """


class ClaimExtractor(Protocol):
    def extract(self, doc: Document) -> List[Claim]:
        """Return claims as evidence-backed spans (direct quotes and/or reported speech).

        `Claim.claim_text` may be a cleaned version of `evidence_text`, but `start_char/end_char`
        must match `evidence_text` in `doc.text[start:end]`.
        """


class AttributionEngine(Protocol):
    def link(self, doc: Document, mentions: List[Mention], claims: List[Claim]) -> List[Attribution]:
        """Link claims to speakers using mentions. Must return offsets for the speaker spans.

        Implementations should prefer spans that already exist in `mentions`, but may
        create additional speaker spans if needed.
        """


class CombinedExtractor(Protocol):
    def run(self, doc: Document) -> CombinedResult:
        """End-to-end convenience method that runs actor extraction, claim extraction,
        and attribution, returning a consolidated result.
        """

