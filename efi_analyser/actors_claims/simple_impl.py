from __future__ import annotations

import itertools
import re
import uuid
from typing import List, Optional, Tuple

from .protocols import ActorExtractor, ClaimExtractor, AttributionEngine, CombinedExtractor
from .types import (
    Attribution,
    Claim,
    ClaimMode,
    CombinedResult,
    Document,
    Mention,
    MentionType,
)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


class SimpleActorExtractor(ActorExtractor):
    """Very lightweight actor extractor based on regex heuristics.

    - PERSON: sequences of 2+ capitalized words, optionally preceded by a role.
    - ORG: capitalized phrases containing typical org indicators (e.g., 'Greenpeace', 'Ministry').

    This is intentionally simple to avoid heavy model dependencies in v0 tests.
    """

    ROLE_WORDS = {
        "president",
        "prime minister",
        "secretary",
        "governor",
        "mayor",
        "director",
        "professor",
        "dr",
        "mr",
        "mrs",
        "ms",
        "chair",
        "commissioner",
        "chief",
        "energy minister",
        "minister",
    }

    ORG_HINTS = {
        "ministry",
        "government",
        "university",
        "institute",
        "agency",
        "department",
        "committee",
        "council",
        "greenpeace",
        "company",
        "corp",
        "inc",
        "ltd",
        "gmbh",
    }

    # Rough pattern for capitalized multi-word spans (handles ASCII; adequate for tests)
    CAP_SEQ = re.compile(r"\b([A-Z][\w\-']+(?:\s+[A-Z][\w\-']+)+)\b")

    def extract(self, doc: Document) -> List[Mention]:
        text = doc.text
        mentions: List[Mention] = []

        # Find candidate capitalized sequences
        for m in self.CAP_SEQ.finditer(text):
            span_text = m.group(1)
            start, end = m.span(1)
            lower = span_text.lower()

            mtype = None
            if any(hint in lower for hint in self.ORG_HINTS):
                mtype = MentionType.ORG
            else:
                # If preceded by a role word within 25 chars, consider ROLE/Person
                pre_ctx = text[max(0, start - 25):start].lower()
                if any(rw in pre_ctx for rw in self.ROLE_WORDS):
                    mtype = MentionType.PERSON
                else:
                    mtype = MentionType.PERSON

            mentions.append(
                Mention(
                    mention_id=_new_id("m"),
                    doc_id=doc.doc_id,
                    text=span_text,
                    start_char=start,
                    end_char=end,
                    type=mtype,
                    confidence=0.6 if mtype == MentionType.ORG else 0.7,
                )
            )

        # Deduplicate exact same spans
        unique: dict[Tuple[int, int], Mention] = {}
        for mm in mentions:
            unique[(mm.start_char, mm.end_char)] = mm
        return list(unique.values())


class SimpleClaimExtractor(ClaimExtractor):
    """Find direct quotes and basic reported speech using regex patterns.

    Returns claims with evidence spans and mode. Offsets are 0-based [start,end).
    """

    # Match “quoted text” or "quoted text"
    QUOTE_PATTERNS = [
        re.compile(r"\“([^\”]+)\”"),  # smart double quotes
        re.compile(r'"([^\"]+)"'),     # ASCII double quotes
        re.compile(r"\‘([^\’]+)\’"),  # smart single quotes
    ]

    # Reported speech patterns (basic):
    # 1) According to X, CONTENT.
    ACCORDING_TO = re.compile(
        r"\b(?:According to|according to)\s+([^,]+?),\s+([^\.!?]+)", re.UNICODE
    )
    # 2) X said/stated/claimed/told (that) CONTENT.
    SAID_PAT = re.compile(
        r"\b([A-Z][^,\.;:]{0,80}?)\s+(?:said|says|stated|claimed|told|argued|added|noted|asserted)\s+(?:that\s+)?([^\.!?]+)",
        re.UNICODE,
    )

    def _clean_claim_text(self, s: str) -> str:
        return s.strip().strip(",;: ")

    def extract(self, doc: Document) -> List[Claim]:
        text = doc.text
        claims: List[Claim] = []

        # Direct quotes
        for pat in self.QUOTE_PATTERNS:
            for m in pat.finditer(text):
                ev = m.group(1)
                start, end = m.span(1)
                # Ignore very short or single-word scare quotes
                cleaned = self._clean_claim_text(ev)
                if len(cleaned) < 8 or not re.search(r"\s", cleaned):
                    continue
                claims.append(
                    Claim(
                        claim_id=_new_id("c"),
                        doc_id=doc.doc_id,
                        claim_text=cleaned,
                        evidence_text=ev,
                        start_char=start,
                        end_char=end,
                        mode=ClaimMode.DIRECT_QUOTE,
                        confidence=0.9,
                    )
                )

        # Reported speech
        for m in self.ACCORDING_TO.finditer(text):
            content = m.group(2)
            start, end = m.span(2)
            claims.append(
                Claim(
                    claim_id=_new_id("c"),
                    doc_id=doc.doc_id,
                    claim_text=self._clean_claim_text(content),
                    evidence_text=content,
                    start_char=start,
                    end_char=end,
                    mode=ClaimMode.REPORTED_SPEECH,
                    confidence=0.7,
                )
            )

        for m in self.SAID_PAT.finditer(text):
            content = m.group(2)
            start, end = m.span(2)
            claims.append(
                Claim(
                    claim_id=_new_id("c"),
                    doc_id=doc.doc_id,
                    claim_text=self._clean_claim_text(content),
                    evidence_text=content,
                    start_char=start,
                    end_char=end,
                    mode=ClaimMode.REPORTED_SPEECH,
                    confidence=0.7,
                )
            )

        # Deduplicate overlapping identical spans
        unique: dict[Tuple[int, int], Claim] = {}
        for cl in claims:
            key = (cl.start_char, cl.end_char)
            if key not in unique:
                unique[key] = cl
        return list(unique.values())


class SimpleAttributionEngine(AttributionEngine):
    """Link claims to nearby speaker mentions using simple patterns.

    - For direct quotes, search within a window for verbs like 'said' and take the NP.
    - For reported speech, reuse the capture groups from simple patterns when possible,
      otherwise pick the nearest preceding PERSON/ORG mention.
    """

    # After-quote: "...," said SPEAKER
    AFTER_QUOTE = re.compile(
        r"[,\s]*(?:said|says|stated|told|added)\s+([^\.;:]+)", re.UNICODE
    )
    # Before-quote: SPEAKER said 
    BEFORE_QUOTE = re.compile(
        r"([^\.;:]+?)\s+(?:said|says|stated|told|added)\s*$", re.UNICODE
    )

    def _find_speaker_near(self, text: str, start: int, end: int) -> Optional[Tuple[str, int, int]]:
        # Look right after the quote
        right = text[end : min(len(text), end + 120)]
        m = self.AFTER_QUOTE.match(right)
        if m:
            sp_text = m.group(1).strip().strip(", ")
            # Map back to doc offsets
            sp_start = end + m.start(1)
            sp_end = end + m.end(1)
            return sp_text, sp_start, sp_end
        # Look before the quote (line/sentence)
        left = text[max(0, start - 120) : start]
        m2 = self.BEFORE_QUOTE.search(left)
        if m2:
            sp_text = m2.group(1).strip().strip(", ")
            sp_start = start - (len(left) - m2.start(1))
            sp_end = start - (len(left) - m2.end(1))
            return sp_text, sp_start, sp_end
        return None

    def _nearest_mention(self, mentions: List[Mention], pos: int) -> Optional[Mention]:
        before = [m for m in mentions if m.start_char <= pos]
        if not before:
            return None
        return max(before, key=lambda m: m.start_char)

    def link(self, doc: Document, mentions: List[Mention], claims: List[Claim]) -> List[Attribution]:
        atts: List[Attribution] = []
        text = doc.text

        # Index mentions by span for quick lookup
        def find_matching_mention(span_start: int, span_end: int, span_text: str) -> Optional[Mention]:
            for m in mentions:
                if m.start_char == span_start and m.end_char == span_end:
                    return m
            # Fallback: exact text match overlapping
            for m in mentions:
                if m.text == span_text and not (m.end_char <= span_start or m.start_char >= span_end):
                    return m
            return None

        for i, cl in enumerate(claims):
            if cl.mode == ClaimMode.DIRECT_QUOTE:
                sp = self._find_speaker_near(text, cl.start_char, cl.end_char)
                if sp is not None:
                    s_text, s_start, s_end = sp
                    mm = find_matching_mention(s_start, s_end, s_text)
                    atts.append(
                        Attribution(
                            attribution_id=_new_id("a"),
                            doc_id=doc.doc_id,
                            claim_id=cl.claim_id,
                            speaker_text=s_text,
                            speaker_start_char=s_start,
                            speaker_end_char=s_end,
                            mode=cl.mode,
                            confidence=0.85,
                            mention_id=mm.mention_id if mm else None,
                        )
                    )
                    continue
                # Fallback: nearest preceding mention
                mm = self._nearest_mention(mentions, cl.start_char)
                if mm:
                    atts.append(
                        Attribution(
                            attribution_id=_new_id("a"),
                            doc_id=doc.doc_id,
                            claim_id=cl.claim_id,
                            speaker_text=mm.text,
                            speaker_start_char=mm.start_char,
                            speaker_end_char=mm.end_char,
                            mode=cl.mode,
                            confidence=0.55,
                            mention_id=mm.mention_id,
                        )
                    )
            else:  # Reported speech
                # Heuristic: take nearest preceding mention
                mm = self._nearest_mention(mentions, cl.start_char)
                if mm:
                    atts.append(
                        Attribution(
                            attribution_id=_new_id("a"),
                            doc_id=doc.doc_id,
                            claim_id=cl.claim_id,
                            speaker_text=mm.text,
                            speaker_start_char=mm.start_char,
                            speaker_end_char=mm.end_char,
                            mode=cl.mode,
                            confidence=0.7,
                            mention_id=mm.mention_id,
                        )
                    )

        return atts


class SimpleCombinedExtractor(CombinedExtractor):
    def __init__(
        self,
        actor_extractor: Optional[ActorExtractor] = None,
        claim_extractor: Optional[ClaimExtractor] = None,
        attribution_engine: Optional[AttributionEngine] = None,
    ):
        self.actor_extractor = actor_extractor or SimpleActorExtractor()
        self.claim_extractor = claim_extractor or SimpleClaimExtractor()
        self.attribution_engine = attribution_engine or SimpleAttributionEngine()

    def run(self, doc: Document) -> CombinedResult:
        mentions = self.actor_extractor.extract(doc)
        claims = self.claim_extractor.extract(doc)
        atts = self.attribution_engine.link(doc, mentions, claims)
        return CombinedResult(doc=doc, mentions=mentions, claims=claims, attributions=atts)
