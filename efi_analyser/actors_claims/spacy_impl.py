from __future__ import annotations

from typing import List, Optional, Tuple

try:
    import spacy
    from spacy.language import Language
except Exception:  # pragma: no cover - optional dependency at runtime
    spacy = None  # type: ignore
    Language = None  # type: ignore

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

import re
import uuid


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _default_spacy_for_lang(lang: str) -> Optional["Language"]:
    if spacy is None:
        return None
    # Best-effort: try language-specific small model, then multilingual, then blank
    candidates = []
    if lang.startswith("en"):
        candidates = [
            "en_core_web_sm",
            "xx_ent_wiki_sm",
        ]
    elif lang.startswith("de"):
        candidates = [
            "de_core_news_sm",
            "xx_ent_wiki_sm",
        ]
    elif lang.startswith("id"):
        # spaCy has no official Indonesian NER; rely on xx or blank
        candidates = ["xx_ent_wiki_sm"]
    else:
        candidates = ["xx_ent_wiki_sm"]

    for name in candidates:
        try:
            return spacy.load(name)
        except Exception:
            continue

    try:
        nlp = spacy.blank(lang)
    except Exception:
        nlp = spacy.blank("xx")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


class SpacyActorExtractor(ActorExtractor):
    """Actor extractor using spaCy NER if present; otherwise returns empty list.

    For tests or constrained environments, you can pass a custom `nlp` with
    an `EntityRuler` to simulate NER without downloading large models.
    """

    def __init__(self, nlp: Optional["Language"] = None):
        self._nlp = nlp

    def extract(self, doc: Document) -> List[Mention]:
        if spacy is None:
            return []
        nlp = self._nlp or _default_spacy_for_lang(doc.language)
        if nlp is None:
            return []
        sdoc = nlp(doc.text)
        mentions: List[Mention] = []
        for ent in sdoc.ents:
            if ent.label_ in {"PERSON", "ORG"}:
                mtype = MentionType.PERSON if ent.label_ == "PERSON" else MentionType.ORG
                mentions.append(
                    Mention(
                        mention_id=_new_id("m"),
                        doc_id=doc.doc_id,
                        text=ent.text,
                        start_char=ent.start_char,
                        end_char=ent.end_char,
                        type=mtype,
                        confidence=0.85,
                    )
                )
        return mentions


class SpacyClaimExtractor(ClaimExtractor):
    """Claims based on quotes and simple reported speech patterns, with sentence awareness.

    Uses spaCy for sentence segmentation when available; otherwise behaves like simple regex.
    """

    QUOTE_PATTERNS = [
        re.compile(r"\“([^\”]+)\”"),  # smart double quotes
        re.compile(r'"([^\"]+)"'),     # ASCII double quotes
        re.compile(r"\‘([^\’]+)\’"),  # smart single quotes
    ]
    ACCORDING_TO = re.compile(
        r"\b(?:According to|according to)\s+([^,]+?),\s+([^\.!?]+)", re.UNICODE
    )
    SAID_PAT = re.compile(
        r"\b([A-Z][^,\.;:]{0,80}?)\s+(?:said|says|stated|claimed|told|argued|added|noted|asserted)\s+(?:that\s+)?([^\.!?]+)",
        re.UNICODE,
    )

    def __init__(self, nlp: Optional["Language"] = None):
        self._nlp = nlp

    def _clean(self, s: str) -> str:
        return s.strip().strip(",;: ")

    def extract(self, doc: Document) -> List[Claim]:
        text = doc.text
        claims: List[Claim] = []

        # Direct quotes first (doc-level regex)
        for pat in self.QUOTE_PATTERNS:
            for m in pat.finditer(text):
                ev = m.group(1)
                start, end = m.span(1)
                cleaned = self._clean(ev)
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

        # Sentence-aware reported speech
        nlp = self._nlp or (spacy and _default_spacy_for_lang(doc.language))
        if nlp is not None:
            sdoc = nlp(text)
            for sent in sdoc.sents:
                stext = sent.text
                offset0 = sent.start_char
                for m in self.ACCORDING_TO.finditer(stext):
                    content = m.group(2)
                    s_start, s_end = m.span(2)
                    start = offset0 + s_start
                    end = offset0 + s_end
                    claims.append(
                        Claim(
                            claim_id=_new_id("c"),
                            doc_id=doc.doc_id,
                            claim_text=self._clean(content),
                            evidence_text=content,
                            start_char=start,
                            end_char=end,
                            mode=ClaimMode.REPORTED_SPEECH,
                            confidence=0.75,
                        )
                    )
                for m in self.SAID_PAT.finditer(stext):
                    content = m.group(2)
                    s_start, s_end = m.span(2)
                    start = offset0 + s_start
                    end = offset0 + s_end
                    claims.append(
                        Claim(
                            claim_id=_new_id("c"),
                            doc_id=doc.doc_id,
                            claim_text=self._clean(content),
                            evidence_text=content,
                            start_char=start,
                            end_char=end,
                            mode=ClaimMode.REPORTED_SPEECH,
                            confidence=0.75,
                        )
                    )
        else:
            # Fallback: doc-level regex
            for m in self.ACCORDING_TO.finditer(text):
                content = m.group(2)
                start, end = m.span(2)
                claims.append(
                    Claim(
                        claim_id=_new_id("c"),
                        doc_id=doc.doc_id,
                        claim_text=self._clean(content),
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
                        claim_text=self._clean(content),
                        evidence_text=content,
                        start_char=start,
                        end_char=end,
                        mode=ClaimMode.REPORTED_SPEECH,
                        confidence=0.7,
                    )
                )

        # Deduplicate by span
        seen = {}
        out: List[Claim] = []
        for c in claims:
            key = (c.start_char, c.end_char)
            if key in seen:
                continue
            seen[key] = True
            out.append(c)
        return out


class SpacyAttributionEngine(AttributionEngine):
    """Attribution using sentence window and spaCy ents if available.

    For direct quotes, favor speakers in the same sentence and around reporting verbs.
    For reported speech, pick nearest preceding entity in the sentence.
    Falls back to simple proximity if spaCy is unavailable.
    """

    VERBS = {"said", "says", "stated", "told", "added", "argued", "noted", "asserted"}

    def __init__(self, nlp: Optional["Language"] = None):
        self._nlp = nlp

    def _sentences(self, text: str) -> List[Tuple[int, int]]:
        if spacy is None:
            return [(0, len(text))]
        nlp = self._nlp or _default_spacy_for_lang("en")
        if nlp is None:
            return [(0, len(text))]
        doc = nlp(text)
        return [(s.start_char, s.end_char) for s in doc.sents]

    def link(self, doc: Document, mentions: List[Mention], claims: List[Claim]) -> List[Attribution]:
        atts: List[Attribution] = []
        sents = self._sentences(doc.text)

        def find_sent(pos: int) -> Tuple[int, int]:
            for s_start, s_end in sents:
                if s_start <= pos < s_end:
                    return s_start, s_end
            return 0, len(doc.text)

        def ents_in_span(a: int, b: int) -> List[Mention]:
            return [m for m in mentions if not (m.end_char <= a or m.start_char >= b)]

        for cl in claims:
            s_start, s_end = find_sent(cl.start_char)
            cand_ents = ents_in_span(s_start, s_end)
            # Prefer nearest preceding mention in the sentence
            cand_ents.sort(key=lambda m: (0 if m.start_char <= cl.start_char else 1, -m.start_char))
            if cand_ents:
                mm = cand_ents[0]
                atts.append(
                    Attribution(
                        attribution_id=_new_id("a"),
                        doc_id=doc.doc_id,
                        claim_id=cl.claim_id,
                        speaker_text=mm.text,
                        speaker_start_char=mm.start_char,
                        speaker_end_char=mm.end_char,
                        mode=cl.mode,
                        confidence=0.8 if cl.mode == ClaimMode.DIRECT_QUOTE else 0.7,
                        mention_id=mm.mention_id,
                    )
                )
                continue
            # Fallback: no entity in sentence; skip to avoid fabricating
        return atts


class SpacyCombinedExtractor(CombinedExtractor):
    def __init__(
        self,
        nlp: Optional["Language"] = None,
        actor_extractor: Optional[ActorExtractor] = None,
        claim_extractor: Optional[ClaimExtractor] = None,
        attribution_engine: Optional[AttributionEngine] = None,
    ):
        self.nlp = nlp
        self.actor_extractor = actor_extractor or SpacyActorExtractor(nlp)
        self.claim_extractor = claim_extractor or SpacyClaimExtractor(nlp)
        self.attribution_engine = attribution_engine or SpacyAttributionEngine(nlp)

    def run(self, doc: Document) -> CombinedResult:
        mentions = self.actor_extractor.extract(doc)
        claims = self.claim_extractor.extract(doc)
        atts = self.attribution_engine.link(doc, mentions, claims)
        return CombinedResult(doc=doc, mentions=mentions, claims=claims, attributions=atts)

