from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import re
import uuid

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    from transformers import pipeline  # type: ignore
except Exception:  # pragma: no cover
    pipeline = None  # type: ignore

from .protocols import ActorExtractor, CombinedExtractor, ClaimExtractor, AttributionEngine
from .types import (
    Mention,
    MentionType,
    Document,
    CombinedResult,
)
from .simple_impl import SimpleClaimExtractor, SimpleAttributionEngine
from .spacy_impl import SpacyClaimExtractor, SpacyAttributionEngine


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _default_hf_model(language: str) -> str:
    # Prefer small, commonly cached models
    if language.startswith("en"):
        return "dslim/bert-base-NER"
    # Multilingual high-resource model; decent generalization without being huge
    return "Davlan/xlm-roberta-base-ner-hrl"


class HfActorExtractor(ActorExtractor):
    """HuggingFace Transformer-based NER for PERSON/ORG mentions.

    - Uses token-classification pipeline with grouped entities and char offsets.
    - Defaults to an English model for en*, and a multilingual model otherwise.
    - Returns Mention objects with 0-based [start,end) offsets.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[int] = None,
        aggregation_strategy: str = "simple",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.aggregation_strategy = aggregation_strategy
        self._pipes: Dict[str, object] = {}

    def _get_pipe(self, language: str) -> Optional[object]:
        if pipeline is None:  # transformers missing
            return None
        model = self.model_name or _default_hf_model(language)
        if model in self._pipes:
            return self._pipes[model]
        dev = self.device
        if dev is None:
            if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                dev = 0
            else:
                dev = -1
        # grouped_entities improves handling of subword tokens on older transformers versions
        nlp = pipeline(
            "token-classification",
            model=model,
            aggregation_strategy=self.aggregation_strategy,
            device=dev,
            grouped_entities=True,
        )
        self._pipes[model] = nlp
        return nlp

    def extract(self, doc: Document) -> List[Mention]:
        nlp = self._get_pipe(doc.language)
        if nlp is None:
            return []
        outputs = nlp(doc.text)
        mentions: List[Mention] = []
        
        def is_name_char(ch: str) -> bool:
            return ch.isalpha() or ch in {"'", "’", "-"}

        def expand_to_word(text: str, start: int, end: int) -> tuple[int, int]:
            # Expand left if the span starts mid-token
            while start > 0 and is_name_char(text[start - 1]) and is_name_char(text[start]):
                start -= 1
            # Expand right if the span ends mid-token
            while end < len(text) and end > 0 and is_name_char(text[end]) and is_name_char(text[end - 1]):
                end += 1
            return start, end
        for ent in outputs:
            # transformers pipeline returns keys: entity_group, score, word, start, end
            label = ent.get("entity_group") or ent.get("entity")
            if label is None:
                continue
            label = str(label)
            if label.startswith("I-") or label.startswith("B-"):
                label = label[2:]
            if label in {"PER", "PERSON"}:
                mtype = MentionType.PERSON
            elif label in {"ORG", "ORGANIZATION"}:
                mtype = MentionType.ORG
            else:
                continue
            start = int(ent.get("start", -1))
            end = int(ent.get("end", -1))
            if start < 0 or end <= start:
                continue
            # Adjust to token boundaries and slice from the original doc text to avoid subword artifacts
            start, end = expand_to_word(doc.text, start, end)
            text = doc.text[start:end]
            mentions.append(
                Mention(
                    mention_id=_new_id("m"),
                    doc_id=doc.doc_id,
                    text=text,
                    start_char=start,
                    end_char=end,
                    type=mtype,
                    confidence=float(ent.get("score", 0.0)),
                )
            )
        # Merge overlapping/adjacent spans of the same type to consolidate full names
        mentions.sort(key=lambda m: (m.type.value, m.start_char, m.end_char))
        merged: List[Mention] = []
        for m in mentions:
            if not merged:
                merged.append(m)
                continue
            last = merged[-1]
            if m.type == last.type and m.start_char <= last.end_char + 1:
                # Extend last span
                new_start = min(last.start_char, m.start_char)
                new_end = max(last.end_char, m.end_char)
                new_text = doc.text[new_start:new_end]
                last.start_char = new_start
                last.end_char = new_end
                last.text = new_text
                last.confidence = max((last.confidence or 0.0), (m.confidence or 0.0))
            else:
                merged.append(m)
        # Deduplicate exact spans
        dedup: Dict[Tuple[int, int, str], Mention] = {}
        for m in merged:
            key = (m.start_char, m.end_char, m.type.value)
            prev = dedup.get(key)
            if prev is None or (m.confidence or 0) > (prev.confidence or 0):
                dedup[key] = m
        return list(dedup.values())


class HfCombinedExtractor(CombinedExtractor):
    """Convenience stack using HF NER for actors, regex/sent-based claims, and simple/spaCy attribution."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[int] = None,
        use_spacy_helpers: bool = True,
    ) -> None:
        self.actor_extractor = HfActorExtractor(model_name=model_name, device=device)
        # Prefer spaCy sentence splitting when available
        self.claim_extractor: ClaimExtractor = SpacyClaimExtractor() if use_spacy_helpers else SimpleClaimExtractor()
        self.attribution_engine: AttributionEngine = (
            SpacyAttributionEngine() if use_spacy_helpers else SimpleAttributionEngine()
        )

    def run(self, doc: Document) -> CombinedResult:
        mentions = self.actor_extractor.extract(doc)
        claims = self.claim_extractor.extract(doc)
        atts = self.attribution_engine.link(doc, mentions, claims)
        return CombinedResult(doc=doc, mentions=mentions, claims=claims, attributions=atts)
