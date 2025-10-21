import re
from typing import List

import pytest

from efi_analyser.actors_claims import (
    Document,
    SimpleActorExtractor,
    SimpleClaimExtractor,
    SimpleAttributionEngine,
    SimpleCombinedExtractor,
    ClaimMode,
)


def test_combined_extractor_end_to_end():
    # Dummy multilingual-ish news snippet with direct and reported speech
    text = (
        '"Coal is vital for growth," said Energy Minister Arifin Tasrif. '
        "According to Greenpeace Indonesia, coal plants harm public health. "
        "Arifin Tasrif said that transition to renewables is necessary."
    )

    doc = Document(doc_id="doc1", language="en", text=text)
    extractor = SimpleCombinedExtractor()
    result = extractor.run(doc)

    # Basic sanity checks
    assert len(result.claims) >= 3, "Should extract at least 3 claims"
    assert len(result.mentions) >= 2, "Should find at least 2 mentions"

    # Offsets should map to evidence text exactly
    for cl in result.claims:
        assert doc.text[cl.start_char:cl.end_char] == cl.evidence_text
        assert cl.mode in (ClaimMode.DIRECT_QUOTE, ClaimMode.REPORTED_SPEECH)

    # Expect these specific extractions
    claim_texts = {cl.claim_text for cl in result.claims}
    assert "Coal is vital for growth" in claim_texts
    assert "coal plants harm public health" in claim_texts
    assert "transition to renewables is necessary" in claim_texts

    # Attribution should exist for each claim (may be heuristic)
    # Note: For a simple baseline, allow at least 2 confident links
    assert len(result.attributions) >= 2

    # Check at least one attribution to each expected speaker
    speakers = {a.speaker_text.strip() for a in result.attributions}
    # The direct quote should attribute to Energy Minister Arifin Tasrif
    assert any("Arifin Tasrif" in s for s in speakers)
    assert any("Greenpeace Indonesia" in s for s in speakers)


def _dummy_doc():
    text = (
        '"Coal is vital for growth," said Energy Minister Arifin Tasrif. '
        "According to Greenpeace Indonesia, coal plants harm public health. "
        "Arifin Tasrif said that transition to renewables is necessary."
    )
    return Document(doc_id="docA", language="en", text=text)


def test_simple_actor_extractor_mentions():
    doc = _dummy_doc()
    ext = SimpleActorExtractor()
    mentions = ext.extract(doc)
    mtexts = {m.text for m in mentions}
    assert any("Arifin Tasrif" in m for m in mtexts)
    assert any("Greenpeace Indonesia" in m for m in mtexts)


def test_simple_claim_extractor_spans():
    doc = _dummy_doc()
    ext = SimpleClaimExtractor()
    claims = ext.extract(doc)
    texts = {c.claim_text for c in claims}
    assert "Coal is vital for growth" in texts
    assert "coal plants harm public health" in texts
    assert "transition to renewables is necessary" in texts
    # Evidence span check
    for c in claims:
        assert doc.text[c.start_char:c.end_char] == c.evidence_text


def test_simple_attribution_engine_links():
    doc = _dummy_doc()
    mentions = SimpleActorExtractor().extract(doc)
    claims = SimpleClaimExtractor().extract(doc)
    atts = SimpleAttributionEngine().link(doc, mentions, claims)
    assert len(atts) >= 2
    speakers = {a.speaker_text for a in atts}
    assert any("Arifin Tasrif" in s for s in speakers)
    assert any("Greenpeace Indonesia" in s for s in speakers)


def test_no_claims_without_reporting_or_quotes():
    text = (
        "The ministry released a report on air quality. "
        "Data show particulate levels increased in several regions."
    )
    doc = Document(doc_id="doc_nil", language="en", text=text)
    claims = SimpleClaimExtractor().extract(doc)
    assert len(claims) == 0
    atts = SimpleAttributionEngine().link(doc, [], claims)
    assert len(atts) == 0


def test_ignore_scare_quotes_single_word():
    text = 'The so-called "green" policies are controversial.'
    doc = Document(doc_id="doc_scare", language="en", text=text)
    claims = SimpleClaimExtractor().extract(doc)
    # Should not treat single-word scare quotes as a claim
    assert len([c for c in claims if c.mode == ClaimMode.DIRECT_QUOTE]) == 0


def test_no_attribution_when_no_speaker_near_quote():
    text = '"This is great." The event was attended by many.'
    doc = Document(doc_id="doc_quote_only", language="en", text=text)
    mentions = SimpleActorExtractor().extract(doc)
    claims = SimpleClaimExtractor().extract(doc)
    # We expect a claim extracted but no clear speaker mention nearby
    assert len(claims) >= 1
    atts = SimpleAttributionEngine().link(doc, mentions, claims)
    # Heuristics shouldn't fabricate a speaker if none is nearby
    assert len(atts) == 0
