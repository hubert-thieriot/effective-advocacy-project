import pytest

from efi_analyser.actors_claims import (
    Document,
    SpacyActorExtractor,
    SpacyClaimExtractor,
    SpacyAttributionEngine,
    SpacyCombinedExtractor,
)


def _make_nlp_with_ruler():
    import spacy
    nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns([
        {"label": "PERSON", "pattern": "Energy Minister Arifin Tasrif"},
        {"label": "PERSON", "pattern": "Arifin Tasrif"},
        {"label": "ORG", "pattern": "Greenpeace Indonesia"},
    ])
    return nlp


@pytest.mark.skipif(False, reason="spaCy available via dependency; using blank pipeline")
def test_spacy_actor_extractor_with_ruler():
    text = (
        '"Coal is vital for growth," said Energy Minister Arifin Tasrif. '
        "According to Greenpeace Indonesia, coal plants harm public health. "
        "Arifin Tasrif said that transition to renewables is necessary."
    )
    doc = Document(doc_id="d1", language="en", text=text)
    nlp = _make_nlp_with_ruler()
    ext = SpacyActorExtractor(nlp)
    mentions = ext.extract(doc)
    mtexts = {m.text for m in mentions}
    assert any("Arifin Tasrif" in m for m in mtexts)
    assert any("Greenpeace Indonesia" in m for m in mtexts)


def test_spacy_combined_with_ruler_end_to_end():
    text = (
        '"Coal is vital for growth," said Energy Minister Arifin Tasrif. '
        "According to Greenpeace Indonesia, coal plants harm public health. "
        "Arifin Tasrif said that transition to renewables is necessary."
    )
    doc = Document(doc_id="d2", language="en", text=text)
    nlp = _make_nlp_with_ruler()
    comb = SpacyCombinedExtractor(nlp)
    result = comb.run(doc)
    assert len(result.claims) >= 3
    assert len(result.mentions) >= 2
    assert len(result.attributions) >= 2
    # Evidence spans are valid
    for cl in result.claims:
        assert doc.text[cl.start_char:cl.end_char] == cl.evidence_text

