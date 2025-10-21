import os
import pytest

transformers = pytest.importorskip("transformers")

from efi_analyser.actors_claims import (
    Document,
    HfActorExtractor,
    HfCombinedExtractor,
)


# pytestmark = pytest.mark.skipif(
#     os.environ.get("RUN_HF_TESTS") != "1",
#     reason="Set RUN_HF_TESTS=1 to run HF-based tests (downloads models)",
# )


def _sample_text():
    return (
        '"Coal is vital for growth," said Energy Minister Arifin Tasrif. '
        "According to Greenpeace Indonesia, coal plants harm public health. "
        "Arifin Tasrif said that transition to renewables is necessary."
    )


def test_hf_actor_extractor_en_demo():
    doc = Document(doc_id="hf1", language="en", text=_sample_text())
    # English model with good performance and small size
    ext = HfActorExtractor(model_name="dslim/bert-base-NER")
    mentions = ext.extract(doc)
    assert len(mentions) >= 1
    mtexts = {m.text for m in mentions}
    # Prefer full name after merging subwords/spans
    assert any("Arifin Tasrif" in t for t in mtexts) or any("Tasrif" in t for t in mtexts)
    # Expect at least one org-like token (may be just "Greenpeace")
    assert any("Greenpeace" in t for t in mtexts)


def test_hf_combined_extractor_demo():
    doc = Document(doc_id="hf2", language="en", text=_sample_text())
    comb = HfCombinedExtractor(model_name="dslim/bert-base-NER")
    res = comb.run(doc)
    assert len(res.claims) >= 3
    assert len(res.mentions) >= 1
    # At least one attribution should be created
    assert len(res.attributions) >= 1
    # Offsets integrity for claims
    for c in res.claims:
        assert doc.text[c.start_char:c.end_char] == c.evidence_text
