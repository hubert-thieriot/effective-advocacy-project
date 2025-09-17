"""
NLI scorer using generic LLM backend.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .nli_scorer import NLIScorer
from .llm_scorer import LLMInterface, LLMScorerConfig


class NLILLMScorerConfig(LLMScorerConfig):
    """Configuration for NLI LLM scorer."""
    pass


class NLILLMScorer(NLIScorer):
    """NLI scorer using generic LLM backend with NLI-specific prompts."""

    def __init__(self, name: str = "nli_llm", config: Optional[NLILLMScorerConfig] = None, scorer_backend=None):
        """
        Initialize NLI LLM scorer.

        Args:
            name: Scorer name
            config: Configuration
            scorer_backend: LLM scorer backend to use (LLMScorer, etc.)
                           If None, uses default LLMScorer
        """
        # Initialize with NLI task
        super().__init__(name, config)
        self.config = config or NLILLMScorerConfig()

        # Use provided backend or create default
        if scorer_backend is not None:
            self._llm_interface = scorer_backend
        else:
            self._llm_interface = LLMInterface(name + "_base", config=self.config)

    def _build_nli_messages(self, premise: str, hypothesis: str) -> List[Dict[str, str]]:
        """Build NLI-specific messages for LLM."""
        system = (
            "You are a natural language inference (NLI) classifier.\n"
            "Given a PREMISE and a HYPOTHESIS, decide the relationship.\n\n"
            "Output ONLY JSON in the form:\n"
            "{\"label\": \"entails|contradicts|neutral\", \"score\": <0-1>, \"rationale\": \"<30 words>\"}\n\n"
            "RULES:\n"
            "- No text before/after JSON. Start with {, end with }.\n"
            "- Labels:\n"
            "  * entails: If PREMISE true → HYPOTHESIS must be true.\n"
            "  * contradicts: If PREMISE true → HYPOTHESIS must be false.\n"
            "  * neutral: PREMISE does not guarantee truth/falsehood of HYPOTHESIS.\n\n"
            "Decision procedure:\n"
            "1. Check entailment: Is HYPOTHESIS guaranteed by PREMISE? If yes, label entails.\n"
            "2. Else check contradiction: Does PREMISE make HYPOTHESIS impossible? If yes, label contradicts.\n"
            "3. Else label neutral.\n\n"
            "Scoring:\n"
            "- 0.85–1.0: very clear relation\n"
            "- 0.65–0.84: moderately clear\n"
            "- 0.50–0.64: weak evidence\n\n"
            "Rationale:\n"
            "- ≤30 words.\n"
            "- Explain why label applies; do not restate premise/hypothesis verbatim.\n"
        )

        user = (
            f"Premise: \"{premise}\"\n\n"
            f"Hypothesis: \"{hypothesis}\"\n\n"
            "Determine the NLI relationship.\n\n"
            "Examples:\n"
            "- Premise: \"All cats are mammals. Garfield is a cat.\" \n"
            "  Hypothesis: \"Garfield is a mammal.\" \n"
            "  → {\"label\": \"entails\", \"score\": 0.95, \"rationale\": \"Cats are mammals; Garfield is a cat; therefore Garfield is a mammal.\"}\n\n"
            "- Premise: \"The meeting is today.\" \n"
            "  Hypothesis: \"The meeting is tomorrow.\" \n"
            "  → {\"label\": \"contradicts\", \"score\": 0.90, \"rationale\": \"Meeting times directly conflict.\"}\n\n"
            "- Premise: \"Paris is in France.\" \n"
            "  Hypothesis: \"London is a city.\" \n"
            "  → {\"label\": \"neutral\", \"score\": 0.60, \"rationale\": \"Unrelated facts; neither entailment nor contradiction.\"}\n\n"
            "Now apply the same logic. Output ONLY JSON, nothing else."
        )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def batch_score(self, targets: List[str], passages: List[str]) -> List[Dict[str, float]]:
        """Score target-passage pairs for NLI using LLM."""
        if not passages:
            return [{"entails": 0.0, "contradicts": 0.0, "neutral": 0.0} for _ in passages]

        results = []
        total_pairs = len(targets)

        if self._llm_interface.config.verbose:
            print(f"🔬 Processing {total_pairs} NLI pairs with {self.name}...")

        for i, (target, passage) in enumerate(zip(targets, passages)):
            if self._llm_interface.config.verbose:
                print(f"  [{i+1}/{total_pairs}] Scoring premise-hypothesis pair...")

            # Build NLI-specific messages
            messages = self._build_nli_messages(target, passage)

            # Get raw LLM response
            raw_response = self._llm_interface.infer(messages)

            # Parse NLI response
            nli_scores = self._parse_nli_response(raw_response)
            results.append(nli_scores)

        if self._llm_interface.config.verbose:
            print(f"✅ Completed NLI scoring for {total_pairs} pairs with {self.name}")

        return results

    def _parse_nli_response(self, raw_response: str) -> Dict[str, float]:
        """Parse raw LLM response into NLI scores."""
        nli_label = None  # No default fallback label
        confidence = 0.0  # No default confidence

        try:
            import json
            parsed = json.loads(raw_response)
            nli_label = parsed.get("label")
            if nli_label:
                confidence = parsed.get("score", parsed.get("confidence", 0.0))  # Support both "score" and "confidence" keys
                # Ensure confidence is between 0 and 1
                confidence = max(0.0, min(1.0, confidence))
        except:
            # If JSON parsing fails, try to extract from text
            text_lower = raw_response.lower()
            if "entails" in text_lower:
                nli_label = "entails"
                confidence = 0.1  # Minimal confidence for text extraction
            elif "contradicts" in text_lower:
                nli_label = "contradicts"
                confidence = 0.1  # Minimal confidence for text extraction

        # Create NLI-style output
        nli_scores = {"entails": 0.0, "contradicts": 0.0, "neutral": 0.0}
        if nli_label and nli_label in nli_scores:
            nli_scores[nli_label] = confidence

        return nli_scores
