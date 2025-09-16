"""
NLI scorer using OpenAI GPT models.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..scorers.nli_scorer import NLIScorer
from .openai_interface import OpenAIInterface, OpenAIConfig


class NLIOpenAIConfig(OpenAIConfig):
    """Configuration for NLI OpenAI scorer."""
    pass


class NLIOpenAIScorer(NLIScorer):
    """NLI scorer using OpenAI GPT models with NLI-specific prompts."""

    def __init__(self, name: str = "nli_openai", config: Optional[NLIOpenAIConfig] = None, scorer_backend=None):
        """
        Initialize NLI OpenAI scorer.

        Args:
            name: Scorer name
            config: Configuration
            scorer_backend: OpenAI interface backend to use
                           If None, uses default OpenAIInterface
        """
        # Initialize with NLI task
        super().__init__(name, config)
        self.config = config or NLIOpenAIConfig()

        # Use provided backend or create default
        if scorer_backend is not None:
            self._openai_interface = scorer_backend
        else:
            self._openai_interface = OpenAIInterface(name + "_base", config=self.config)

    def _build_nli_messages(self, premise: str, hypothesis: str) -> List[Dict[str, str]]:
        """Build NLI-specific messages for OpenAI."""
        system = (
            "Determine the relationship between the premise and hypothesis. "
            "Output ONLY JSON: {\"label\": \"entails|contradicts|neutral\", \"score\": <0-1>, \"rationale\": \"<30 words>\"}.\n\n"
            "RULES: NO text before/after JSON. Start with {, end with }.\n\n"
            "Labels:\n"
            "- entails: premise logically implies hypothesis\n"
            "- contradicts: premise contradicts hypothesis\n"
            "- neutral: no clear entailment or contradiction"
        )

        user = (
            f"Premise: \"{premise}\"\n\n"
            f"Hypothesis: \"{hypothesis}\"\n\n"
            "Determine the NLI relationship. Examples:\n"
            "- \"All cats are mammals. Garfield is a cat.\" → {\"label\": \"entails\", \"score\": 0.95, \"rationale\": \"Cats are mammals\"}\n"
            "- \"The meeting is today. The meeting is tomorrow.\" → {\"label\": \"contradicts\", \"score\": 0.90, \"rationale\": \"Today vs tomorrow conflict\"}\n"
            "- \"Paris is in France. London is a city.\" → {\"label\": \"neutral\", \"score\": 0.60, \"rationale\": \"Unrelated statements\"}\n\n"
            "Output ONLY JSON, nothing else."
        )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def batch_score(self, targets: List[str], passages: List[str]) -> List[Dict[str, float]]:
        """Score target-passage pairs for NLI using OpenAI."""
        if not passages:
            return [{"entails": 0.0, "contradicts": 0.0, "neutral": 0.0} for _ in passages]

        results = []

        for target, passage in zip(targets, passages):
            # Build NLI-specific messages
            messages = self._build_nli_messages(target, passage)

            # Get raw OpenAI response
            raw_response = self._openai_interface.infer(messages)

            # Parse NLI response
            nli_scores = self._parse_nli_response(raw_response)
            results.append(nli_scores)

        return results

    def _parse_nli_response(self, raw_response: str) -> Dict[str, float]:
        """Parse raw OpenAI response into NLI scores."""
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
