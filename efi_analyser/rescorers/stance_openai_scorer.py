"""
Stance scorer using OpenAI GPT models.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from ..scorers.stance_scorer import StanceScorer
from .openai_interface import OpenAIInterface, OpenAIConfig


class StanceOpenAIConfig(OpenAIConfig):
    """Configuration for Stance OpenAI scorer."""
    pass


class StanceOpenAIScorer(StanceScorer):
    """Stance scorer using OpenAI GPT models with stance-specific prompts."""

    def __init__(self, name: str = "stance_openai", config: Optional[StanceOpenAIConfig] = None, scorer_backend=None):
        """
        Initialize Stance OpenAI scorer.

        Args:
            name: Scorer name
            config: Configuration
            scorer_backend: OpenAI interface backend to use
                           If None, uses default OpenAIInterface
        """
        super().__init__(name, config)
        self.config = config or StanceOpenAIConfig()

        # Use provided backend or create default
        if scorer_backend is not None:
            self._openai_interface = scorer_backend
        else:
            self._openai_interface = OpenAIInterface(name + "_base", config=self.config)

    def _build_stance_messages(self, target: str, text: str) -> List[Dict[str, str]]:
        """Build stance-specific messages for OpenAI."""
        system = (
            "Determine the stance of the text toward the target. "
            "Output ONLY JSON: {\"label\": \"pro|anti|neutral|uncertain\", \"score\": <0-1>, \"rationale\": \"<30 words>\"}.\n\n"
            "RULES: NO text before/after JSON. Start with {, end with }.\n\n"
            "Labels:\n"
            "- pro: text expresses support for the target\n"
            "- anti: text expresses opposition to the target\n"
            "- neutral: text is balanced or objective\n"
            "- uncertain: stance is unclear or ambiguous"
        )

        user = (
            f"Target: \"{target}\"\n\n"
            f"Text: \"{text}\"\n\n"
            "Determine the stance. Examples:\n"
            "- Target: \"Climate change is real\", Text: \"We must act now to combat climate change\" → {\"label\": \"pro\", \"score\": 0.95, \"rationale\": \"Strong support for action\"}\n"
            "- Target: \"Nuclear energy\", Text: \"Nuclear power is dangerous and should be banned\" → {\"label\": \"anti\", \"score\": 0.90, \"rationale\": \"Expresses opposition to nuclear energy\"}\n"
            "- Target: \"Electric cars\", Text: \"Electric vehicles have both advantages and disadvantages\" → {\"label\": \"neutral\", \"score\": 0.70, \"rationale\": \"Balanced view of pros and cons\"}\n\n"
            "Output ONLY JSON, nothing else."
        )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def batch_score(self, targets: List[str], passages: List[str]) -> List[Dict[str, float]]:
        """Score target-passage pairs for stance using OpenAI."""
        if not passages:
            return [{"pro": 0.0, "anti": 0.0, "neutral": 0.0, "uncertain": 0.0} for _ in passages]

        results = []

        for target, passage in zip(targets, passages):
            # Build stance-specific messages
            messages = self._build_stance_messages(target, passage)

            # Get raw OpenAI response
            raw_response = self._openai_interface.infer(messages)

            # Parse stance response
            stance_scores = self._parse_stance_response(raw_response)
            results.append(stance_scores)

        return results

    def _parse_stance_response(self, raw_response: str) -> Dict[str, float]:
        """Parse raw OpenAI response into stance scores."""
        stance_label = None  # No default fallback label
        confidence = 0.0  # No default confidence

        try:
            import json
            parsed = json.loads(raw_response)
            stance_label = parsed.get("label")
            if stance_label:
                confidence = parsed.get("score", parsed.get("confidence", 0.0))  # Support both "score" and "confidence" keys
                # Ensure confidence is between 0 and 1
                confidence = max(0.0, min(1.0, confidence))
        except:
            # If JSON parsing fails, try to extract from text
            text_lower = raw_response.lower()
            if "pro" in text_lower and "anti" not in text_lower:
                stance_label = "pro"
                confidence = 0.1  # Minimal confidence for text extraction
            elif "anti" in text_lower:
                stance_label = "anti"
                confidence = 0.1  # Minimal confidence for text extraction
            elif "uncertain" in text_lower:
                stance_label = "uncertain"
                confidence = 0.1  # Minimal confidence for text extraction

        # Create stance-style output
        stance_scores = {"pro": 0.0, "anti": 0.0, "neutral": 0.0, "uncertain": 0.0}
        if stance_label and stance_label in stance_scores:
            stance_scores[stance_label] = confidence

        return stance_scores
