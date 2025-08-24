"""NLI-based re-scorer for improving retrieval quality."""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel

from efi_core.protocols import ReScorer
from efi_core.retrieval.retriever import SearchResult


class NLIReScorerConfig(BaseModel):
    """Configuration for :class:`NLIReScorer`.

    Attributes:
        model_name: HuggingFace model name to use for NLI.
        batch_size: Number of pairs to process per batch.
        device: Device id for transformers pipeline (``-1`` for CPU).
        entailment_label: Label representing entailment in model output.
    """

    model_name: str = "facebook/bart-large-mnli"
    batch_size: int = 8
    device: int = -1
    entailment_label: str = "ENTAILMENT"


class NLIReScorer(ReScorer[SearchResult]):
    """Re-score search results using a Natural Language Inference model.

    The model estimates how strongly each document chunk entails the query
    (finding) text. Entailment probabilities are used as the new scores.
    """

    def __init__(self, config: Optional[NLIReScorerConfig] = None) -> None:
        self.config = config or NLIReScorerConfig()
        self._pipeline = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the transformers pipeline for NLI."""
        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                "text-classification",
                model=self.config.model_name,
                device=self.config.device,
            )
        except Exception as exc:  # pragma: no cover - graceful degradation
            print(
                f"⚠️ Warning: Failed to load NLI model {self.config.model_name}: {exc}."
                " Re-scoring disabled."
            )
            self._pipeline = None

    def rescore(self, query: str, matches: List[SearchResult]) -> List[SearchResult]:
        """Re-score retrieved matches using NLI entailment scores.

        Args:
            query: The finding text.
            matches: Retrieved matches containing chunk text in ``metadata['text']``.

        Returns:
            The list of matches sorted by entailment probability.
        """
        if not matches or self._pipeline is None:
            return matches

        inputs = []
        for match in matches:
            chunk_text = match.metadata.get("text", "")
            inputs.append({"text": chunk_text, "text_pair": query})

        results = self._pipeline(
            inputs,
            batch_size=self.config.batch_size,
            truncation=True,
            return_all_scores=True,
        )

        rescored: List[SearchResult] = []
        for match, scores in zip(matches, results):
            entail_score = 0.0
            for item in scores:
                if item["label"].upper().startswith(self.config.entailment_label):
                    entail_score = float(item["score"])
                    break
            rescored.append(
                SearchResult(
                    item_id=match.item_id,
                    score=entail_score,
                    metadata={**match.metadata, "nli_score": entail_score},
                )
            )

        rescored.sort(key=lambda x: x.score, reverse=True)
        return rescored
