"""
NLI scorer using HuggingFace models.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel

from .nli_scorer import NLIScorer


class NLIHFScorerConfig(BaseModel):
    """Configuration for :class:`NLIHFScorer`.

    Attributes:
        model_name: HuggingFace model name to use for NLI.
        batch_size: Number of pairs to process per batch.
        device: Device id for transformers pipeline (``-1`` for CPU).
        entailment_label: Label representing entailment in model output.
        contradiction_label: Label representing contradiction in model output.
        neutral_label: Label representing neutral in model output.
    """

    model_name: str = "typeform/distilbert-base-uncased-mnli"
    batch_size: int = 8
    device: int = -1
    max_length: int = 384
    local_files_only: bool = False
    entailment_label: str = "ENTAILMENT"  # For typeform model: ENTAILMENT = entailment
    contradiction_label: str = "CONTRADICTION"  # For typeform model: CONTRADICTION = contradiction
    neutral_label: str = "NEUTRAL"  # For typeform model: NEUTRAL = neutral


class NLIHFScorer(NLIScorer):
    """NLI-based scorer for evaluating target-passage entailment relationships using HuggingFace models.

    The model estimates how strongly each passage entails the target text.
    Provides entailment, contradiction, and neutral probabilities.
    """

    def __init__(self, name: str = "nli_hf", config: Optional[NLIHFScorerConfig] = None) -> None:
        super().__init__(name, config)
        self.config = config or NLIHFScorerConfig()
        self._pipeline = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the transformers pipeline for NLI."""
        try:
            import os
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
            try:
                import torch
                if self.config.device == -1:
                    device = 0 if torch.cuda.is_available() else -1
                else:
                    device = self.config.device
            except Exception:
                device = self.config.device
            import time
            t0 = time.perf_counter()
            print(f"[NLI-HF] Loading model {self.config.model_name} (device={device}, local_only={self.config.local_files_only})...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                local_files_only=self.config.local_files_only,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_name,
                local_files_only=self.config.local_files_only,
            )
            self._pipeline = pipeline(
                task="text-classification",
                model=model,
                tokenizer=tokenizer,
                device=device,
                framework="pt",
            )

            print(".2f")
        except Exception as exc:  # pragma: no cover - graceful degradation
            print(
                f"⚠️ Warning: Failed to load NLI model {self.config.model_name}: {exc}."
                " Re-scoring disabled."
            )
            self._pipeline = None

    def batch_score(self, targets: List[str], passages: List[str]) -> List[Dict[str, float]]:
        """Score target-passage pairs using NLI model.

        Args:
            targets: List of target texts (e.g., queries/findings)
            passages: List of passage texts to score against targets

        Returns:
            List of dictionaries with label probabilities for each pair
        """
        if not passages or self._pipeline is None:
            return [{"entails": 0.0, "contradicts": 0.0, "neutral": 0.0} for _ in passages]

        # Create inputs for the model
        inputs = []
        valid_indices = []
        for i, (target, passage) in enumerate(zip(targets, passages)):
            if passage and target:
                inputs.append({"text": passage, "text_pair": target})
                valid_indices.append(i)

        if not inputs:
            return [{"entails": 0.0, "contradicts": 0.0, "neutral": 0.0} for _ in passages]

        # Set thread guards before inference to prevent stalls
        self._set_thread_guards()

        results = self._pipeline(
            inputs,
            batch_size=self.config.batch_size,
            truncation=True,
            max_length=self.config.max_length,
            top_k=None,
        )

        # Convert results to standardized format
        scores_list = []
        result_idx = 0

        for i in range(len(passages)):
            if i in valid_indices and result_idx < len(results):
                model_scores = results[result_idx]
                entail_score = 0.0
                contradiction_score = 0.0
                neutral_score = 0.0

                # Extract scores for each label
                for item in model_scores:
                    label = item["label"]
                    score = float(item["score"])
                    if label == self.config.entailment_label:
                        entail_score = score
                    elif label == self.config.contradiction_label:
                        contradiction_score = score
                    elif label == self.config.neutral_label:
                        neutral_score = score

                scores_list.append({
                    "entails": entail_score,
                    "contradicts": contradiction_score,
                    "neutral": neutral_score
                })
                result_idx += 1
            else:
                scores_list.append({"entails": 0.0, "contradicts": 0.0, "neutral": 0.0})

        return scores_list

    def _set_thread_guards(self) -> None:
        """Set thread guards to prevent stalls during inference."""
        import os
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

        import torch
        torch.set_num_threads(1)