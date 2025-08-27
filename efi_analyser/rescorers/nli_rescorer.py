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

    model_name: str = "typeform/distilbert-base-uncased-mnli"
    batch_size: int = 8
    device: int = -1
    max_length: int = 384
    local_files_only: bool = False
    entailment_label: str = "ENTAILMENT"  # For typeform model: ENTAILMENT = entailment


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
            print(f"[NLI] Loading model {self.config.model_name} (device={device}, local_only={self.config.local_files_only})...")
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
            
            print(f"[NLI] Model loaded in {time.perf_counter() - t0:.2f}s")
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
            # Skip empty text - return 0 score
            if not chunk_text or not query:
                continue
            # For MNLI models: use dictionary format with premise and hypothesis
            inputs.append({"text": chunk_text, "text_pair": query})

        import os
        
        # Set thread guards before inference to prevent stalls
        self._set_thread_guards()
        
        results = self._pipeline(
            inputs,
            batch_size=self.config.batch_size,
            truncation=True,
            max_length=self.config.max_length,
            top_k=None,
        )

        rescored: List[SearchResult] = []
        result_index = 0
        
        for match in matches:
            chunk_text = match.metadata.get("text", "")
            
            # Handle empty text - return 0 score
            if not chunk_text or not query:
                rescored.append(
                    SearchResult(
                        item_id=match.item_id,
                        score=0.0,
                        metadata={**match.metadata, "nli_score": 0.0},
                    )
                )
                continue
            
            # Get scores for this match
            if result_index < len(results):
                scores = results[result_index]
                entail_score = float('nan')
                
                # Find entailment score using configured label
                for item in scores:
                    if item["label"] == self.config.entailment_label:
                        entail_score = float(item["score"])
                        break
                
                rescored.append(
                    SearchResult(
                        item_id=match.item_id,
                        score=entail_score,
                        metadata={**match.metadata, "nli_score": entail_score},
                    )
                )
                result_index += 1
            else:
                # Fallback if results don't match
                rescored.append(
                    SearchResult(
                        item_id=match.item_id,
                        score=0.0,
                        metadata={**match.metadata, "nli_score": 0.0},
                    )
                )

        rescored.sort(key=lambda x: x.score, reverse=True)
        return rescored
    
    def _set_thread_guards(self) -> None:
        """Set thread guards to prevent stalls during inference."""
        import os
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        
        import torch
        torch.set_num_threads(1)
