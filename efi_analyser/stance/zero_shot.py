"""Zero-shot stance detection using Hugging Face pipelines."""

from __future__ import annotations

import logging
from typing import Sequence, Tuple

from transformers import pipeline

from efi_analyser.stance.detector import StanceDetector
from efi_analyser.stance.types import STANCE_LABELS, StanceAssignments, StanceResult


class ZeroShotStanceDetector(StanceDetector):
    """Zero-shot stance detector using a transformers pipeline."""

    def __init__(self, config: object, paths: object) -> None:
        super().__init__(config, paths)
        model_name = getattr(self.config, "zero_shot_model", "facebook/bart-large-mnli")
        self._pipeline = pipeline("zero-shot-classification", model=model_name)
        self._labels = list(STANCE_LABELS)
        self._batch_size = self._resolve_batch_size()

    def _resolve_batch_size(self) -> int:
        explicit = getattr(self.config, "zero_shot_batch_size", None)
        if isinstance(explicit, int) and explicit > 0:
            return explicit
        classifier_cfg = getattr(self.config, "classifier", None)
        if classifier_cfg is not None:
            batch = getattr(classifier_cfg, "batch_size", None)
            if isinstance(batch, int) and batch > 0:
                return batch
        annot_cfg = getattr(self.config, "annotation", None)
        if annot_cfg is not None:
            batch = getattr(annot_cfg, "batch_size", None)
            if isinstance(batch, int) and batch > 0:
                return batch
        return 8

    def detect(self, chunks: Sequence[Tuple[str, str]]) -> StanceAssignments:
        results = StanceAssignments()
        targets = list(getattr(self.config, "targets", []) or [])
        if not targets:
            return results
        if not chunks:
            return results

        chunk_ids = [str(chunk_id) for chunk_id, _ in chunks]
        texts = [text for _, text in chunks]
        total_chunks = len(texts)
        logger = logging.getLogger(self.__class__.__name__)

        for target in targets:
            logger.info(
                "Zero-shot stance: target '%s' over %d chunks (batch_size=%d)",
                target,
                total_chunks,
                self._batch_size,
            )
            batch_indices = range(0, total_chunks, self._batch_size)
            try:
                from tqdm import tqdm

                total_batches = (total_chunks + self._batch_size - 1) // self._batch_size
                batch_indices = tqdm(
                    batch_indices,
                    total=total_batches,
                    desc=f"Stance target: {target}",
                    unit="batch",
                    leave=False,
                )
            except Exception:
                pass

            for start in batch_indices:
                batch_texts = texts[start : start + self._batch_size]
                batch_ids = chunk_ids[start : start + self._batch_size]
                outputs = self._pipeline(
                    batch_texts,
                    self._labels,
                    hypothesis_template=f"This text {{}} {target}.",
                    batch_size=self._batch_size,
                )
                for chunk_id, text, response in zip(batch_ids, batch_texts, outputs):
                    scores = {label: float(score) for label, score in zip(response["labels"], response["scores"])}
                    label = max(scores, key=scores.get) if scores else "neutral"
                    results.append(
                        StanceResult(
                            chunk_id=chunk_id,
                            target=str(target),
                            text=text,
                            scores=scores,
                            label=label,
                        )
                    )
        return results


__all__ = ["ZeroShotStanceDetector"]
