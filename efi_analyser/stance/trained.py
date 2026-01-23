"""Trained stance detector implementation."""

from __future__ import annotations

from typing import Sequence, Tuple, List

from efi_analyser.stance.detector import StanceDetector
from efi_analyser.stance.trainer import StanceClassifierModel
from efi_analyser.stance.types import STANCE_LABELS, StanceAssignments, StanceResult, build_stance_input


class TrainedStanceDetector(StanceDetector):
    """Stance detector backed by a trained classifier."""

    def __init__(self, config: object, paths: object) -> None:
        super().__init__(config, paths)
        self._model: StanceClassifierModel | None = None

    def load_model(self) -> StanceClassifierModel:
        if self._model is not None:
            return self._model
        classifier_dir = getattr(self.paths, "classifier_dir", None)
        if classifier_dir is None:
            raise ValueError("paths.classifier_dir is required for trained stance detection")
        self._model = StanceClassifierModel.load(classifier_dir)
        return self._model

    def detect(self, chunks: Sequence[Tuple[str, str]]) -> StanceAssignments:
        targets = list(getattr(self.config, "targets", []) or [])
        if not targets:
            return StanceAssignments()

        model = self.load_model()
        inputs: List[str] = []
        index: List[Tuple[str, str, str]] = []
        for chunk_id, text in chunks:
            for target in targets:
                inputs.append(build_stance_input(str(target), text))
                index.append((str(chunk_id), str(target), text))

        if not inputs:
            return StanceAssignments()

        probs = model.predict_proba_batch(inputs, batch_size=max(1, getattr(self.config.classifier, "batch_size", 8)))
        results = StanceAssignments()
        for (chunk_id, target, text), scores in zip(index, probs):
            label = max(scores, key=scores.get) if scores else STANCE_LABELS[-1]
            results.append(
                StanceResult(
                    chunk_id=chunk_id,
                    target=target,
                    text=text,
                    scores=scores,
                    label=label,
                )
            )
        return results


__all__ = ["TrainedStanceDetector"]
