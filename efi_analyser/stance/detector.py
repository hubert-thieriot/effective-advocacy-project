"""Stance detector interface and factory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence, Tuple

from efi_analyser.stance.types import StanceAssignments


class StanceDetector(ABC):
    """Abstract base for stance detectors."""

    def __init__(self, config: object, paths: object) -> None:
        self.config = config
        self.paths = paths

    @abstractmethod
    def detect(self, chunks: Sequence[Tuple[str, str]]) -> StanceAssignments:
        """Detect stance for all chunks."""
        raise NotImplementedError

    @classmethod
    def create(cls, config: object, paths: object) -> "StanceDetector":
        """Factory for stance detector implementations."""
        mode = getattr(config, "mode", "zero_shot")
        if mode == "zero_shot":
            from efi_analyser.stance.zero_shot import ZeroShotStanceDetector

            return ZeroShotStanceDetector(config, paths)
        from efi_analyser.stance.trained import TrainedStanceDetector

        return TrainedStanceDetector(config, paths)


__all__ = ["StanceDetector"]
