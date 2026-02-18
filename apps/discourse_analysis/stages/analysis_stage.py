"""
Analysis stage for discourse analysis pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from efi_core.pipeline import PipelineStage

from apps.discourse_analysis.analysis.aggregator import build_aggregates, DocumentFilter
from apps.discourse_analysis.analysis.types import DiscourseAggregates

from .base import StageContext


class AnalysisStage(PipelineStage[StageContext, DiscourseAggregates]):
    """Aggregate frame + stance signals for reporting."""

    def __init__(self, name: str, output_dir: Path):
        super().__init__(name, output_dir)
        self._aggregates_dir: Optional[Path] = None

    def should_run(self, input_data: Optional[StageContext]) -> bool:
        if input_data is None:
            return False
        self._aggregates_dir = input_data.paths.aggregates_dir
        return True

    def execute(self, input_data: StageContext) -> DiscourseAggregates:
        state = input_data.state

        if not state.frame_classifications:
            self.logger.info("No frame classifications available; producing empty aggregates.")
            aggregates = DiscourseAggregates()
        else:
            # Build document filter from config
            doc_filter = DocumentFilter.from_config(input_data.config.filter)

            aggregates = build_aggregates(
                state.frame_classifications,
                stance_results=None,  # Stance has been removed
                doc_filter=doc_filter,
            )

        self._aggregates_dir.mkdir(parents=True, exist_ok=True)
        aggregates.save(self._aggregates_dir)
        return aggregates

    def load_cached(self) -> DiscourseAggregates:
        if self._aggregates_dir and self._aggregates_dir.exists():
            return DiscourseAggregates.load(self._aggregates_dir)
        raise NotImplementedError("No cached aggregates available.")

    def get_metadata(self) -> dict:
        return {"stage": self.name}
