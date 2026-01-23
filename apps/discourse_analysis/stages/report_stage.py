"""
Report generation stage for discourse analysis pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from efi_core.pipeline import PipelineStage

from apps.discourse_analysis.analysis.types import DiscourseAggregates
from apps.discourse_analysis.report.builder import ReportBuilder

from .base import StageContext


@dataclass
class ReportOutput:
    report_path: Path
    plots_dir: Path


class ReportStage(PipelineStage[StageContext, ReportOutput]):
    """Generate the discourse analysis HTML report."""

    def __init__(self, name: str, output_dir: Path):
        super().__init__(name, output_dir)
        self._report_path: Optional[Path] = None

    def should_run(self, input_data: Optional[StageContext]) -> bool:
        if input_data is None:
            return False
        aggregates_path = input_data.paths.aggregates_dir / "aggregates.json"
        if not aggregates_path.exists():
            self.logger.warning("Aggregates not found; skipping report.")
            return False
        self._report_path = input_data.paths.html
        return True

    def execute(self, input_data: StageContext) -> ReportOutput:
        builder = ReportBuilder(
            state=input_data.state,
            config=input_data.config,
            paths=input_data.paths,
        )
        report_path = builder.build()
        plots_dir = input_data.paths.plots_dir
        return ReportOutput(report_path=report_path, plots_dir=plots_dir)

    def load_cached(self) -> ReportOutput:
        if self._report_path and self._report_path.exists():
            return ReportOutput(report_path=self._report_path, plots_dir=self._report_path.parent / "plots")
        raise NotImplementedError("No cached report available.")

    def get_metadata(self) -> dict:
        return {"stage": self.name, "report_path": str(self._report_path) if self._report_path else None}
