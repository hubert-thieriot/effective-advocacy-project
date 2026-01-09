"""
Report generation stage for narrative framing pipeline.

This is a proof-of-concept demonstrating the Pipeline pattern.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from efi_core.pipeline import PipelineStage, StageResult

from apps.narrative_framing.aggregates import Aggregates
from apps.narrative_framing.report import ReportBuilder
from apps.narrative_framing.run import load_schema
from efi_analyser.frames import FrameAssignments, FrameSchema
from efi_analyser.frames.classifier import DocumentClassifications, FrameClassifierArtifacts
from .base import StageContext


@dataclass
class ReportOutput:
    """Output from report generation stage"""
    report_path: Path
    plots_dir: Path


class ReportStage(PipelineStage[StageContext, ReportOutput]):
    """
    Pipeline stage for generating HTML reports.

    This stage:
    1. Checks if aggregates exist
    2. Loads schema and classifications
    3. Generates HTML report with visualizations
    4. Saves report to output directory

    Example:
        stage = ReportStage("report", output_dir)
        result = stage.run(report_input)
        if result.success:
            print(f"Report generated: {result.data.report_path}")
    """

    def __init__(self, name: str, output_dir: Path):
        super().__init__(name, output_dir)
        self._report_path: Optional[Path] = None

    def should_run(self, input_data: Optional[StageContext]) -> bool:
        """Check if report should be generated.

        Report runs if:
        - Input data is provided
        - Aggregates exist
        """
        if input_data is None:
            self.logger.warning("No input data for report stage")
            return False

        paths = input_data.paths
        config = input_data.config

        # Check if aggregates exist
        aggregates_dir = paths.aggregates_dir
        if not aggregates_dir or not aggregates_dir.exists():
            self.logger.warning(f"Aggregates directory does not exist: {aggregates_dir}")
            return False

        # Load aggregates to verify they exist
        aggregates = Aggregates.load(aggregates_dir)
        if aggregates is None:
            self.logger.warning("No aggregates found - cannot generate report")
            return False

        # Check if report already exists
        report_path = paths.html or (
            paths.report_dir / "frame_report.html" if paths.report_dir else self.output_dir / "frame_report.html"
        )
        self._report_path = report_path

        # Always generate the report when aggregates exist.
        if config.regenerate_report_only:
            self.logger.info("Regenerate mode - will generate report")
        else:
            self.logger.info("Generating report (always true when aggregates exist)")
        return True

    def execute(self, input_data: StageContext) -> ReportOutput:
        """Generate the HTML report.

        Args:
            input_data: Stage context

        Returns:
            ReportOutput with paths to generated files
        """
        self.logger.info("Generating HTML report...")

        paths = input_data.paths
        config = input_data.config
        state = input_data.state

        schema = state.schema
        if schema is None and paths.schema_path and paths.schema_path.exists():
            schema = load_schema(paths.schema_path)

        assignments = state.assignments or FrameAssignments()
        if not assignments and paths.assignments_path and paths.assignments_path.exists():
            try:
                assignments = FrameAssignments.load(paths.assignments_path)
            except Exception as exc:
                self.logger.warning(f"Failed to load cached assignments: {exc}")

        classifications = state.classifications or DocumentClassifications()
        if not classifications and paths.classifications_dir and paths.classifications_dir.exists():
            classifications = DocumentClassifications.from_folder(paths.classifications_dir)

        aggregates = state.aggregates
        if aggregates is None and paths.aggregates_dir:
            aggregates = Aggregates.load(paths.aggregates_dir)

        classifier_predictions: List[Dict[str, object]] = []
        if paths.results_dir:
            predictions_path = paths.results_dir / "frame_classifier_predictions.json"
            if predictions_path.exists():
                try:
                    artifacts = FrameClassifierArtifacts.load_predictions(predictions_path)
                    classifier_predictions = artifacts.predictions
                except Exception as exc:
                    self.logger.warning(f"Failed to load classifier predictions: {exc}")

        # Update state with loaded data if needed
        if schema and not state.schema:
            state.schema = schema
        if assignments and not state.assignments:
            state.assignments = assignments
        if classifications and not state.classifications:
            state.classifications = classifications
        if aggregates and not state.aggregates:
            state.aggregates = aggregates

        # Collect total doc IDs from classifications
        total_doc_ids = []
        if state.classifications:
            total_doc_ids = [doc.doc_id for doc in state.classifications if doc.doc_id]

        # Generate report using ReportBuilder with the actual state
        report_builder = ReportBuilder(
            state=state,
            config=config,
            paths=paths,
            total_doc_ids=total_doc_ids,
            corpora_map=state.corpora_map,
        )
        report_builder.build()

        # Determine report path
        report_path = paths.html or self.output_dir / "frame_report.html"
        plots_dir = paths.plots_dir or (report_path.parent / "plots")

        self.logger.info(f"Report generated: {report_path}")

        return ReportOutput(
            report_path=report_path,
            plots_dir=plots_dir
        )

    def load_cached(self) -> ReportOutput:
        """Load cached report if it exists."""
        if self._report_path and self._report_path.exists():
            plots_dir = self._report_path.parent / "plots"
            return ReportOutput(
                report_path=self._report_path,
                plots_dir=plots_dir
            )
        raise NotImplementedError("No cached report available")

    def get_metadata(self) -> dict:
        """Get metadata about report generation."""
        return {
            "stage": self.name,
            "report_path": str(self._report_path) if self._report_path else None,
        }
