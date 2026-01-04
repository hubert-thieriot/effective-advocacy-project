"""
Report generation stage for narrative framing pipeline.

This is a proof-of-concept demonstrating the Pipeline pattern.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from efi_core.pipeline import PipelineStage, StageResult

from apps.narrative_framing.aggregates import Aggregates
from apps.narrative_framing.report import write_html_report


@dataclass
class ReportInput:
    """Input data for report generation stage"""
    aggregates_dir: Path
    schema_path: Path
    classifications_path: Path
    corpus_names: list[str]
    output_dir: Path
    config: any  # NarrativeFramingConfig


@dataclass
class ReportOutput:
    """Output from report generation stage"""
    report_path: Path
    plots_dir: Path


class ReportStage(PipelineStage[ReportInput, ReportOutput]):
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

    def should_run(self, input_data: Optional[ReportInput]) -> bool:
        """Check if report should be generated.

        Report runs if:
        - Input data is provided
        - Aggregates exist
        - Either report doesn't exist OR we're in regenerate mode
        """
        if input_data is None:
            self.logger.warning("No input data for report stage")
            return False

        # Check if aggregates exist
        aggregates_dir = input_data.aggregates_dir
        if not aggregates_dir.exists():
            self.logger.warning(f"Aggregates directory does not exist: {aggregates_dir}")
            return False

        # Load aggregates to verify they exist
        aggregates = Aggregates.load(aggregates_dir)
        if aggregates is None:
            self.logger.warning("No aggregates found - cannot generate report")
            return False

        # Check if report already exists
        report_path = input_data.output_dir / "report.html"
        self._report_path = report_path

        # Always run if regenerate_report_only is True
        if input_data.config.regenerate_report_only:
            self.logger.info("Regenerate mode - will generate report")
            return True

        # Run if report doesn't exist
        if not report_path.exists():
            self.logger.info("Report does not exist - will generate")
            return True

        # Skip if report exists and not in regenerate mode
        self.logger.info("Report exists and not in regenerate mode - skipping")
        return False

    def execute(self, input_data: ReportInput) -> ReportOutput:
        """Generate the HTML report.

        Args:
            input_data: Report generation parameters

        Returns:
            ReportOutput with paths to generated files
        """
        self.logger.info("Generating HTML report...")

        # Load aggregates
        aggregates = Aggregates.load(input_data.aggregates_dir)
        if aggregates is None:
            raise ValueError("Aggregates not found")

        # Generate report using existing write_html_report function
        report_path = write_html_report(
            aggregates_dir=input_data.aggregates_dir,
            schema_path=input_data.schema_path,
            classifications_path=input_data.classifications_path,
            corpus_names=input_data.corpus_names,
            output_dir=input_data.output_dir,
            config=input_data.config,
            document_metadata=None,  # TODO: pass from workflow state
        )

        plots_dir = input_data.output_dir / "plots"

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
