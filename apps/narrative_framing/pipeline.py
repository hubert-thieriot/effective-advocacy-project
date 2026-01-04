"""
Narrative framing analysis pipeline.

This module defines the complete pipeline for narrative framing analysis,
composed of individual stages that can be tested and modified independently.

Example usage:
    from apps.narrative_framing import load_config
    from apps.narrative_framing.pipeline import NarrativeFramingPipeline

    config = load_config(Path("config.yaml"))
    pipeline = NarrativeFramingPipeline(config, output_dir=Path("results"))

    results = pipeline.run()

    for stage_name, result in results.items():
        if result.success:
            print(f"✓ {stage_name}")
        else:
            print(f"✗ {stage_name}: {result.error}")
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from efi_core.pipeline import Pipeline, PipelineStage

from .config_models import NarrativeFramingConfig
from .stages.aggregation_stage import AggregationStage
from .stages.base import StageContext, WorkflowPaths, WorkflowState


class NarrativeFramingPipeline(Pipeline):
    """
    Multi-stage pipeline for narrative framing analysis.

    Stages:
    1. Corpus Loading - Load and prepare corpora
    2. Induction - Induce frame schema from sample documents
    3. Annotation - Annotate documents with LLM
    4. Training - Train classifier (if enabled)
    5. Classification - Apply classifier or LLM to all documents
    6. Aggregation - Build temporal/domain aggregations
    7. Report - Generate HTML report and visualizations

    Each stage:
    - Has clear inputs and outputs
    - Can be skipped if cached results exist
    - Handles errors gracefully
    - Is independently testable
    """

    def __init__(self, config: NarrativeFramingConfig, output_dir: Path):
        super().__init__(config, output_dir)

        # Initialize workflow paths
        self.paths = self._init_paths(output_dir)

        # Initialize workflow state (shared across stages)
        self.state = WorkflowState()

        # Corpus names
        self.corpus_names = list(config.iter_corpus_names())

    def _init_paths(self, output_dir: Path) -> WorkflowPaths:
        """Initialize all workflow paths."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        return WorkflowPaths(
            results_dir=output_dir,
            induction_dir=output_dir / "induction",
            annotation_dir=output_dir / "annotation",
            training_dir=output_dir / "training",
            classifier_dir=output_dir / "classifier",
            classifications_dir=output_dir / "classifications",
            aggregates_dir=output_dir / "aggregates",
            report_dir=output_dir / "report",
            plots_dir=output_dir / "plots",
            schema_path=output_dir / "induction" / "schema.json",
            assignments_path=output_dir / "annotation" / "assignments.json",
            classifications_path=output_dir / "classifications" / "classifications.json",
        )

    def build_stages(self) -> List[PipelineStage]:
        """Build the pipeline stages.

        Returns:
            Ordered list of stages to execute
        """
        # Create stage context
        context = StageContext(
            config=self.config,
            paths=self.paths,
            state=self.state,
            corpus_names=self.corpus_names,
            allow_new_work=not self.config.regenerate_report_only,
        )

        # TODO: Add all stages once extracted
        # For now, just aggregation as a proof-of-concept
        stages = [
            # CorpusLoadingStage("corpus_loading", self.output_dir),
            # InductionStage("induction", self.output_dir),
            # AnnotationStage("annotation", self.output_dir),
            # TrainingStage("training", self.output_dir),
            # ClassificationStage("classification", self.output_dir),
            AggregationStage("aggregation", self.output_dir),
            # ReportStage("report", self.output_dir),
            # PlotStage("plots", self.output_dir),
        ]

        return stages

    def run(self):
        """Run the pipeline and return results.

        Returns:
            Dictionary mapping stage names to their results
        """
        self.logger.info("=" * 60)
        self.logger.info("NARRATIVE FRAMING PIPELINE")
        self.logger.info("=" * 60)
        self.logger.info(f"Config: {self.config.corpus}")
        self.logger.info(f"Output: {self.output_dir}")
        self.logger.info("=" * 60)

        results = super().run()

        # Print summary
        self.logger.info("=" * 60)
        self.logger.info("PIPELINE SUMMARY")
        self.logger.info("=" * 60)

        for stage_name, result in results.items():
            status = "✓" if result.success else "✗"
            msg = "success" if result.success else f"error: {result.error}"
            self.logger.info(f"{status} {stage_name}: {msg}")

        self.logger.info("=" * 60)

        return results
