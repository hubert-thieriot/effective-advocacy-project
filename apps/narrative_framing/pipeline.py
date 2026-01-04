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
from .filtering import Filter
from .stages import (
    CorpusLoadingStage,
    InductionStage,
    AnnotationStage,
    TrainingStage,
    ClassificationStage,
    AggregationStage,
    ReportStage,
    StageContext,
    WorkflowPaths,
    WorkflowState,
)


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

        # Initialize filter
        self.filter = Filter(
            exclude_regex=config.filter.chunk.exclude_regex,
            exclude_min_hits=config.filter.chunk.exclude_min_hits,
            trim_after_markers=config.filter.chunk.trim_after_markers,
            keywords=config.filter.chunk.keywords,
        )

        # Load prompt templates
        self._load_prompts()

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

    def _load_prompts(self) -> None:
        """Load prompt templates from files."""
        # Import here to avoid circular dependency
        from apps.narrative_framing.run import (
            _resolve_default_prompt_paths,
            _read_text_or_fail,
        )

        try:
            prompt_paths = _resolve_default_prompt_paths()
            self.ind_sys_t = _read_text_or_fail(prompt_paths["induction_system"])
            self.ind_usr_t = _read_text_or_fail(prompt_paths["induction_user"])
            self.app_sys_t = _read_text_or_fail(prompt_paths["application_system"])
            self.app_usr_t = _read_text_or_fail(prompt_paths["application_user"])
        except Exception as e:
            self.logger.warning(f"Failed to load prompt templates: {e}")
            # Use fallback templates
            self.ind_sys_t = "{system_message}"
            self.ind_usr_t = "{user_message}"
            self.app_sys_t = "{system_message}"
            self.app_usr_t = "{user_message}"

    def build_stages(self) -> List[PipelineStage]:
        """Build the pipeline stages.

        Returns:
            Ordered list of stages to execute
        """
        # Create stage context with all necessary data
        context = StageContext(
            config=self.config,
            paths=self.paths,
            state=self.state,
            corpus_names=self.corpus_names,
            allow_new_work=not self.config.regenerate_report_only,
            filter=self.filter,
            filter_spec=self.filter.to_spec(),
            ind_sys_t=self.ind_sys_t,
            ind_usr_t=self.ind_usr_t,
            app_sys_t=self.app_sys_t,
            app_usr_t=self.app_usr_t,
        )

        # Build complete pipeline
        # Note: Each stage receives context but modifies shared state
        stages = [
            CorpusLoadingStage("corpus_loading", self.output_dir),
            InductionStage("induction", self.output_dir),
            AnnotationStage("annotation", self.output_dir),
            TrainingStage("training", self.output_dir),
            ClassificationStage("classification", self.output_dir),
            AggregationStage("aggregation", self.output_dir),
            # ReportStage("report", self.output_dir),  # TODO: Wire up report stage
        ]

        # Store context for stages to access
        self._context = context

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

        # Build stages (creates context)
        self.stages = self.build_stages()
        results = {}

        # Run each stage with context
        for stage in self.stages:
            self.logger.info(f"Pipeline stage: {stage.name}")
            result = stage.run(self._context)
            results[stage.name] = result

            if result.error:
                self.logger.error(f"Pipeline stopped due to error in {stage.name}")
                break

            # Update state with stage output if applicable
            if result.data is not None:
                self._update_state_from_stage(stage.name, result.data)

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

    def _update_state_from_stage(self, stage_name: str, data):
        """Update workflow state with stage output."""
        if stage_name == "corpus_loading":
            self.state.corpora_map = data
        elif stage_name == "induction":
            self.state.schema = data
        elif stage_name == "annotation":
            self.state.assignments = data
        elif stage_name == "classification":
            self.state.classifications = data
        elif stage_name == "aggregation":
            self.state.aggregates = data
