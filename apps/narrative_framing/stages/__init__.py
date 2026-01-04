"""
Pipeline stages for narrative framing workflow.

This module contains individual pipeline stages that can be composed into
the full narrative framing workflow using the efi_core.pipeline framework.

Each stage:
- Inherits from PipelineStage[InputType, OutputType]
- Implements should_run() for caching logic
- Implements execute() for the actual work
- Optionally implements load_cached() and save_result()

This modular approach makes each stage:
- Independently testable
- Reusable in different contexts
- Easy to understand (single responsibility)
- Simple to modify or extend

Example usage:
    from efi_core.pipeline import Pipeline
    from .report_stage import ReportStage, ReportInput

    # Create stage
    stage = ReportStage("report", output_dir)

    # Prepare input
    input_data = ReportInput(...)

    # Run stage
    result = stage.run(input_data)

    if result.success:
        print(f"Report: {result.data.report_path}")
    else:
        print(f"Error: {result.error}")
"""

from .base import StageContext, WorkflowPaths, WorkflowState
from .corpus_loading_stage import CorpusLoadingStage
from .induction_stage import InductionStage
from .annotation_stage import AnnotationStage
from .training_stage import TrainingStage
from .classification_stage import ClassificationStage
from .aggregation_stage import AggregationStage
from .report_stage import ReportStage, ReportInput, ReportOutput

__all__ = [
    # Base classes
    "StageContext",
    "WorkflowPaths",
    "WorkflowState",
    # Stages
    "CorpusLoadingStage",
    "InductionStage",
    "AnnotationStage",
    "TrainingStage",
    "ClassificationStage",
    "AggregationStage",
    "ReportStage",
    "ReportInput",
    "ReportOutput",
]
