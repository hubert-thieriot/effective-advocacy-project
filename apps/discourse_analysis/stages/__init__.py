"""Pipeline stages for discourse analysis."""

from .base import StageContext, WorkflowPaths, WorkflowState
from .corpus_loading_stage import CorpusLoadingStage
from .framing_stage import FramingStage
from .stance_detection_stage import StanceDetectionStage
from .analysis_stage import AnalysisStage
from .report_stage import ReportStage

__all__ = [
    "StageContext",
    "WorkflowPaths",
    "WorkflowState",
    "CorpusLoadingStage",
    "FramingStage",
    "StanceDetectionStage",
    "AnalysisStage",
    "ReportStage",
]
