"""Pipeline stages for discourse analysis."""

from .base import StageContext, WorkflowPaths, WorkflowState
from .corpus_loading_stage import CorpusLoadingStage
from .framing_stage import FramingStage
from .ner_stage import NERStage
from .claims_analysis_stage import ClaimsAnalysisStage
from .dna_stage import DNAStage
from .analysis_stage import AnalysisStage
from .report_stage import ReportStage

__all__ = [
    "StageContext",
    "WorkflowPaths",
    "WorkflowState",
    "CorpusLoadingStage",
    "FramingStage",
    "NERStage",
    "ClaimsAnalysisStage",
    "DNAStage",
    "AnalysisStage",
    "ReportStage",
]
