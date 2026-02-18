"""
Base classes and types for discourse analysis pipeline stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from apps.discourse_analysis.config_models import DiscourseAnalysisConfig


@dataclass
class WorkflowPaths:
    """Paths used throughout the discourse analysis workflow."""

    results_dir: Path

    framing_dir: Path
    framing_schema_path: Path
    framing_assignments_path: Path
    framing_classifier_dir: Path
    framing_classifications_dir: Path

    ner_dir: Path
    ner_entities_path: Path
    ner_consolidated_path: Path

    # Claims analysis paths
    claims_dir: Path
    statements_path: Path
    claims_schema_path: Path
    agreements_path: Path

    # DNA analysis paths
    dna_dir: Path
    dna_network_path: Path
    dna_coalition_path: Path

    aggregates_dir: Path
    report_dir: Path
    plots_dir: Path
    html: Path


@dataclass
class WorkflowState:
    """Shared mutable state across pipeline stages."""

    corpora: Optional[Any] = None  # EmbeddedCorpora
    sampler: Optional[Any] = None  # EmbeddedCorporaSampler

    frame_schema: Optional[Any] = None  # FrameSchema
    frame_assignments: Optional[Any] = None  # FrameAssignments
    frame_classifications: Optional[Any] = None  # DocumentClassifications
    frame_annotation_candidates: List[Any] = field(default_factory=list)  # List[Candidate]

    ner_result: Optional[Any] = None  # NERResult (includes consolidated via ner_result.consolidated)

    claims_result: Optional[Any] = None  # ClaimsAnalysisResult

    dna_result: Optional[Any] = None        # DNANetworkResult
    coalition_result: Optional[Any] = None  # CoalitionAnalysisResult

    aggregates: Optional[Any] = None


@dataclass
class StageContext:
    """Context passed to each pipeline stage."""

    config: DiscourseAnalysisConfig
    paths: WorkflowPaths
    state: WorkflowState
    corpus_names: List[str]

    allow_new_work: bool = True

    filter: Optional[Any] = None
    filter_spec: Optional[Any] = None
    ind_sys_t: str = ""
    ind_usr_t: str = ""
    app_sys_t: str = ""
    app_usr_t: str = ""
