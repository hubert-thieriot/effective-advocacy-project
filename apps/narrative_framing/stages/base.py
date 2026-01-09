"""
Base classes and types for narrative framing pipeline stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from apps.narrative_framing.config_models import NarrativeFramingConfig


@dataclass
class WorkflowPaths:
    """Paths used throughout the workflow"""
    results_dir: Path
    induction_dir: Optional[Path] = None
    annotation_dir: Optional[Path] = None
    training_dir: Optional[Path] = None
    classifier_dir: Optional[Path] = None
    classifications_dir: Optional[Path] = None
    aggregates_dir: Optional[Path] = None
    report_dir: Optional[Path] = None
    plots_dir: Optional[Path] = None
    schema_path: Optional[Path] = None
    assignments_path: Optional[Path] = None
    classifications_path: Optional[Path] = None
    # Legacy-style aliases for report builder compatibility
    assignments: Optional[Path] = None
    html: Optional[Path] = None


@dataclass
class WorkflowState:
    """Shared state across pipeline stages"""
    schema: Optional[Any] = None  # FrameSchema
    induction_samples: List[Any] = field(default_factory=list)  # List[Tuple[str, str]]
    assignments: Optional[Any] = None  # FrameAssignments
    classifier_predictions: List[Dict[str, Any]] = field(default_factory=list)  # Classifier predictions
    classifier_model: Optional[Any] = None  # FrameClassifierModel
    classifications: Optional[Any] = None  # DocumentClassifications
    aggregates: Optional[Any] = None  # Aggregates
    corpora_map: Optional[Dict[str, Any]] = None  # Dict[str, EmbeddedCorpus]
    sampler: Optional[Any] = None  # EmbeddedCorporaSampler
    document_metadata: Optional[List[Dict[str, Any]]] = None


@dataclass
class StageContext:
    """Context passed to each pipeline stage"""
    config: NarrativeFramingConfig
    paths: WorkflowPaths
    state: WorkflowState
    corpus_names: List[str]

    # Per-stage control flags
    allow_new_work: bool = True

    # Additional context from workflow (set by pipeline)
    filter: Optional[Any] = None  # Filter object
    filter_spec: Optional[Any] = None  # FilterSpec
    ind_sys_t: str = ""  # Induction system template
    ind_usr_t: str = ""  # Induction user template
    app_sys_t: str = ""  # Application system template
    app_usr_t: str = ""  # Application user template
