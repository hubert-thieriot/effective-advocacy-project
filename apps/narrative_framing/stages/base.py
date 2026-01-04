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


@dataclass
class WorkflowState:
    """Shared state across pipeline stages"""
    schema: Optional[Any] = None  # FrameSchema
    assignments: Optional[Any] = None  # FrameAssignments
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
