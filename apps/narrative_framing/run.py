#!/usr/bin/env python3
"""Command-line entry point for the narrative framing workflow."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse

import os
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env if present (for WANDB_*, etc.)
load_dotenv()

# Set tokenizers parallelism to false to avoid warnings when processes fork
# This must be set before any tokenizers are loaded
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from apps.narrative_framing.aggregates import Aggregates
from apps.narrative_framing.config import NarrativeFramingConfig, load_config


from efi_analyser.chunkers.sentence_chunker import SentenceChunker
from efi_analyser.frames import (
    Frame,
    FrameAssignments,
    FrameSchema    
)
from efi_analyser.frames.classifier import (
    DocumentClassifications,
    FrameClassifierModel
)
from efi_analyser.scorers.openai_interface import OpenAIConfig, OpenAIInterface
from efi_corpus.embedded.embedded_corpus import EmbeddedCorpus
from efi_analyser.frames.plotting import PlotConfig, run_plots
from efi_core.utils import normalize_date

try:  # Prefer spaCy-based chunker when available.
    from efi_analyser.chunkers import TextChunker, TextChunkerConfig  # type: ignore
    _TEXT_CHUNKER_ERROR = None
except Exception as exc:  # pragma: no cover - informational only
    TextChunker = None  # type: ignore
    TextChunkerConfig = None  # type: ignore
    _TEXT_CHUNKER_ERROR = exc


@dataclass
class ResultPaths:
    schema: Optional[Path] = None
    assignments: Optional[Path] = None
    classifier_predictions: Optional[Path] = None
    classifier_dir: Optional[Path] = None
    # Aggregates folder with strategy-specific files
    aggregates_dir: Optional[Path] = None
    classifications_dir: Optional[Path] = None
    frame_timeseries: Optional[Path] = None
    html: Optional[Path] = None


@dataclass
class WorkflowState:
    """Mutable state threaded through the workflow stages."""

    schema: Optional[FrameSchema] = None
    induction_samples: List[Tuple[str, str]] = field(default_factory=list)

    # LLM application / annotation
    assignments: FrameAssignments = field(default_factory=FrameAssignments)
    induction_reused: bool = False
    assignments_reused: bool = False

    # Classifier
    classifier_predictions: List[Dict[str, object]] = field(default_factory=list)
    classifier_model: Optional[FrameClassifierModel] = None

    # Corpus-wide classification and aggregates
    classifications: DocumentClassifications = field(default_factory=DocumentClassifications)
    aggregates: Optional[Aggregates] = None


# --------------------------- Prompt helpers ---------------------------------
def _read_text_or_fail(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def _resolve_default_prompt_paths() -> Dict[str, Path]:
    base = Path("prompts")
    paths = {
        "induction_system": base / "induction" / "system.jinja",
        "induction_user": base / "induction" / "user.jinja",
        "application_system": base / "application" / "system.jinja",
        "application_user": base / "application" / "user.jinja",
    }
    for key, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing default prompt template '{key}': {p}")
    return paths


# def _save_resolved_messages(directory: Path, name_prefix: str, messages_list: List[List[Dict[str, str]]]) -> None:
#     directory.mkdir(parents=True, exist_ok=True)
#     for idx, messages in enumerate(messages_list, start=1):
#         for m in messages:
#             role = str(m.get("role", "unknown")).lower()
#             content = str(m.get("content", ""))
#             out = directory / f"{name_prefix}_{idx:03d}_{role}.txt"
#             out.write_text(content, encoding="utf-8")


def _copy_templates_to_results(paths_map: Dict[str, Path], out_dir: Path) -> None:
    dst_ind = out_dir / "prompts" / "induction" / "templates"
    dst_app = out_dir / "prompts" / "application" / "templates"
    dst_ann = out_dir / "prompts" / "frame_annotation" / "templates"
    for dst in (dst_ind, dst_app, dst_ann):
        dst.mkdir(parents=True, exist_ok=True)
    (dst_ind / "system.jinja").write_text(_read_text_or_fail(paths_map["induction_system"]), encoding="utf-8")
    (dst_ind / "user.jinja").write_text(_read_text_or_fail(paths_map["induction_user"]), encoding="utf-8")
    for dst in (dst_app, dst_ann):
        (dst / "system.jinja").write_text(_read_text_or_fail(paths_map["application_system"]), encoding="utf-8")
        (dst / "user.jinja").write_text(_read_text_or_fail(paths_map["application_user"]), encoding="utf-8")


def save_schema(path: Path, schema: FrameSchema) -> None:
    payload = {
        "domain": schema.domain,
        "notes": schema.notes,
        "frames": [
            {
                "frame_id": frame.frame_id,
                "short_name": frame.short_name,
                "name": frame.name,
                "description": frame.description,
                "keywords": frame.keywords,
                "examples": frame.examples,
                "anti_triggers": frame.anti_triggers,
                "boundary_notes": frame.boundary_notes,
            }
            for frame in schema.frames
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_schema(path: Path) -> FrameSchema:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = [
        Frame(
            frame_id=item["frame_id"],
            name=item["name"],
            description=item.get("description", ""),
            keywords=item.get("keywords", []),
            examples=item.get("examples", []),
            short_name=str(
                item.get("short_name")
                or (item.get("name", "") if item.get("name") else item.get("frame_id", ""))
            ).strip(),
            anti_triggers=item.get("anti_triggers", []),
            boundary_notes=item.get("boundary_notes", []),
        )
        for item in payload.get("frames", [])
    ]
    return FrameSchema(
        domain=payload.get("domain", ""),
        frames=frames,
        notes=payload.get("notes", ""),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Narrative framing induction + application workflow")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML configuration file",
        required=True
    )
    return parser.parse_args()


def run_pipeline(config: NarrativeFramingConfig, output_dir: Optional[Path] = None) -> Dict[str, object]:
    """Run the narrative framing pipeline (new modular approach).

    This is the new pipeline-based implementation with modular stages.
    Each stage is independently testable and has clear responsibilities.

    Args:
        config: Narrative framing configuration
        output_dir: Output directory (defaults to config.results_dir)

    Returns:
        Dictionary mapping stage names to their results
    """
    from apps.narrative_framing.pipeline import NarrativeFramingPipeline

    if output_dir is None:
        output_dir = config.results_dir or Path("results/narrative_framing")

    pipeline = NarrativeFramingPipeline(config, output_dir)
    results = pipeline.run()
    return results


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_pipeline(config)


if __name__ == "__main__":
    main()
