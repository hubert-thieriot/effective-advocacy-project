#!/usr/bin/env python3
"""Command-line entry point for the narrative framing workflow."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env if present (for WANDB_*, etc.)
load_dotenv()

from apps.narrative_framing.aggregation_document import (
    DocumentFrameAggregate,
    FrameAggregationStrategy,
    WeightedFrameAggregator,
    OccurrenceFrameAggregator,
)
from apps.narrative_framing.aggregation_temporal import (
    TemporalAggregator,
    period_aggregates_to_records,
)
from apps.narrative_framing.aggregation_domain import DomainAggregator
from apps.narrative_framing.config import ClassifierSettings, NarrativeFramingConfig, load_config
from apps.narrative_framing.report import write_html_report
from apps.narrative_framing.filtering import (
    make_filter_spec,
    filter_text as nf_filter_text,
    filter_chunks as nf_filter_chunks,
)

from efi_analyser.chunkers.sentence_chunker import SentenceChunker
from efi_analyser.embedders import SentenceTransformerEmbedder
from efi_analyser.frames import Frame, FrameAssignment, FrameInducer, FrameSchema, LLMFrameApplicator
from efi_analyser.frames.classifier import (
    CompositeCorpusSampler,
    FrameClassifierSpec,
    FrameClassifierTrainer,
    FrameLabelSet,
    SamplerConfig,
)
from efi_analyser.frames.classifier.model import FrameClassifierModel
from efi_analyser.scorers.openai_interface import OpenAIConfig, OpenAIInterface
from efi_corpus.embedded.embedded_corpus import EmbeddedCorpus
from efi_analyser.frames.identifiers import (
    make_global_doc_id,
    make_global_passage_id,
    split_global_doc_id,
    split_passage_id,
)

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
    chunk_classifications_dir: Optional[Path] = None
    frame_timeseries: Optional[Path] = None
    html: Optional[Path] = None


@dataclass
class ClassifierRun:
    predictions: List[Dict[str, object]]
    model: Optional[FrameClassifierModel]


def resolve_result_paths(results_dir: Optional[Path]) -> ResultPaths:
    if not results_dir:
        return ResultPaths()
    results_dir.mkdir(parents=True, exist_ok=True)
    classifier_dir = results_dir / "classifier"
    aggregates_dir = results_dir / "aggregates"
    aggregates_dir.mkdir(parents=True, exist_ok=True)
    return ResultPaths(
        schema=results_dir / "frame_schema.json",
        assignments=results_dir / "frame_assignments.json",
        classifier_predictions=results_dir / "frame_classifier_predictions.json",
        classifier_dir=classifier_dir,
        aggregates_dir=aggregates_dir,
        chunk_classifications_dir=results_dir / "classified_chunks",
        frame_timeseries=results_dir / "frame_timeseries.json",
        html=results_dir / "frame_report.html",
    )


def _publish_to_docs_assets(run_dir_name: str, results_html: Optional[Path]) -> None:
    """Copy exported Plotly PNGs and the HTML report into docs/ for GitHub Pages.

    - PNGs are expected in results/plots/<run_name>/plots
    - HTML report copied to docs/reports/<run_name>/frame_report.html
    """
    try:
        # Prefer plots under the run directory; fall back to legacy location
        plots_src = (results_html.parent / "plots") if results_html else None
        if not plots_src or not plots_src.exists():
            plots_src = Path("results/plots") / run_dir_name / "plots"
        plots_dst = Path("docs/assets") / run_dir_name / "plots"
        report_dst = Path("docs/reports") / run_dir_name
        if plots_src and plots_src.exists():
            plots_dst.mkdir(parents=True, exist_ok=True)
            for png in sorted(plots_src.glob("*.png")):
                shutil.copy2(png, plots_dst / png.name)
        if results_html and results_html.exists():
            report_dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(results_html, report_dst / "frame_report.html")
    except Exception as exc:
        print(f"⚠️ Failed to publish to docs: {exc}")


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


def _save_resolved_messages(directory: Path, name_prefix: str, messages_list: List[List[Dict[str, str]]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for idx, messages in enumerate(messages_list, start=1):
        for m in messages:
            role = str(m.get("role", "unknown")).lower()
            content = str(m.get("content", ""))
            out = directory / f"{name_prefix}_{idx:03d}_{role}.txt"
            out.write_text(content, encoding="utf-8")


def _copy_templates_to_results(paths_map: Dict[str, Path], out_dir: Path) -> None:
    dst_ind = out_dir / "prompts" / "induction" / "templates"
    dst_app = out_dir / "prompts" / "application" / "templates"
    dst_ind.mkdir(parents=True, exist_ok=True)
    dst_app.mkdir(parents=True, exist_ok=True)
    (dst_ind / "system.jinja").write_text(_read_text_or_fail(paths_map["induction_system"]), encoding="utf-8")
    (dst_ind / "user.jinja").write_text(_read_text_or_fail(paths_map["induction_user"]), encoding="utf-8")
    (dst_app / "system.jinja").write_text(_read_text_or_fail(paths_map["application_system"]), encoding="utf-8")
    (dst_app / "user.jinja").write_text(_read_text_or_fail(paths_map["application_user"]), encoding="utf-8")


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
        )
        for item in payload.get("frames", [])
    ]
    return FrameSchema(
        domain=payload.get("domain", ""),
        frames=frames,
        notes=payload.get("notes", ""),
    )


def save_assignments(path: Path, assignments: Sequence[FrameAssignment]) -> None:
    serialized = [
        {
            "passage_id": assignment.passage_id,
            "passage_text": assignment.passage_text,
            "probabilities": assignment.probabilities,
            "top_frames": assignment.top_frames,
            "rationale": assignment.rationale,
            "evidence_spans": assignment.evidence_spans,
            "metadata": assignment.metadata,
        }
        for assignment in assignments
    ]
    path.write_text(json.dumps(serialized, indent=2, ensure_ascii=False), encoding="utf-8")


def load_assignments(path: Path) -> List[FrameAssignment]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assignments: List[FrameAssignment] = []
    for item in payload:
        assignments.append(
            FrameAssignment(
                passage_id=item["passage_id"],
                passage_text=item.get("passage_text", ""),
                probabilities=item.get("probabilities", {}),
                top_frames=item.get("top_frames", []),
                rationale=item.get("rationale", ""),
                evidence_spans=item.get("evidence_spans", []),
                metadata=item.get("metadata", {}),
            )
        )
    return assignments


def save_classifier_predictions(path: Path, predictions: Sequence[Dict[str, object]]) -> None:
    path.write_text(json.dumps(list(predictions), indent=2, ensure_ascii=False), encoding="utf-8")


def load_classifier_predictions(path: Path) -> List[Dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data)


def load_document_aggregates(path: Path) -> List[DocumentFrameAggregate]:
    """Load document aggregates from a single JSON file."""
    if not path.exists():
        return []
    
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: Iterable[Dict[str, object]] = payload
    
    aggregates: List[DocumentFrameAggregate] = []
    for item in records:
        aggregates.append(
            DocumentFrameAggregate(
                doc_id=item["doc_id"],
                frame_scores={k: float(v) for k, v in item.get("frame_scores", {}).items()},
                total_weight=float(item.get("total_weight", 0.0)),
                published_at=item.get("published_at"),
                title=item.get("title"),
                url=item.get("url"),
                top_frames=list(item.get("top_frames", [])),
            )
        )
    return aggregates


def save_aggregates_json(path: Path, data: object) -> None:
    """Save aggregation data to JSON file."""
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def build_all_aggregates(
    aggregates_dir: Path,
    document_aggregates_weighted: List[DocumentFrameAggregate],
    document_aggregates_occurrence: List[DocumentFrameAggregate],
    frame_ids: List[str],
) -> Dict[str, object]:
    """Build all aggregation combinations and save them to files.
    
    Returns a dictionary with all aggregates for passing to the report.
    """
    aggregates = {}
    
    print("Building document aggregates...")
    # Save document aggregates
    if document_aggregates_weighted:
        save_aggregates_json(
            aggregates_dir / "documents_weighted.json",
            [agg.to_dict() for agg in document_aggregates_weighted]
        )
        aggregates["documents_weighted"] = document_aggregates_weighted
    if document_aggregates_occurrence:
        save_aggregates_json(
            aggregates_dir / "documents_occurrence.json",
            [agg.to_dict() for agg in document_aggregates_occurrence]
        )
        aggregates["documents_occurrence"] = document_aggregates_occurrence
    
    print("Building temporal aggregates...")
    # Temporal aggregates - year with/without zeros
    for keep_zeros in [True, False]:
        key_suffix = "with_zeros" if keep_zeros else "without_zeros"
        agg = TemporalAggregator(
            period="year",
            weight_by_document_weight=True,
            keep_documents_with_no_frames=keep_zeros
        )
        yearly_result = agg.aggregate(document_aggregates_weighted)
        save_aggregates_json(
            aggregates_dir / f"year_weighted_{key_suffix}.json",
            [
                {
                    "period_id": p.period_id,
                    "frame_scores": p.frame_scores,
                    "document_count": p.document_count,
                }
                for p in yearly_result
            ]
        )
        aggregates[f"year_weighted_{key_suffix}"] = yearly_result
    
    print("Building domain aggregates...")
    # Domain aggregates - with/without zeros
    for keep_zeros in [True, False]:
        key_suffix = "with_zeros" if keep_zeros else "without_zeros"
        
        domain_agg = DomainAggregator(
            keep_documents_with_no_frames=keep_zeros,
            weight_by_document_weight=True,
            avg_or_sum="avg"
        )
        domain_aggregates = domain_agg.aggregate(document_aggregates_weighted)
        
        # Convert to old format for compatibility
        domain_frame_summaries = [
            {
                "domain": da.domain,
                "count": da.document_count,
                "shares": da.frame_scores,
            }
            for da in domain_aggregates
        ]
        
        save_aggregates_json(
            aggregates_dir / f"domain_weighted_{key_suffix}.json",
            domain_frame_summaries
        )
        aggregates[f"domain_weighted_{key_suffix}"] = domain_frame_summaries
    
    print("Building global aggregates...")
    # Global aggregates - with/without zeros
    for keep_zeros in [True, False]:
        key_suffix = "with_zeros" if keep_zeros else "without_zeros"
        agg = TemporalAggregator(
            period="all",
            weight_by_document_weight=True,
            keep_documents_with_no_frames=keep_zeros
        )
        global_result = agg.aggregate(document_aggregates_weighted)
        if global_result:
            global_data = {
                "period_id": global_result[0].period_id,
                "frame_scores": global_result[0].frame_scores,
                "document_count": global_result[0].document_count,
            }
            save_aggregates_json(
                aggregates_dir / f"global_weighted_{key_suffix}.json",
                global_data
            )
            aggregates[f"global_weighted_{key_suffix}"] = global_result
    
    print("Building occurrence aggregates...")
    # Occurrence aggregates - global and yearly with/without zeros
    if document_aggregates_occurrence:
        # Global occurrence aggregates
        for keep_zeros in [True, False]:
            key_suffix = "with_zeros" if keep_zeros else "without_zeros"
            agg = TemporalAggregator(
                period="all",
                weight_by_document_weight=False,  # Occurrence: don't weight by document
                keep_documents_with_no_frames=keep_zeros
            )
            global_result = agg.aggregate(document_aggregates_occurrence)
            if global_result:
                global_data = {
                    "period_id": global_result[0].period_id,
                    "frame_scores": global_result[0].frame_scores,
                    "document_count": global_result[0].document_count,
                }
                save_aggregates_json(
                    aggregates_dir / f"global_occurrence_{key_suffix}.json",
                    global_data
                )
                aggregates[f"global_occurrence_{key_suffix}"] = global_result
        
        # Yearly occurrence aggregates
        for keep_zeros in [True, False]:
            key_suffix = "with_zeros" if keep_zeros else "without_zeros"
            agg = TemporalAggregator(
                period="year",
                weight_by_document_weight=False,  # Occurrence: don't weight by document
                keep_documents_with_no_frames=keep_zeros
            )
            yearly_result = agg.aggregate(document_aggregates_occurrence)
            save_aggregates_json(
                aggregates_dir / f"year_occurrence_{key_suffix}.json",
                [
                    {
                        "period_id": p.period_id,
                        "frame_scores": p.frame_scores,
                        "document_count": p.document_count,
                    }
                    for p in yearly_result
                ]
            )
            aggregates[f"year_occurrence_{key_suffix}"] = yearly_result
    
    print("Building time series with 30-day rolling average...")
    # 30-day running average time series
    # Don't normalize each period - compute avg_score, apply rolling average, then normalize globally
    # This prevents all frames from appearing equal due to daily normalization
    temporal_agg = TemporalAggregator(
        period="day",
        weight_by_document_weight=True,
        avg_or_sum="avg",
        rolling_window=30
    )
    frame_timeseries_aggregates = temporal_agg.aggregate(document_aggregates_weighted)
    frame_timeseries_records = period_aggregates_to_records(frame_timeseries_aggregates)
    save_aggregates_json(aggregates_dir / "time_series_30day.json", frame_timeseries_records)
    aggregates["time_series_30day"] = frame_timeseries_records
    return aggregates


def _get_embedded_corpus(
    corpora: Mapping[str, EmbeddedCorpus],
    corpus_name: Optional[str],
) -> EmbeddedCorpus:
    if corpus_name and corpus_name in corpora:
        return corpora[corpus_name]
    if len(corpora) == 1:
        return next(iter(corpora.values()))
    raise KeyError(f"Corpus '{corpus_name}' not found among {list(corpora.keys())}")


def _list_global_doc_ids(corpora: Mapping[str, EmbeddedCorpus]) -> List[str]:
    doc_ids: List[str] = []
    for corpus_name, embedded in corpora.items():
        for local_doc_id in embedded.corpus.list_ids():
            doc_ids.append(make_global_doc_id(corpus_name if len(corpora) > 1 else None, local_doc_id))
    return doc_ids


def save_chunk_classification(directory: Path, payload: Dict[str, object]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    doc_id = str(payload.get("doc_id"))
    if not doc_id:
        raise ValueError("Chunk classification payload missing doc_id")
    path = directory / f"{doc_id}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_chunk_classifications(directory: Path, doc_ids: Optional[Iterable[str]] = None) -> List[Dict[str, object]]:
    if not directory.exists():
        return []
    doc_id_filter = set(doc_ids) if doc_ids is not None else None
    payloads: List[Dict[str, object]] = []
    for child in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(child.read_text(encoding="utf-8"))
        except Exception:
            continue
        doc_id = str(payload.get("doc_id", child.stem)).strip()
        if doc_id_filter and doc_id not in doc_id_filter:
            continue
        payloads.append(payload)
    return payloads


def cleanup_cached_chunks_by_regex(
    directory: Path, 
    exclude_regexes: Sequence[str],
    dry_run: bool = False
) -> Tuple[int, List[str]]:
    """
    Remove cached chunk classification files that match any of the exclude regex patterns.
    
    Args:
        directory: Path to the classified_chunks directory
        exclude_regexes: List of regex patterns to match against chunk text
        dry_run: If True, only report what would be deleted without actually deleting
        
    Returns:
        Tuple of (number_of_files_removed, list_of_removed_doc_ids)
    """
    if not directory.exists():
        return 0, []
    
    import re
    
    # Compile regex patterns
    compiled_patterns = []
    for pattern in exclude_regexes:
        try:
            compiled_patterns.append(re.compile(pattern))
        except re.error as e:
            print(f"Warning: Invalid regex pattern '{pattern}': {e}")
            continue
    
    if not compiled_patterns:
        return 0, []
    
    removed_count = 0
    removed_doc_ids = []
    
    for child in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(child.read_text(encoding="utf-8"))
        except Exception:
            continue
            
        doc_id = str(payload.get("doc_id", child.stem)).strip()
        
        # Check if any chunk text matches any exclude pattern
        should_remove = False
        chunks = payload.get("chunks", [])
        
        for chunk in chunks:
            chunk_text = str(chunk.get("text", "")).strip()
            if not chunk_text:
                continue
                
            for pattern in compiled_patterns:
                if pattern.search(chunk_text):
                    should_remove = True
                    break
            if should_remove:
                break
        
        if should_remove:
            removed_doc_ids.append(doc_id)
            if not dry_run:
                child.unlink(missing_ok=True)
            removed_count += 1
    
    return removed_count, removed_doc_ids


def cleanup_cached_chunks_by_keywords(
    directory: Path, 
    required_keywords: Sequence[str],
    dry_run: bool = False
) -> Tuple[int, List[str]]:
    """
    Remove cached chunk classification files that don't contain any of the required keywords.
    
    Args:
        directory: Path to the classified_chunks directory
        required_keywords: List of keywords that must be present (OR logic)
        dry_run: If True, only report what would be deleted without actually deleting
        
    Returns:
        Tuple of (number_of_files_removed, list_of_removed_doc_ids)
    """
    if not directory.exists():
        return 0, []
    
    # Normalize keywords (lowercase)
    normalized_keywords = [str(k).strip().lower() for k in required_keywords if str(k).strip()]
    if not normalized_keywords:
        return 0, []
    
    removed_count = 0
    removed_doc_ids = []
    
    for child in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(child.read_text(encoding="utf-8"))
        except Exception:
            continue
            
        doc_id = str(payload.get("doc_id", child.stem)).strip()
        chunks = payload.get("chunks", [])
        
        # Check if any chunk contains any of the required keywords
        has_keyword = False
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            text = str(chunk.get("text", "")).lower()
            if any(kw in text for kw in normalized_keywords):
                has_keyword = True
                break
        
        # Remove if no keywords found (opposite of the filtering logic)
        if not has_keyword:
            removed_doc_ids.append(doc_id)
            if not dry_run:
                child.unlink(missing_ok=True)
            removed_count += 1
    
    return removed_count, removed_doc_ids


def write_chunk_classifications(directory: Path, documents: Sequence[Dict[str, object]]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    keep_doc_ids: set[str] = set()
    for payload in documents:
        doc_id = str(payload.get("doc_id"))
        if not doc_id:
            continue
        keep_doc_ids.add(doc_id)
        path = directory / f"{doc_id}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Remove stale files not present in the provided documents
    for existing in directory.glob("*.json"):
        if existing.stem not in keep_doc_ids:
            existing.unlink(missing_ok=True)


def enrich_assignments_with_metadata(
    assignments: Sequence[FrameAssignment],
    corpora: Mapping[str, EmbeddedCorpus],
) -> None:
    url_cache: Dict[str, Dict[str, str]] = {}
    for assignment in assignments:
        corpus_name, local_doc_id, _ = split_passage_id(assignment.passage_id)
        global_doc_id = make_global_doc_id(corpus_name if len(corpora) > 1 else None, local_doc_id)
        assignment.metadata["doc_id"] = local_doc_id
        assignment.metadata["global_doc_id"] = global_doc_id
        if corpus_name:
            assignment.metadata["corpus"] = corpus_name

        cache_key = global_doc_id if len(corpora) > 1 else local_doc_id
        if cache_key not in url_cache:
            embedded_corpus = _get_embedded_corpus(corpora, corpus_name)
            index_entry = embedded_corpus.corpus.get_index_entry(local_doc_id) or {}
            meta = embedded_corpus.corpus.get_metadata(local_doc_id)
            fetch_info = embedded_corpus.corpus.get_fetch_info(local_doc_id)
            merged_meta: Dict[str, str] = {}
            for source in (index_entry, meta, fetch_info):
                if not isinstance(source, dict):
                    continue
                for key, value in source.items():
                    if key not in merged_meta and isinstance(value, str):
                        merged_meta[key] = value

            doc_folder_path = embedded_corpus.corpus.layout.doc_dir(local_doc_id)
            doc_folder_path_abs = doc_folder_path.resolve() if doc_folder_path.exists() else doc_folder_path
            url_cache[cache_key] = {
                "url": merged_meta.get("url", ""),
                "title": merged_meta.get("title", ""),
                "published_at": merged_meta.get("published_at", ""),
                "doc_folder_path": str(doc_folder_path_abs),
            }

        cache_entry = url_cache[cache_key]
        if cache_entry.get("url"):
            assignment.metadata["url"] = cache_entry["url"]
        if cache_entry.get("title") and not assignment.metadata.get("title"):
            assignment.metadata["title"] = cache_entry["title"]
        if cache_entry.get("published_at") and not assignment.metadata.get("published_at"):
            assignment.metadata["published_at"] = cache_entry["published_at"]
        if cache_entry.get("doc_folder_path"):
            assignment.metadata["doc_folder_path"] = cache_entry["doc_folder_path"]


def print_schema(schema: FrameSchema) -> None:
    print(f"\nDomain: {schema.domain}")
    print(f"Frames discovered: {len(schema.frames)}")
    print("----------------------------------------")
    for frame in schema.frames:
        short_label = frame.short_name or (frame.name.split()[0] if frame.name else frame.frame_id)
        print(f"[{frame.frame_id}] {short_label} – {frame.name}")
        print(f"  Description: {frame.description}")
        if frame.keywords:
            print(f"  Keywords: {', '.join(frame.keywords)}")
        if frame.examples:
            for example in frame.examples:
                print(f"  Example: {example}")
        print()
    if schema.notes:
        print(f"Notes: {schema.notes}")


def preview_assignments(assignments: Sequence[FrameAssignment], limit: int = 5) -> None:
    if not assignments:
        print("\nNo assignments to display.")
        return
    print("\nSample frame assignments:")
    print("----------------------------------------")
    for assignment in assignments[:limit]:
        top_frames = ", ".join(
            f"{fid}:{assignment.probabilities[fid]:.2f}" for fid in assignment.top_frames
        )
        print(f"{assignment.passage_id} → {top_frames if top_frames else '—'}")
        if assignment.rationale:
            print(f"  Rationale: {assignment.rationale}")
        if assignment.evidence_spans:
            print(f"  Evidence: {assignment.evidence_spans}")
        print()


def train_and_apply_classifier(
    settings: ClassifierSettings,
    schema: FrameSchema,
    assignments: Sequence[FrameAssignment],
    samples: Sequence[Tuple[str, str]],
    output_dir: Optional[Path],
    *,
    run_name: Optional[str] = None,
) -> ClassifierRun:
    if not settings.enabled:
        return ClassifierRun(predictions=[], model=None)
    if not assignments:
        print("⚠️ No assignments available for classifier training; skipping classifier step.")
        return ClassifierRun(predictions=[], model=None)

    label_set = FrameLabelSet.from_assignments(schema, assignments, source="llm")
    if not label_set.passages:
        print("⚠️ Label set is empty; skipping classifier training.")
        return ClassifierRun(predictions=[], model=None)

    spec_kwargs = {
        "model_name": settings.model_name,
        "batch_size": settings.batch_size,
        "num_train_epochs": settings.num_train_epochs,
        "learning_rate": settings.learning_rate,
        "weight_decay": settings.weight_decay,
        "warmup_ratio": settings.warmup_ratio,
        "max_length": settings.max_length,
        "report_to": settings.report_to,
        "logging_dir": settings.logging_dir,
        "eval_threshold": settings.eval_threshold,
        "eval_top_k": settings.eval_top_k,
        "eval_steps": settings.eval_steps,
    }
    if run_name:
        spec_kwargs["run_name"] = run_name
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        spec_kwargs["output_dir"] = str(output_dir)
    else:
        print("⚠️ Results directory not configured; classifier artifacts will use the trainer default output path.")
    trainer = FrameClassifierTrainer(FrameClassifierSpec(**spec_kwargs))
    # Create a small dev split so we can track per-epoch metrics.
    # Ensure at least 1 dev example when possible.
    n_total = len(label_set.passages)
    base_train_ratio = 0.9
    base_dev_ratio = 0.1
    if n_total >= 2 and int(n_total * base_dev_ratio) == 0:
        dev_ratio = 1.0 / n_total
        train_ratio = max(0.0, 1.0 - dev_ratio)
    else:
        train_ratio = base_train_ratio
        dev_ratio = base_dev_ratio
    train_set, dev_set, _ = label_set.split(train_ratio=train_ratio, dev_ratio=dev_ratio, seed=13)
    print(
        f"→ Training classifier model: {settings.model_name} (epochs={settings.num_train_epochs}, lr={settings.learning_rate})\n"
        f"  Using {len(train_set.passages)} train / {len(dev_set.passages)} dev passages for evaluation"
    )
    model = trainer.train(train_set, eval_set=dev_set)
    print("→ Classifier training complete.")

    # Optional: cross-validation to assess overfitting/generalization
    if getattr(settings, "cv_folds", None) and int(settings.cv_folds) >= 2:
        try:
            k = int(settings.cv_folds)
        except Exception:
            k = 0
        if k >= 2:
            print(f"→ Running {k}-fold cross-validation for classifier overfitting check…")
            try:
                trainer.cross_validate(label_set, k_folds=k, seed=settings.seed)
            except Exception as e:
                print(f"⚠️ Cross-validation failed: {e}")

    texts = [text for _, text in samples]
    if not texts:
        print("⚠️ No passages available for classifier inference; skipping predictions.")
        return ClassifierRun(predictions=[], model=model)

    inference_batch = max(1, settings.inference_batch_size or settings.batch_size)
    print(
        f"→ Running classifier inference on {len(texts)} passages (batch size {inference_batch})..."
    )
    
    # Add progress bar for classifier inference
    from tqdm import tqdm
    with tqdm(total=len(texts), desc="Classifier inference", unit="passages", 
             bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        probs = model.predict_proba_batch(texts, batch_size=inference_batch, progress_callback=lambda n: pbar.update(n))
    predictions: List[Dict[str, object]] = []
    for (passage_id, _), prob_dict in zip(samples, probs):
        ordered = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
        predictions.append(
            {
                "passage_id": passage_id,
                "probabilities": prob_dict,
                "top_frames": [fid for fid, _ in ordered[:3]],
            }
        )

    if output_dir is not None:
        model.save(output_dir)
    else:
        print("⚠️ Skipping classifier model persistence (no results directory provided).")
    return ClassifierRun(predictions=predictions, model=model)


def classify_corpus_chunks(
    model: FrameClassifierModel,
    corpora: Mapping[str, EmbeddedCorpus],
    batch_size: int,
    sample_size: Optional[int] = None,
    seed: Optional[int] = None,
    output_dir: Optional[Path] = None,
    doc_ids: Optional[Sequence[str]] = None,
    *,
    require_keywords: Optional[Sequence[str]] = None,
    exclude_regex: Optional[Sequence[str]] = None,
    exclude_min_hits: Optional[Dict[str, int]] = None,
    trim_after_markers: Optional[Sequence[str]] = None,
) -> List[Dict[str, object]]:
    if doc_ids is not None:
        doc_id_list = list(doc_ids)
    else:
        doc_id_list = _list_global_doc_ids(corpora)
    if not doc_id_list:
        return []

    if doc_ids is None and sample_size is not None:
        limited = max(0, min(sample_size, len(doc_id_list)))
        rng = random.Random(seed)
        rng.shuffle(doc_id_list)
        doc_id_list = doc_id_list[:limited]
    elif doc_ids is None:
        # When no explicit sample size and doc_ids not provided, respect original ordering.
        doc_id_list = list(doc_id_list)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    classified_documents: List[Dict[str, object]] = []

    # Unified filtering spec
    spec = make_filter_spec(
        exclude_regex=exclude_regex,
        exclude_min_hits=exclude_min_hits,
        trim_after_markers=trim_after_markers,
        keywords=require_keywords,
    )

    iterator = tqdm(
        doc_id_list,
        desc="Classifying documents",
        unit="doc",
        leave=False,
    )

    for doc_id in iterator:
        corpus_name, local_doc_id = split_global_doc_id(doc_id)
        embedded_corpus = _get_embedded_corpus(corpora, corpus_name)

        doc = embedded_corpus.corpus.get_document(local_doc_id)
        if doc is None:
            continue

        published_at = doc.published_at
        if not published_at:
            metadata = embedded_corpus.corpus.get_metadata(local_doc_id)
            if metadata and isinstance(metadata, dict):
                published_at = metadata.get("published_at")

        chunks = embedded_corpus.get_chunks(local_doc_id, materialize_if_necessary=True) or []
        texts: List[str] = []
        chunk_text_pairs: List[Tuple[str, str]] = []
        for chunk in chunks:
            text = nf_filter_text((chunk.text or ""), spec)
            if not text:
                continue
            local_passage_id = f"{local_doc_id}:chunk{int(chunk.chunk_id):03d}"
            chunk_id = make_global_passage_id(
                corpus_name if len(corpora) > 1 else None,
                local_passage_id,
            )
            texts.append(text)
            chunk_text_pairs.append((chunk_id, text))

        if not texts:
            continue

        # Apply optional keyword gate: skip classification if none of the chunks
        # contain any of the required keywords (case-insensitive substring).
        if spec.keywords is not None:
            lowered_texts = [t.lower() for t in texts]
            if not any(any(kw in t for kw in spec.keywords) for t in lowered_texts):
                continue

        probabilities = model.predict_proba_batch(texts, batch_size=batch_size)
        chunk_records: List[Dict[str, object]] = []
        for (chunk_id, passage_text), probs in zip(chunk_text_pairs, probabilities):
            ordered = sorted(probs.items(), key=lambda item: item[1], reverse=True)
            chunk_records.append(
                {
                    "chunk_id": chunk_id,
                    "text": passage_text,
                    "probabilities": {frame_id: float(score) for frame_id, score in probs.items()},
                    "top_frames": [fid for fid, _ in ordered[:3]],
                }
            )

        payload = {
            "doc_id": doc_id,
            "corpus": corpus_name,
            "local_doc_id": local_doc_id,
            "title": doc.title,
            "url": doc.url,
            "published_at": published_at,
            "chunks": chunk_records,
        }

        classified_documents.append(payload)
        if output_dir:
            save_chunk_classification(output_dir, payload)

    iterator.close()

    return classified_documents


def aggregate_classified_documents(
    documents: Sequence[Dict[str, object]],
    schema: FrameSchema,
    aggregator: Optional[FrameAggregationStrategy] = None,
    *,
    require_keywords: Optional[Sequence[str]] = None,
    exclude_regex: Optional[Sequence[str]] = None,
    exclude_min_hits: Optional[Dict[str, int]] = None,
    trim_after_markers: Optional[Sequence[str]] = None,
) -> Sequence[DocumentFrameAggregate]:
    if not documents:
        return []

    frame_ids = [frame.frame_id for frame in schema.frames]
    active_aggregator = aggregator or WeightedFrameAggregator(frame_ids)

    # Unified filter spec
    spec = make_filter_spec(
        exclude_regex=exclude_regex,
        exclude_min_hits=exclude_min_hits,
        trim_after_markers=trim_after_markers,
        keywords=require_keywords,
    )

    for doc_record in documents:
        doc_id = str(doc_record.get("doc_id"))
        chunks = doc_record.get("chunks", [])
        if not doc_id or not isinstance(chunks, Sequence):
            continue

        # Apply per-chunk cleanup/exclusion before aggregation
        filtered_chunks: List[Dict[str, object]] = nf_filter_chunks(chunks, spec)

        if not filtered_chunks:
            continue

        # Apply optional keyword filter at the document level: require at least
        # one chunk to contain any of the configured keywords.
        if spec.keywords is not None:
            # Skip this document entirely if no chunk contains a keyword
            has_keyword = False
            for chunk in filtered_chunks:
                if not isinstance(chunk, dict):
                    continue
                text = str(chunk.get("text", "")).lower()
                if any(kw in text for kw in spec.keywords):
                    has_keyword = True
                    break
            if not has_keyword:
                continue
        published_at = doc_record.get("published_at")
        title = doc_record.get("title")
        url = doc_record.get("url")
        for chunk in filtered_chunks:
            if not isinstance(chunk, dict):
                continue
            passage_text = str(chunk.get("text", ""))
            if not passage_text.strip():
                continue
            probabilities = chunk.get("probabilities", {})
            if isinstance(probabilities, dict):
                probs = {fid: float(val) for fid, val in probabilities.items()}
            else:
                probs = {}
            active_aggregator.accumulate(
                doc_id=doc_id,
                passage_text=passage_text,
                probabilities=probs,
                published_at=published_at,
                title=title,
                url=url,
            )

    return active_aggregator.finalize()



def run_workflow(config: NarrativeFramingConfig) -> None:
    # ============================================================================
    # INITIALIZATION AND SETUP
    # ============================================================================
    corpus_names = list(config.iter_corpus_names())
    if not corpus_names:
        raise ValueError("At least one corpus must be configured.")

    missing_corpora = [name for name in corpus_names if not (config.corpora_root / name).exists()]
    if missing_corpora:
        raise FileNotFoundError(
            f"Corpus paths not found: {', '.join(str(config.corpora_root / name) for name in missing_corpora)}"
        )
    config.workspace_root.mkdir(parents=True, exist_ok=True)
    paths = resolve_result_paths(config.results_dir)

    # Snapshot the input configuration into results/configs with timestamp
    try:
        base_results_dir = config.results_dir or Path("results/narrative_framing")
        cfg_dir = base_results_dir / "configs"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Prefer copying original YAML if known
        src_path = getattr(config, "source_config_path", None)
        if isinstance(src_path, Path) and src_path.exists():
            dst_name = f"{src_path.stem}_{ts}{src_path.suffix}"
            shutil.copy2(src_path, cfg_dir / dst_name)
        else:
            # Fall back to dumping current config state as YAML
            try:
                from dataclasses import asdict
                import yaml  # type: ignore
                data = asdict(config)
                # Convert Path objects to strings for YAML serialization
                for k, v in list(data.items()):
                    if isinstance(v, Path):
                        data[k] = str(v)
                dst_name = f"config_{ts}.yaml"
                (cfg_dir / dst_name).write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
            except Exception as exc:
                print(f"⚠️ Failed to dump config to YAML: {exc}")
    except Exception as exc:
        print(f"⚠️ Failed to snapshot configuration: {exc}")

    # ============================================================================
    # VARIABLE INITIALIZATION
    # ============================================================================
    
    schema: Optional[FrameSchema] = None
    assignments: List[FrameAssignment] = []
    application_samples: List[Tuple[str, str]] = []
    classifier_predictions: List[Dict[str, object]] = []
    classifier_model: Optional[FrameClassifierModel] = None
    chunk_classifications: List[Dict[str, object]] = []
    document_aggregates_weighted: List[DocumentFrameAggregate] = []
    all_aggregates: Dict[str, object] = {}

    induction_samples: List[Tuple[str, str]] = []
    induction_reused = False
    assignments_reused = False
    target_application_count = max(0, config.application_sample_size)
    
    # =============================================================================
    # PREPARE EMBEDDED CORPUS
    # =============================================================================
    if len(corpus_names) == 1:
        singular_path = config.corpora_root / corpus_names[0]
        print(f"Loading embedded corpus from {singular_path}...")
    else:
        joined = ", ".join(corpus_names)
        print(f"Loading embedded corpora: {joined}")

    if TextChunker is not None:
        try:
            chunker = TextChunker(TextChunkerConfig(max_words=config.target_words))
        except Exception as exc:
            print(
                "⚠️ Falling back to sentence chunker because TextChunker initialization failed:",
                exc,
            )
            chunker = SentenceChunker()
    else:
        if _TEXT_CHUNKER_ERROR is not None:
            print(
                "⚠️ Falling back to sentence chunker because spaCy TextChunker is unavailable:",
                _TEXT_CHUNKER_ERROR,
            )
        chunker = SentenceChunker()
    embedder = SentenceTransformerEmbedder(lazy_load=True)
    corpora_map: Dict[str, EmbeddedCorpus] = {}
    for name in corpus_names:
        corpora_map[name] = EmbeddedCorpus(
            corpus_path=config.corpora_root / name,
            workspace_path=config.workspace_root,
            chunker=chunker,
            embedder=embedder,
        )
    sampler = CompositeCorpusSampler(corpora_map, policy=config.sampling_policy)
    keywords = config.filter_keywords

    # ============================================================================
    # CONFIGURE LLM CLIENTS
    # ============================================================================
    
    # Use model-appropriate temperature defaults, with config overrides
    induction_temp = config.induction_temperature if config.induction_temperature is not None else OpenAIConfig.get_default_temperature(config.induction_model)
    application_temp = config.application_temperature if config.application_temperature is not None else OpenAIConfig.get_default_temperature(config.application_model)
    
    openai_config = OpenAIConfig(
        model=config.induction_model,
        temperature=induction_temp,
        timeout=600.0,
        ignore_cache=False,
        verbose=True,  # Enable warnings for temperature overrides
    )
    applicator_config = OpenAIConfig(
        model=config.application_model,
        temperature=application_temp,
        timeout=600.0,
        ignore_cache=False,
        verbose=False,  # Disable verbose logging to simplify output
    )

    # ============================================================================
    # FRAME INDUCTION (OR RELOAD FROM CACHE)
    # ============================================================================
    
    # Resolve and validate default prompt templates; copy raw templates into results
    prompt_paths = _resolve_default_prompt_paths()
    _copy_templates_to_results(prompt_paths, config.results_dir or Path("results/narrative_framing"))

    ind_sys_t = _read_text_or_fail(prompt_paths["induction_system"])
    ind_usr_t = _read_text_or_fail(prompt_paths["induction_user"])
    app_sys_t = _read_text_or_fail(prompt_paths["application_system"])
    app_usr_t = _read_text_or_fail(prompt_paths["application_user"])

    if config.regenerate_report_only:
        # In regenerate mode, require cached schema to exist
        if not (paths.schema and paths.schema.exists()):
            raise FileNotFoundError("Cannot regenerate report: cached frame_schema.json not found")
        schema = load_schema(paths.schema)
        induction_reused = True
        print(f"Reloaded frame schema from {paths.schema} (regenerate mode)")
    elif config.reload_induction and paths.schema and paths.schema.exists():
        schema = load_schema(paths.schema)
        induction_reused = True
        print(f"Reloaded frame schema from {paths.schema}")
    else:
        if config.reload_induction and (not paths.schema or not paths.schema.exists()):
            print("⚠️ Cached schema not found; running induction instead.")
        print("Collecting passages for frame induction...")
        induction_samples = sampler.collect(
            SamplerConfig(
                sample_size=config.induction_sample_size,
                seed=config.seed,
                keywords=keywords,
                exclude_regex=config.filter_exclude_regex,
                exclude_min_hits=config.filter_exclude_min_hits,
                trim_after_markers=config.filter_trim_after_markers,
            )
        )
        print(f"Collected {len(induction_samples)} passages for induction.")
        induction_passages = [text for _, text in induction_samples]
        inducer_client = OpenAIInterface(name="frame_induction", config=openai_config)
        inducer = FrameInducer(
            llm_client=inducer_client,
            domain=config.domain,
            frame_target=config.induction_frame_target,
            max_passages_per_call=max(20, min(config.induction_sample_size, 80)),
            max_total_passages=config.induction_sample_size * 2,
            induction_guidance=config.induction_guidance,
            system_template=ind_sys_t,
            user_template=ind_usr_t,
        )
        schema = inducer.induce(induction_passages)
        if not schema.schema_id:
            schema.schema_id = config.domain.replace(" ", "_")
        print(f"Induction produced {len(schema.frames)} frames.")

        if paths.schema:
            save_schema(paths.schema, schema)
        # Save resolved induction prompts (all calls) under results/prompts/induction/resolved
        if getattr(inducer, "emitted_messages", None):
            resolved_dir = (config.results_dir or Path("results")) / "prompts" / "induction" / "resolved"
            _save_resolved_messages(resolved_dir, "induction_call", inducer.emitted_messages)

    if schema is None:
        raise RuntimeError("Frame schema is required to proceed. Ensure induction completes successfully.")

    # ============================================================================
    # FRAME APPLICATION (OR RELOAD FROM CACHE)
    # ============================================================================
    assignment_map: Dict[str, FrameAssignment] = {}
    assignment_list: List[FrameAssignment] = []

    if config.regenerate_report_only:
        # In regenerate mode, try to load assignments from cache (optional)
        if paths.assignments and paths.assignments.exists():
            try:
                cached_assignments = load_assignments(paths.assignments)
                for cached in cached_assignments:
                    if cached.passage_id in assignment_map:
                        continue
                    assignment_map[cached.passage_id] = cached
                    assignment_list.append(cached)
                if assignment_list:
                    assignments_reused = True
                    print(f"Reloaded {len(assignment_list)} LLM frame assignments from cache (regenerate mode).")
            except Exception as exc:
                print(f"⚠️ Failed to load cached assignments: {exc}")
        # Skip application work in regenerate mode
        assignments = assignment_list
        application_samples = [
            (item.passage_id, item.passage_text) for item in assignment_list
        ]
        target_application_count = max(0, config.application_sample_size)
    elif (
        config.reload_application
        and paths.assignments
        and paths.assignments.exists()
    ):
        try:
            cached_assignments = load_assignments(paths.assignments)
            for cached in cached_assignments:
                if cached.passage_id in assignment_map:
                    continue
                assignment_map[cached.passage_id] = cached
                assignment_list.append(cached)
            if assignment_list:
                assignments_reused = True
                print(f"Reloaded {len(assignment_list)} LLM frame assignments from cache.")
        except Exception as exc:
            print(f"⚠️ Failed to load cached assignments: {exc}")

        target_application_count = max(0, config.application_sample_size)
        existing_count = len(assignment_list)
        additional_needed = max(0, target_application_count - existing_count)

        if additional_needed > 0:
            exclude_ids: List[str] = list(assignment_map.keys())
            if induction_samples:
                exclude_ids.extend(pid for pid, _ in induction_samples)
            try:
                new_samples = sampler.collect(
                    SamplerConfig(
                        sample_size=additional_needed,
                        seed=config.seed + 1,
                        keywords=keywords,
                        exclude_passage_ids=exclude_ids or None,
                        exclude_regex=config.filter_exclude_regex,
                        exclude_min_hits=config.filter_exclude_min_hits,
                        trim_after_markers=config.filter_trim_after_markers,
                    )
                )
            except ValueError as exc:
                print(f"⚠️ Unable to gather {additional_needed} new passages for application: {exc}")
                new_samples = []

            if new_samples:
                applicator_client = OpenAIInterface(name="frame_application", config=applicator_config)
                applicator = LLMFrameApplicator(
                    llm_client=applicator_client,
                    batch_size=config.application_batch_size,
                    max_chars_per_passage=None,
                    chunk_overlap_chars=0,
                    system_template=app_sys_t,
                    user_template=app_usr_t,
                )
                with tqdm(
                    total=len(new_samples),
                    desc=f"Applying frames ({applicator_client.config.model})",
                    unit="passages",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
                ) as pbar:
                    batch_size = applicator.batch_size
                    for i in range(0, len(new_samples), batch_size):
                        batch = new_samples[i : i + batch_size]
                        batch_assignments = applicator.batch_assign(
                            schema,
                            batch,
                            top_k=config.application_top_k,
                        )
                        for assignment in batch_assignments:
                            if assignment.passage_id in assignment_map:
                                continue
                            assignment_map[assignment.passage_id] = assignment
                            assignment_list.append(assignment)
                        pbar.update(len(batch))
                print(
                    f"Received {len(assignment_list) - existing_count} new assignments; total cached assignments: {len(assignment_list)}."
                )
                if paths.assignments:
                    save_assignments(paths.assignments, assignment_list)
                # Save resolved application prompts used across batches
                if getattr(applicator, "emitted_messages", None):
                    resolved_dir = (config.results_dir or Path("results")) / "prompts" / "application" / "resolved"
                    _save_resolved_messages(resolved_dir, "application_batch", applicator.emitted_messages)
            else:
                print(
                    f"⚠️ No new passages were sampled; continuing with {len(assignment_list)} cached assignments."
                )

    # Limit to requested application_sample_size when cache contains more
    if target_application_count and len(assignment_list) > target_application_count:
        # Use a deterministic shuffle so repeated runs with the same seed are reproducible
        print(
            f"Limiting cached assignments from {len(assignment_list)} to a shuffled subset of {target_application_count} for this run."
        )
        rng = random.Random(config.seed + 97)
        indices = list(range(len(assignment_list)))
        rng.shuffle(indices)
        picked = set(indices[:target_application_count])
        assignments = [assignment_list[i] for i in indices[:target_application_count]]
        application_samples = [
            (assignment_list[i].passage_id, assignment_list[i].passage_text) for i in indices[:target_application_count]
        ]
    else:
        assignments = assignment_list
        application_samples = [
            (item.passage_id, item.passage_text) for item in assignment_list
        ]

    if target_application_count > 0 and not assignments and not application_samples:
        # No cached assignments and no new sampling occurred; need to generate initial set.
        application_samples = sampler.collect(
            SamplerConfig(
                sample_size=config.application_sample_size,
                seed=config.seed + 1,
                keywords=keywords,
                exclude_passage_ids=None,
            )
        )
        applicator_client = OpenAIInterface(name="frame_application", config=applicator_config)
        applicator = LLMFrameApplicator(
            llm_client=applicator_client,
            batch_size=config.application_batch_size,
            max_chars_per_passage=None,
            chunk_overlap_chars=0,
            system_template=app_sys_t,
            user_template=app_usr_t,
        )
        print(f"Applying frames to {len(application_samples)} passages using {applicator_client.name}...")
        with tqdm(
            total=len(application_samples),
            desc="Applying frames",
            unit="passages",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
        ) as pbar:
            batch_size = applicator.batch_size
            for i in range(0, len(application_samples), batch_size):
                batch = application_samples[i : i + batch_size]
                batch_assignments = applicator.batch_assign(
                    schema,
                    batch,
                    top_k=config.application_top_k,
                )
                for assignment in batch_assignments:
                    if assignment.passage_id in assignment_map:
                        continue
                    assignment_map[assignment.passage_id] = assignment
                    assignment_list.append(assignment)
                pbar.update(len(batch))
        assignments = assignment_list
        if paths.assignments:
            save_assignments(paths.assignments, assignments)
        print(f"Received {len(assignments)} assignments from LLM.")
        if getattr(applicator, "emitted_messages", None):
            resolved_dir = (config.results_dir or Path("results")) / "prompts" / "application" / "resolved"
            _save_resolved_messages(resolved_dir, "application_batch", applicator.emitted_messages)

    # Ensure application_samples covers at least available assignments when none were set
    if not application_samples and assignments:
        application_samples = [
            (item.passage_id, item.passage_text) for item in assignments
        ]

    # ============================================================================
    # POST-PROCESS ASSIGNMENTS AND DISPLAY RESULTS
    # ============================================================================
    # Always persist the full cached set with metadata (not the run-limited slice)
    if assignment_list:
        enrich_assignments_with_metadata(assignment_list, corpora_map)
        if paths.assignments:
            save_assignments(paths.assignments, assignment_list)
        print("Assignments enriched with document metadata and saved to cache (full set).")

    print_schema(schema)

    if assignments:
        preview_assignments(assignments, limit=5)

    # ============================================================================
    # LOAD CLASSIFIER
    # ============================================================================
    
    if config.regenerate_report_only:
        # In regenerate mode, try to load classifier predictions and model from cache (optional)
        if paths.classifier_predictions and paths.classifier_predictions.exists():
            try:
                classifier_predictions = load_classifier_predictions(paths.classifier_predictions)
                print(f"Reloaded {len(classifier_predictions)} classifier predictions from cache (regenerate mode).")
            except Exception as exc:
                print(f"⚠️ Failed to load cached classifier predictions: {exc}")
        if paths.classifier_dir and paths.classifier_dir.exists():
            try:
                classifier_model = FrameClassifierModel.load(paths.classifier_dir)
                print(f"Reloaded classifier model from {paths.classifier_dir} (regenerate mode).")
            except FileNotFoundError:
                classifier_model = None
            except Exception as exc:
                print(f"⚠️ Failed to load classifier model from {paths.classifier_dir}: {exc}")
                classifier_model = None
    else:
        if (
            config.classifier.enabled
            and config.reload_classifier
            and paths.classifier_predictions
            and paths.classifier_predictions.exists()
            and assignments_reused
        ):
            classifier_predictions = load_classifier_predictions(paths.classifier_predictions)
            print(f"Reloaded {len(classifier_predictions)} classifier predictions from cache.")

        if (
            config.classifier.enabled
            and config.reload_classifier
            and classifier_model is None
            and paths.classifier_dir is not None
        ):
            try:
                classifier_model = FrameClassifierModel.load(paths.classifier_dir)
                print(f"Reloaded classifier model from {paths.classifier_dir}.")
            except FileNotFoundError:
                classifier_model = None
            except Exception as exc:  # pragma: no cover - informational only
                print(f"⚠️ Failed to load classifier model from {paths.classifier_dir}: {exc}")
                classifier_model = None
        elif (
            config.classifier.enabled
            and config.reload_classifier
            and classifier_model is None
            and paths.classifier_dir is None
        ):
            print("⚠️ Results directory not provided; skipping classifier model reload.")

    # ============================================================================
    # TRAIN CLASSIFIER (IF NEEDED)
    # ============================================================================
    
    if config.regenerate_report_only:
        # Skip classifier training in regenerate mode
        pass
    elif schema and assignments and config.classifier.enabled:
        need_training = not classifier_predictions or classifier_model is None
        if need_training:
            print("Training classifier on LLM-labeled passages...")
            # Build a readable run name that starts with the corpus name(s)
            if len(corpus_names) == 1:
                _run_name = f"{corpus_names[0]}-frame-cls"
            else:
                _run_name = f"{'+' .join(corpus_names)}-frame-cls"

            classifier_run = train_and_apply_classifier(
                settings=config.classifier,
                schema=schema,
                assignments=assignments,
                samples=application_samples
                if application_samples
                else [(a.passage_id, a.passage_text) for a in assignments],
                output_dir=paths.classifier_dir,
                run_name=_run_name,
            )
            classifier_predictions = classifier_run.predictions
            classifier_model = classifier_run.model
            if classifier_predictions:
                print(f"Generated classifier predictions for {len(classifier_predictions)} passages.")
            if classifier_predictions and paths.classifier_predictions:
                save_classifier_predictions(paths.classifier_predictions, classifier_predictions)

    # ============================================================================
    # PREPARE DISPLAY ASSIGNMENTS
    # ============================================================================
    
    display_assignments = list(assignments)
    if schema and application_samples:
        existing_ids = {assignment.passage_id for assignment in display_assignments}
        zero_probs_template = {frame.frame_id: 0.0 for frame in schema.frames}
        for passage_id, text in application_samples:
            if passage_id in existing_ids:
                continue
            display_assignments.append(
                FrameAssignment(
                    passage_id=passage_id,
                    passage_text=text,
                    probabilities=zero_probs_template.copy(),
                    top_frames=[],
                    rationale="",
                    evidence_spans=[],
                )
            )
            
    # ============================================================================
    # CLASSIFY CORPUS DOCUMENTS
    # ============================================================================

    total_doc_ids = _list_global_doc_ids(corpora_map)
    desired_doc_total = config.classifier_corpus_sample_size or len(total_doc_ids)
    desired_doc_total = max(0, min(desired_doc_total, len(total_doc_ids)))

    chunk_classification_map: Dict[str, Dict[str, object]] = {}

    if config.regenerate_report_only:
        # In regenerate mode, try to load chunk classifications from cache (optional)
        if paths.chunk_classifications_dir and paths.chunk_classifications_dir.exists():
            cached_docs = load_chunk_classifications(paths.chunk_classifications_dir)
            for payload in cached_docs:
                doc_id = str(payload.get("doc_id", "")).strip()
                if not doc_id or doc_id in chunk_classification_map:
                    continue
                chunk_classification_map[doc_id] = payload
            if chunk_classification_map:
                print(
                    f"Reloaded chunk classifications for {len(chunk_classification_map)} documents from cache (regenerate mode)."
                )
        # Skip classification work in regenerate mode
    elif (
        config.reload_chunk_classifications
        and paths.chunk_classifications_dir
        and paths.chunk_classifications_dir.exists()
    ):
        # Clean up cached chunks that match exclude regex patterns
        if hasattr(config, 'filter_exclude_regex') and config.filter_exclude_regex:
            print("Cleaning up cached chunks matching exclude regex patterns...")
            removed_count, removed_doc_ids = cleanup_cached_chunks_by_regex(
                paths.chunk_classifications_dir, 
                config.filter_exclude_regex,
                dry_run=False
            )
            if removed_count > 0:
                print(f"🗑️  Removed {removed_count} cached chunk files matching exclude patterns:")
                for doc_id in removed_doc_ids[:10]:  # Show first 10
                    print(f"   - {doc_id}")
                if len(removed_doc_ids) > 10:
                    print(f"   ... and {len(removed_doc_ids) - 10} more")
            else:
                print("✅ No cached chunks matched exclude patterns")
        
        cached_docs = load_chunk_classifications(paths.chunk_classifications_dir)
        for payload in cached_docs:
            doc_id = str(payload.get("doc_id", "")).strip()
            if not doc_id or doc_id in chunk_classification_map:
                continue
            chunk_classification_map[doc_id] = payload
        if chunk_classification_map:
            print(
                f"Reloaded chunk classifications for {len(chunk_classification_map)} documents from cache."
            )

    if config.regenerate_report_only:
        # Skip classification work in regenerate mode
        pass
    elif (
        schema
        and classifier_model
        and desired_doc_total > len(chunk_classification_map)
    ):
        remaining_needed = desired_doc_total - len(chunk_classification_map)
        remaining_doc_ids = [doc_id for doc_id in total_doc_ids if doc_id not in chunk_classification_map]
        if remaining_doc_ids and remaining_needed > 0:
            rng = random.Random(config.seed)
            rng.shuffle(remaining_doc_ids)
            doc_ids_to_classify = remaining_doc_ids[:remaining_needed]
            print(
                f"Classifying {len(doc_ids_to_classify)} additional documents to reach target sample of {desired_doc_total}."
            )
            inference_batch = max(1, config.classifier.inference_batch_size or config.classifier.batch_size)
            new_docs = classify_corpus_chunks(
                model=classifier_model,
                corpora=corpora_map,
                batch_size=inference_batch,
                seed=config.seed,
                output_dir=paths.chunk_classifications_dir,
                doc_ids=doc_ids_to_classify,
                require_keywords=config.filter_keywords,
                exclude_regex=config.filter_exclude_regex,
                exclude_min_hits=config.filter_exclude_min_hits,
                trim_after_markers=config.filter_trim_after_markers,
            )
            for payload in new_docs:
                doc_id = str(payload.get("doc_id", "")).strip()
                if doc_id:
                    chunk_classification_map.setdefault(doc_id, payload)
        if desired_doc_total > len(chunk_classification_map):
            print(
                f"⚠️ Only {len(chunk_classification_map)} documents classified; target was {desired_doc_total}."
            )

    # Determine which classified documents to use this run (without discarding extras on disk).
    ordered_docs = [doc_id for doc_id in total_doc_ids if doc_id in chunk_classification_map]
    if len(ordered_docs) < len(chunk_classification_map):
        # Include any cached docs that might no longer be in corpus ordering.
        ordered_docs.extend(
            doc_id for doc_id in chunk_classification_map.keys() if doc_id not in ordered_docs
        )

    chunk_classifications = []
    for doc_id in ordered_docs:
        if desired_doc_total and len(chunk_classifications) >= desired_doc_total:
            break
        chunk_classifications.append(chunk_classification_map[doc_id])

    if not chunk_classifications and chunk_classification_map:
        chunk_classifications = list(chunk_classification_map.values())
    elif desired_doc_total and len(chunk_classifications) < len(chunk_classification_map):
        print(
            f"Using first {len(chunk_classifications)} classified documents from {len(chunk_classification_map)} available."
        )

    if chunk_classification_map:
        print(
            f"Prepared {len(chunk_classifications)} classified documents (cached total: {len(chunk_classification_map)})."
        )
        
    
    # ============================================================================
    # PREPARE DOCUMENT AGGREGATES
    # ============================================================================
    
    document_aggregates_occurrence: List[DocumentFrameAggregate] = []

    if schema and chunk_classifications and not document_aggregates_weighted:
        # Weighted aggregates (length-weighted, threshold, normalize per config)
        weighted_agg = WeightedFrameAggregator(
            [frame.frame_id for frame in schema.frames],
            top_k=config.application_top_k,
            min_threshold=config.agg_min_threshold_weighted,
            normalize=config.agg_normalize_weighted,
        )
        document_aggregates_weighted = list(
            aggregate_classified_documents(
                documents=chunk_classifications,
                schema=schema,
                aggregator=weighted_agg,
                require_keywords=config.filter_keywords,
                exclude_regex=config.filter_exclude_regex,
                exclude_min_hits=config.filter_exclude_min_hits,
                trim_after_markers=config.filter_trim_after_markers,
            )
        )

        # Occurrence aggregates (binary presence by threshold)
        occurrence_agg = OccurrenceFrameAggregator(
            [frame.frame_id for frame in schema.frames],
            min_threshold=config.agg_min_threshold_occurrence,
            top_k=config.application_top_k,
        )
        document_aggregates_occurrence = list(
            aggregate_classified_documents(
                documents=chunk_classifications,
                schema=schema,
                aggregator=occurrence_agg,
                require_keywords=config.filter_keywords,
                exclude_regex=config.filter_exclude_regex,
                exclude_min_hits=config.filter_exclude_min_hits,
                trim_after_markers=config.filter_trim_after_markers,
            )
        )

    # ============================================================================
    # BUILD ALL AGGREGATES
    # ============================================================================
    
    if schema and document_aggregates_weighted and paths.aggregates_dir:
        frame_ids = [frame.frame_id for frame in schema.frames]
        all_aggregates = build_all_aggregates(
            paths.aggregates_dir,
            document_aggregates_weighted,
            document_aggregates_occurrence,
            frame_ids
        )
    else:
        if not schema:
            print("⚠️ Skipping corpus classification because no schema is available.")
        elif not classifier_model:
            print("⚠️ Skipping corpus classification because the classifier model is missing.")

    # ============================================================================
    # GENERATE REPORT
    # ============================================================================
    
    if schema and paths.html and document_aggregates_weighted:
        # Build classifier lookup for report
        classifier_lookup: Optional[Dict[str, Dict[str, object]]] = None
        if classifier_predictions:
            classifier_lookup = {
                item["passage_id"]: item
                for item in classifier_predictions
                if isinstance(item, dict) and item.get("passage_id")
            }
        
        include_classifier_plots = True if classifier_predictions else False
        
        write_html_report(
            schema=schema,
            assignments=assignments,
            output_path=paths.html,
            classifier_lookup=classifier_lookup,
            classified_documents=len(document_aggregates_weighted),
            classifier_sample_limit=config.classifier_corpus_sample_size,
            include_classifier_plots=include_classifier_plots,
            document_aggregates_weighted=document_aggregates_weighted,
            document_aggregates_occurrence=document_aggregates_occurrence,
            all_aggregates=all_aggregates,
            hide_empty_passages=config.report.hide_empty_passages,
            custom_plots=config.report.custom_plots,
            plot_title=config.report.plot.title,
            plot_subtitle=config.report.plot.subtitle,
            plot_note=config.report.plot.note,
            export_plotly_png_dir=(paths.html.parent / "plots"),
        )
        
        # Publish PNGs and HTML to docs for GitHub Pages
        try:
            _publish_to_docs_assets(paths.html.parent.name, paths.html)
        except Exception as exc:
            print(f"⚠️ Failed to publish docs assets: {exc}")
        
        print(f"\n✅ HTML report written to {paths.html}")
    else:
        if not schema:
            print("⚠️ Cannot generate report: schema is missing.")
        elif not paths.html:
            print("⚠️ Cannot generate report: HTML output path is not configured.")
        elif not document_aggregates_weighted:
            print("⚠️ Cannot generate report: document aggregates are missing.")
    
    # ============================================================================
    # WORKFLOW COMPLETE
    # ============================================================================
    
    print(f"\n✅ Workflow complete. Aggregates computed and saved to {paths.aggregates_dir}")


def apply_overrides(config: NarrativeFramingConfig, args: argparse.Namespace) -> NarrativeFramingConfig:
    if args.results_dir:
        config.results_dir = args.results_dir
    if args.reload_results:
        config.reload_results = True
        config.reload_induction = True
        config.reload_application = True
        config.reload_classifier = True
        config.reload_chunk_classifications = True
    if args.reload_induction:
        config.reload_induction = True
    if args.reload_application:
        config.reload_application = True
    if args.reload_classifier:
        config.reload_classifier = True
    if hasattr(args, 'reload_chunk_classifications') and args.reload_chunk_classifications:
        config.reload_chunk_classifications = True
    if args.train_classifier:
        config.classifier.enabled = True
    if args.induction_model:
        config.induction_model = args.induction_model
    if args.application_model:
        config.application_model = args.application_model
    if args.induction_temperature is not None:
        config.induction_temperature = args.induction_temperature
    if args.application_temperature is not None:
        config.application_temperature = args.application_temperature
    if args.regenerate_report_only:
        config.regenerate_report_only = True
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Narrative framing induction + application workflow")
    default_config = Path(__file__).with_name("config.example.yaml")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help="Path to YAML configuration file",
    )
    parser.add_argument("--results-dir", type=Path, help="Override output directory for results")
    parser.add_argument("--reload-results", action="store_true", help="Reuse cached schema/assignments")
    parser.add_argument("--reload-induction", action="store_true", help="Reuse cached frame schema")
    parser.add_argument("--reload-application", action="store_true", help="Reuse cached LLM frame assignments")
    parser.add_argument("--reload-classifier", action="store_true", help="Reuse cached classifier predictions/model")
    parser.add_argument(
        "--reload-chunk-classifications",
        action="store_true",
        help="Reuse cached per-document chunk classifications",
    )
    parser.add_argument("--skip-application", action="store_true", help="Skip the frame application step")
    parser.add_argument("--train-classifier", action="store_true", help="Force-enable classifier training")
    parser.add_argument("--regenerate-report-only", action="store_true", help="Skip analysis and regenerate report from cached artifacts")
    parser.add_argument("--induction-model", type=str, help="Override the induction model name at runtime")
    parser.add_argument("--application-model", type=str, help="Override the application model name at runtime")
    parser.add_argument("--induction-temperature", type=float, help="Override the induction temperature at runtime")
    parser.add_argument("--application-temperature", type=float, help="Override the application temperature at runtime")
    parser.add_argument("--cleanup-cached-chunks", action="store_true", help="Clean up cached chunks matching exclude regex patterns")
    parser.add_argument("--cleanup-cached-chunks-keywords", action="store_true", help="Clean up cached chunks that don't match filter keywords")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args)
    
    # Handle cleanup options
    if args.cleanup_cached_chunks:
        paths = resolve_result_paths(config.results_dir)
        if paths.chunk_classifications_dir and paths.chunk_classifications_dir.exists():
            if hasattr(config, 'filter_exclude_regex') and config.filter_exclude_regex:
                print("🧹 Cleaning up cached chunks matching exclude regex patterns...")
                removed_count, removed_doc_ids = cleanup_cached_chunks_by_regex(
                    paths.chunk_classifications_dir, 
                    config.filter_exclude_regex,
                    dry_run=False
                )
                if removed_count > 0:
                    print(f"🗑️  Removed {removed_count} cached chunk files matching exclude patterns:")
                    for doc_id in removed_doc_ids[:10]:  # Show first 10
                        print(f"   - {doc_id}")
                    if len(removed_doc_ids) > 10:
                        print(f"   ... and {len(removed_doc_ids) - 10} more")
                else:
                    print("✅ No cached chunks matched exclude patterns")
            else:
                print("⚠️  No filter_exclude_regex patterns configured")
        else:
            print("⚠️  No chunk classifications directory found")
        return
    
    if args.cleanup_cached_chunks_keywords:
        paths = resolve_result_paths(config.results_dir)
        if paths.chunk_classifications_dir and paths.chunk_classifications_dir.exists():
            if hasattr(config, 'filter_keywords') and config.filter_keywords:
                print("🧹 Cleaning up cached chunks that don't match filter keywords...")
                removed_count, removed_doc_ids = cleanup_cached_chunks_by_keywords(
                    paths.chunk_classifications_dir, 
                    config.filter_keywords,
                    dry_run=False
                )
                if removed_count > 0:
                    print(f"🗑️  Removed {removed_count} cached chunk files that don't match keywords:")
                    for doc_id in removed_doc_ids[:10]:  # Show first 10
                        print(f"   - {doc_id}")
                    if len(removed_doc_ids) > 10:
                        print(f"   ... and {len(removed_doc_ids) - 10} more")
                else:
                    print("✅ All cached chunks match filter keywords")
            else:
                print("⚠️  No filter_keywords configured")
        else:
            print("⚠️  No chunk classifications directory found")
        return
    
    run_workflow(config)


if __name__ == "__main__":
    main()
