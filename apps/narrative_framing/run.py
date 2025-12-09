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

from apps.narrative_framing.aggregation_document import DocumentFrameAggregate
from apps.narrative_framing.aggregation_temporal import TemporalAggregator, period_aggregates_to_records
from apps.narrative_framing.aggregation_domain import DomainAggregator
from apps.narrative_framing.aggregates import Aggregates, AggregatesBuilder
from apps.narrative_framing.config import ClassifierSettings, NarrativeFramingConfig, load_config
from apps.narrative_framing.report import ReportBuilder
from apps.narrative_framing.filtering import Filter


from efi_analyser.chunkers.sentence_chunker import SentenceChunker
from efi_analyser.frames import (
    EmbeddedCorpora,
    Frame,
    FrameAssignment,
    FrameAssignments,
    FrameInducer,
    FrameSchema,
    LLMFrameAnnotator,
)
from efi_analyser.frames.classifier import (
    DocumentClassifications,
    EmbeddedCorporaSampler,
    FrameClassifier,
    FrameClassifierModel,
    FrameClassifierTrainer,
    FrameClassifierArtifacts,
    FrameLabelSet,
    SamplerConfig,
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
        classifications_dir=results_dir / "classifications",
        frame_timeseries=results_dir / "frame_timeseries.json",
        html=results_dir / "frame_report.html",
    )


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


class NarrativeFramingWorkflow:
    """Object-oriented orchestrator for the narrative framing workflow."""

    def __init__(self, config: NarrativeFramingConfig) -> None:
        self.config = config
        self.corpus_names: List[str] = list(config.iter_corpus_names())
        if not self.corpus_names:
            raise ValueError("At least one corpus must be configured.")

        missing_corpora = [name for name in self.corpus_names if not (config.corpora_root / name).exists()]
        if missing_corpora:
            raise FileNotFoundError(
                f"Corpus paths not found: {', '.join(str(config.corpora_root / name) for name in missing_corpora)}"
            )

        # Workspace and result paths
        config.workspace_root.mkdir(parents=True, exist_ok=True)
        self.paths = resolve_result_paths(config.results_dir)

        # Shared state and context
        self.state = WorkflowState()
        self.corpora_map: EmbeddedCorpora
        self.sampler: EmbeddedCorporaSampler
        # Use document-level keywords for document selection
        self.keywords: Optional[List[str]] = config.filter.document.keywords
        # Use chunk-level filters for chunk filtering
        self.filter = Filter(
            exclude_regex=config.filter.chunk.exclude_regex,
            exclude_min_hits=config.filter.chunk.exclude_min_hits,
            trim_after_markers=config.filter.chunk.trim_after_markers,
            keywords=config.filter.chunk.keywords,
        )

        self.openai_induction_config: OpenAIConfig
        self.openai_application_config: OpenAIConfig
        self.ind_sys_t: str
        self.ind_usr_t: str
        self.app_sys_t: str
        self.app_usr_t: str

        # Per-step controls: whether we may create new work.
        # regenerate_report_only acts as a global read-only override.
        read_only = bool(config.regenerate_report_only)
        self.allow_new_induction: bool = bool(config.allow_new_induction) and not read_only
        self.allow_new_annotation: bool = bool(config.allow_new_annotation) and not read_only
        self.allow_new_training: bool = bool(config.allow_new_training) and not read_only
        self.allow_new_classification: bool = bool(config.allow_new_classification) and not read_only
        self.allow_new_aggregation: bool = bool(config.allow_new_aggregation) and not read_only

        # Document universe used later in reporting
        self.total_doc_ids: List[str] = []

        self._snapshot_config()
        self._prepare_corpora()
        self._prepare_llm_clients_and_prompts()

    # --------------------------------------------------------------------- #
    # Environment preparation
    # --------------------------------------------------------------------- #
    def _snapshot_config(self) -> None:
        """Persist a copy of the run configuration under results/configs."""
        config = self.config
        try:
            base_results_dir = config.results_dir or Path("results/narrative_framing")
            cfg_dir = base_results_dir / "configs"
            cfg_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
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
                    (cfg_dir / dst_name).write_text(
                        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                        encoding="utf-8",
                    )
                except Exception as exc:
                    print(f"⚠️ Failed to dump config to YAML: {exc}")
        except Exception as exc:
            print(f"⚠️ Failed to snapshot configuration: {exc}")

    def _prepare_corpora(self) -> None:
        """Load embedded corpora, chunker, embedder, and sampler."""
        config = self.config
        if len(self.corpus_names) == 1:
            singular_path = config.corpora_root / self.corpus_names[0]
            print(f"Loading embedded corpus from {singular_path}...")
        else:
            joined = ", ".join(self.corpus_names)
            print(f"Loading embedded corpora: {joined}")

        if TextChunker is not None:
            try:
                chunker = TextChunker(
                    TextChunkerConfig(
                        max_words=config.chunking.target_words,
                        spacy_model=config.chunking.chunker_model,
                    )
                )
            except Exception as exc:
                print(
                    "⚠️ Falling back to sentence chunker because TextChunker initialization failed:",
                    exc,
                )
                chunker = SentenceChunker()
        else:
            chunker = SentenceChunker()

        # No embedder needed for this workflow, as it is managed by the classification model itself
        embedder = None
        corpora: Dict[str, EmbeddedCorpus] = {}
        for name in self.corpus_names:
            corpora[name] = EmbeddedCorpus(
                corpus_path=config.corpora_root / name,
                workspace_path=config.workspace_root,
                chunker=chunker,
                embedder=embedder,
            )
        self.corpora_map = EmbeddedCorpora(corpora)
        self.sampler = EmbeddedCorporaSampler(self.corpora_map, policy=config.sampling_policy)

    def _prepare_llm_clients_and_prompts(self) -> None:
        """Configure OpenAI clients and resolve prompt templates."""
        config = self.config

        induction_temp = (
            config.induction.temperature
            if config.induction.temperature is not None
            else OpenAIConfig.get_default_temperature(config.induction.model)
        )
        application_temp = (
            config.annotation.temperature
            if config.annotation.temperature is not None
            else OpenAIConfig.get_default_temperature(config.annotation.model)
        )

        self.openai_induction_config = OpenAIConfig(
            model=config.induction.model,
            temperature=induction_temp,
            timeout=600.0,
            ignore_cache=False,
            verbose=True,  # Enable warnings for temperature overrides
        )
        self.openai_application_config = OpenAIConfig(
            model=config.annotation.model,
            temperature=application_temp,
            timeout=600.0,
            ignore_cache=False,
            verbose=config.annotation.verbose if config.annotation.verbose is not None else False,
        )

        # Resolve and validate default prompt templates; copy raw templates into results
        prompt_paths = _resolve_default_prompt_paths()
        _copy_templates_to_results(prompt_paths, self.config.results_dir or Path("results/narrative_framing"))

        self.ind_sys_t = _read_text_or_fail(prompt_paths["induction_system"])
        self.ind_usr_t = _read_text_or_fail(prompt_paths["induction_user"])
        self.app_sys_t = _read_text_or_fail(prompt_paths["application_system"])
        self.app_usr_t = _read_text_or_fail(prompt_paths["application_user"])

    # --------------------------------------------------------------------- #
    # Helper methods
    # --------------------------------------------------------------------- #
    @staticmethod
    def _flatten_keywords(
        keywords: Optional[Union[List[str], Dict[str, List[str]]]]
    ) -> Optional[List[str]]:
        """Flatten language-specific keywords dict to a single list for sampling.
        
        The sampler uses substring matching, so we dedupe and return all keywords
        across all languages as a flat list.
        """
        if keywords is None:
            return None
        if isinstance(keywords, list):
            return keywords
        # Dict: flatten all language lists into one unique set
        all_kw = set()
        for kw_list in keywords.values():
            all_kw.update(kw_list)
        return list(all_kw) if all_kw else None

    # --------------------------------------------------------------------- #
    # Public orchestration entrypoint
    # --------------------------------------------------------------------- #
    def run(self) -> None:
        """Execute the full workflow: induction → annotation → train → apply → aggregate → report → plots."""
        self.run_induction()
        self.run_annotation()
        self.run_training()
        self.run_application()
        self.run_aggregation()
        self.run_report()
        self.run_plots()

    # --------------------------------------------------------------------- #
    # Step 1 – Induction
    # --------------------------------------------------------------------- #
    def run_induction(self) -> None:
        config = self.config
        paths = self.paths
        state = self.state

        # Reload from cache when requested, or in regenerate mode
        should_reload = config.reload_induction or config.regenerate_report_only
        if should_reload and paths.schema and paths.schema.exists():
            state.schema = load_schema(paths.schema)
            state.induction_reused = True
            print(f"Reloaded frame schema from {paths.schema}")
        elif should_reload and self.allow_new_induction:
            # Wanted to reload but cache is missing; fall back to running induction.
            print("⚠️ Cached schema not found; running induction instead.")

        # If schema is still missing here, either we never tried to reload
        # (reload_induction is False) or reload failed without allow_new_induction.
        if state.schema is None and not self.allow_new_induction:
            raise FileNotFoundError(
                "Frame schema not found in cache and running induction is disabled in read-only mode."
            )

        if state.schema is None and self.allow_new_induction:
            print("Collecting passages for frame induction...")
            # Use relevance_keywords to filter induction sample (flatten if dict)
            induction_keywords = self._flatten_keywords(config.relevance_keywords)
            if induction_keywords:
                print(f"   Filtering induction sample to passages with {len(induction_keywords)} relevance keywords")
            
            # Get base sampler kwargs and override keywords with relevance_keywords if provided
            sampler_kwargs = self.filter.sampler_kwargs(domain_whitelist=config.filter.document.domain_whitelist)
            if induction_keywords:
                sampler_kwargs["keywords"] = induction_keywords
            
            induction_samples = self.sampler.collect_chunks(
                SamplerConfig(
                    sample_size=config.induction.size,
                    seed=config.seed,
                    date_from=config.filter.date_from,
                    require_document_keywords=config.filter.document.keywords,
                    **sampler_kwargs,
                )
            )
            print(f"Collected {len(induction_samples)} passages for induction.")
            induction_passages = [text for _, text in induction_samples]
            inducer_client = OpenAIInterface(name="frame_induction", config=self.openai_induction_config)
            # Determine max passages per induction call
            if config.induction.batch_size:
                max_per_call = config.induction.batch_size
            else:
                max_per_call = min(config.induction.size, 40)
            
            inducer = FrameInducer(
                llm_client=inducer_client,
                domain=config.domain,
                frame_target=config.induction.frame_target,
                max_passages_per_call=max_per_call,
                max_total_passages=config.induction.size * 2,
                induction_guidance=config.induction.guidance,
                system_template=self.ind_sys_t,
                user_template=self.ind_usr_t,
            )
            schema = inducer.induce(induction_passages)
            if not schema.schema_id:
                schema.schema_id = config.domain.replace(" ", "_")
            print(f"Induction produced {len(schema.frames)} frames.")
            state.schema = schema
            state.induction_samples = induction_samples

            if paths.schema:
                save_schema(paths.schema, schema)
            # Save resolved induction prompts (all calls) under results/prompts/induction/resolved
            if inducer.emitted_messages:
                resolved_dir = (config.results_dir or Path("results")) / "prompts" / "induction" / "resolved"
                _save_resolved_messages(resolved_dir, "induction_call", inducer.emitted_messages)

        if state.schema is None:
            raise RuntimeError("Frame schema is required to proceed. Ensure induction completes successfully.")

    # --------------------------------------------------------------------- #
    # Step 2 – Annotation / LLM application
    # --------------------------------------------------------------------- #
    def run_annotation(self) -> None:
        config = self.config
        paths = self.paths
        sampler = self.sampler
        state = self.state
        schema = state.schema

        if schema is None:
            return

        # Full cached set vs this-run subset
        assignments: FrameAssignments = FrameAssignments()

        induction_samples = state.induction_samples
        target_application_count = max(0, config.annotation.size)

        # 1) Reload cached assignments from disk if requested, or in regenerate mode
        should_reload = config.reload_annotation or config.regenerate_report_only
        if should_reload and paths.assignments and paths.assignments.exists():
            try:
                assignments = FrameAssignments.load(paths.assignments)
                if assignments:
                    print(f"Reloaded {assignments.count} LLM frame assignments from cache.")
            except Exception as exc:
                print(f"⚠️ Failed to load cached assignments: {exc}")

        # 2) Add new assignments if allowed and needed
        if self.allow_new_annotation:
            existing_ids = {a.passage_id for a in assignments}
            existing_count = assignments.count
            additional_needed = max(0, target_application_count - existing_count)

            # Top up to target_application_count using new samples
            if additional_needed > 0:
                exclude_ids: List[str] = list(existing_ids)
                if induction_samples:
                    exclude_ids.extend(pid for pid, _ in induction_samples)
                try:
                    new_candidates = sampler.collect_candidates(
                        SamplerConfig(
                            sample_size=additional_needed,
                            seed=config.seed + 1,
                            exclude_passage_ids=exclude_ids or None,
                            date_from=config.filter.date_from,
                            require_document_keywords=config.filter.document.keywords,
                            **self.filter.sampler_kwargs(domain_whitelist=config.filter.document.domain_whitelist),
                        )
                    )
                except ValueError as exc:
                    print(f"⚠️ Unable to gather {additional_needed} new passages for application: {exc}")
                    new_candidates = []

                if new_candidates:
                    annotator_client = OpenAIInterface(
                        name="frame_annotation", config=self.openai_application_config
                    )
                    resolved_dir = (config.results_dir or Path("results")) / "prompts" / "frame_annotation" / "resolved"
                    annotator = LLMFrameAnnotator(
                        llm_client=annotator_client,
                        batch_size=config.annotation.batch_size,
                        max_chars_per_passage=None,
                        chunk_overlap_chars=0,
                        system_template=self.app_sys_t,
                        user_template=self.app_usr_t,
                        resolved_messages_dir=resolved_dir,
                        resolved_messages_prefix="frame_annotation_batch",
                    )
                    new_assignments = annotator.batch_assign(
                        schema,
                        new_candidates,
                        top_k=config.annotation.top_k,
                        show_progress=True,
                        progress_desc=f"Annotating frames ({annotator_client.config.model})",
                        relevance_keywords=config.annotation.force_zero_if_no_keywords or config.relevance_keywords,
                        guidance=config.annotation.guidance,
                    )
                    assignments.extend(new_assignments)
                    print(
                        f"Received {len(assignments) - existing_count} new assignments; total cached assignments: {len(assignments)}."
                    )
                    if paths.assignments:
                        assignments.save(paths.assignments)
                else:
                    print(
                        f"⚠️ No new passages were sampled; continuing with {assignments.count} cached assignments."
                    )

        # 3) Limit to requested application_sample_size when we have more cached assignments
        if target_application_count and assignments.count > target_application_count:
            print(
                f"Limiting cached assignments from {assignments.count} to a shuffled subset of {target_application_count} for this run."
            )
            assignments = assignments.select_random(target_application_count, config.seed + 97)

        # Persist state
        state.assignments = assignments

    # --------------------------------------------------------------------- #
    # Step 3 – Train classifier (optional)
    # --------------------------------------------------------------------- #
    def run_training(self) -> None:
        config = self.config
        paths = self.paths
        state = self.state

        schema = state.schema
        assignments = state.assignments
        settings: ClassifierSettings = config.classifier

        # Short-circuit if classifier is disabled or we have nothing to train on
        if not settings.enabled:
            return
        if not schema or not assignments:
            print("⚠️ Skipping classifier: no schema or assignments available.")
            return

        # Attempt to reload classifier artifacts when reload_classifier is True or in regenerate mode
        should_reload = config.reload_classifier or config.regenerate_report_only
        print(f"🔍 Classifier reload check: reload_classifier={config.reload_classifier}, regenerate_report_only={config.regenerate_report_only}, should_reload={should_reload}")
        print(f"🔍 Paths: classifier_dir={paths.classifier_dir}, classifier_predictions={paths.classifier_predictions}")
        
        # Check if required files exist for reloading
        classifier_dir_exists = paths.classifier_dir and paths.classifier_dir.exists()
        classifier_json_exists = classifier_dir_exists and (paths.classifier_dir / "frame_classifier.json").exists()
        predictions_exists = paths.classifier_predictions and paths.classifier_predictions.exists()
        
        if paths.classifier_dir:
            print(f"🔍 classifier_dir.exists()={paths.classifier_dir.exists()}")
            if classifier_dir_exists:
                print(f"🔍 frame_classifier.json.exists()={classifier_json_exists}")
        if paths.classifier_predictions:
            print(f"🔍 classifier_predictions.exists()={predictions_exists}")
        
        if (
            should_reload
            and classifier_dir_exists
            and classifier_json_exists
            and predictions_exists
        ):
            try:
                print(f"🔄 Attempting to reload classifier from {paths.classifier_dir}...")
                state.classifier_model = FrameClassifierModel.load(paths.classifier_dir)
                artifacts = FrameClassifierArtifacts.load_predictions(paths.classifier_predictions)
                state.classifier_predictions = artifacts.predictions
                print(f"✅ Reloaded classifier model and predictions from {paths.classifier_dir}.")
                return
            except Exception as exc:
                print(f"⚠️ Failed to load classifier artifacts from {paths.classifier_dir}: {exc}")
                # If reload fails and training is disabled, we can't proceed
                if not self.allow_new_training:
                    raise RuntimeError(
                        f"Cannot reload classifier and training is disabled. "
                        f"Fix the classifier artifacts or set allow_new_training: true"
                    ) from exc
        elif should_reload:
            print(f"⚠️ reload_classifier is True but required files are missing:")
            if not paths.classifier_dir:
                print(f"  - classifier_dir is None")
            elif not paths.classifier_dir.exists():
                print(f"  - classifier_dir does not exist: {paths.classifier_dir}")
            elif not classifier_json_exists:
                print(f"  - frame_classifier.json does not exist in {paths.classifier_dir}")
            if not paths.classifier_predictions:
                print(f"  - classifier_predictions is None")
            elif not paths.classifier_predictions.exists():
                print(f"  - classifier_predictions does not exist: {paths.classifier_predictions}")

        if not self.allow_new_training:
            return

        # Prepare output directory for classifier artifacts
        classifier_output_dir: Optional[Path] = None
        if paths.classifier_dir:
            paths.classifier_dir.mkdir(parents=True, exist_ok=True)
            classifier_output_dir = paths.classifier_dir

        # Configure classifier spec, reusing underlying FrameClassifierSpec via settings
        if len(self.corpus_names) == 1:
            run_name = f"{self.corpus_names[0]}-frame-cls"
        else:
            run_name = f"{'+'.join(self.corpus_names)}-frame-cls"

        settings.run_name = run_name
        if classifier_output_dir is not None:
            settings.output_dir = str(classifier_output_dir)

        # Build label set from LLM assignments
        label_set = FrameLabelSet.from_assignments(schema, assignments, source="llm")
        if not label_set.passages:
            print("⚠️ Label set is empty; skipping classifier training.")
            return

        trainer = FrameClassifierTrainer(settings)

        # Train, optionally cross-validate, and run inference in one call
        artifacts = trainer.run(
            label_set=label_set,
            assignments=assignments,
            cv_folds=settings.cv_folds,
        )
        model = artifacts.model
        predictions = artifacts.predictions

        # Persist artifacts
        if classifier_output_dir is not None:
            if model is not None:
                model.save(classifier_output_dir)
        if predictions and paths.classifier_predictions:
            artifacts.save_predictions(paths.classifier_predictions)

        state.classifier_model = model
        state.classifier_predictions = predictions


    # --------------------------------------------------------------------- #
    # Step 4 – Apply classifier to corpus
    # --------------------------------------------------------------------- #
    def run_application(self) -> None:
        config = self.config
        paths = self.paths
        state = self.state

        schema = state.schema
        classifier_model = state.classifier_model

        # Build document universe
        total_doc_ids = self.corpora_map.list_global_doc_ids_from(config.filter.date_from)
        self.total_doc_ids = list(total_doc_ids)

        desired_doc_total = config.classification.size or len(total_doc_ids)
        desired_doc_total = max(0, min(desired_doc_total, len(total_doc_ids)))

        # Load existing classifications from disk when available, or in regenerate mode
        classifications = DocumentClassifications()
        should_reload = config.reload_classifications or config.regenerate_report_only
        if should_reload and paths.classifications_dir and paths.classifications_dir.exists():
            classifications = DocumentClassifications.from_folder(paths.classifications_dir)
            if classifications.n_docs > 0:
                print(
                    f"✅ Reloaded {classifications.n_docs} classifications for {classifications.n_chunks} chunks from cache."
                )

        if self.allow_new_classification and schema and classifier_model:
            # Classify additional documents if needed
            if desired_doc_total > classifications.n_docs:
                remaining_needed = desired_doc_total - classifications.n_docs
                if remaining_needed > 0:
                    doc_ids_to_classify = self.sampler.collect_docs(
                        sample_size=remaining_needed,
                        seed=config.seed,
                        date_from=config.filter.date_from,
                        exclude_doc_ids=[doc.doc_id for doc in classifications if doc.doc_id],
                        require_keywords=config.filter.document.keywords,
                        domain_whitelist=config.filter.document.domain_whitelist,
                    )
                    if doc_ids_to_classify:
                        print(
                            f"Classifying {len(doc_ids_to_classify)} additional documents "
                            f"to reach target sample of {desired_doc_total}."
                        )
                        batch_size = max(1, config.classifier.batch_size)
                        classifier = FrameClassifier(
                            model=classifier_model,
                            corpora=self.corpora_map,
                            batch_size=batch_size,
                            seed=config.seed,
                            require_keywords=config.filter.document.keywords,
                            exclude_regex=config.filter.chunk.exclude_regex,
                            exclude_min_hits=config.filter.chunk.exclude_min_hits,
                            trim_after_markers=config.filter.chunk.trim_after_markers,
                        )
                        new_classifications = classifier.run(
                            sample_size=None,
                            doc_ids=doc_ids_to_classify,
                            output_dir=paths.classifications_dir,
                        )
                        classifications.extend(new_classifications)

                if desired_doc_total > classifications.n_docs:
                    print(
                        f"⚠️ Only {classifications.n_docs} documents classified; target was {desired_doc_total}."
                    )

        # Persist all classifications in state; downstream steps can apply their
        # own sampling or filtering as needed.
        state.classifications = classifications

    # --------------------------------------------------------------------- #
    # Step 5 – Aggregation (documents → time / domain / corpus / global)
    # --------------------------------------------------------------------- #
    def run_aggregation(self) -> None:
        config = self.config
        paths = self.paths
        state = self.state

        schema = state.schema
        classifications = state.classifications

        if not paths.aggregates_dir:
            print("⚠️ Skipping aggregation: aggregates directory is not configured.")
            return

        aggregates_dir = paths.aggregates_dir
        aggregates: Optional[Aggregates] = None

        # Try to load aggregates from cache if reload is requested, or in regenerate mode
        should_reload = config.reload_aggregates or config.regenerate_report_only
        if should_reload:
            aggregates = Aggregates.load(aggregates_dir)
            if aggregates:
                        mode_msg = " (regenerate mode)" if config.regenerate_report_only else ""
                        print(
                    f"✅ Reloaded {len(aggregates.documents_weighted)} weighted document aggregates "
                    f"and {len(aggregates.documents_occurrence)} occurrence aggregates from cache{mode_msg}."
                )

        # Build new aggregates when not loaded or when allowed and needed
        if aggregates is None:
            if not self.allow_new_aggregation:
                print("⚠️ Skipping aggregation: cannot rebuild (read-only mode) and cache load failed.")
                return
            
            # When classifier is disabled, use LLM annotations for aggregation
            if not classifications and state.assignments:
                print("ℹ️ Classifier disabled; using LLM annotations for aggregation.")
                classifications = state.assignments.to_classifications()
                
            if not schema or not classifications:
                print("⚠️ Skipping aggregation: schema or classifications missing.")
                return
            
            filter_spec = self.filter.to_spec()
            builder = AggregatesBuilder(
                aggregates_dir=aggregates_dir,
                frame_ids=[frame.frame_id for frame in schema.frames],
                corpus_names=self.corpus_names,
                application_top_k=config.annotation.top_k,
                min_threshold_weighted=config.agg_min_threshold_weighted,
                normalize_weighted=config.agg_normalize_weighted,
                min_threshold_occurrence=config.agg_min_threshold_occurrence,
                    filter_spec=filter_spec,
                )
            aggregates = builder.build(classifications)

        state.aggregates = aggregates

    # --------------------------------------------------------------------- #
    # Step 6 – Generate report
    # --------------------------------------------------------------------- #
    def run_report(self) -> None:
        config = self.config
        paths = self.paths
        state = self.state

        # Generate report using ReportBuilder
        report_builder = ReportBuilder(
            state=state,
            config=config,
            paths=paths,
            total_doc_ids=self.total_doc_ids,
            corpora_map=self.corpora_map,
        )
        report_builder.build()

    # --------------------------------------------------------------------- #
    # Step 7 – Additional Plots
    # --------------------------------------------------------------------- #
    def run_plots(self) -> None:
        """Generate additional plots configured via additional_plots."""
        config = self.config
        state = self.state
        
        if not config.additional_plots:
            return
        
        print(f"\n📊 Generating {len(config.additional_plots)} additional plots...")
        
        # Convert config dicts to PlotConfig objects
        plot_configs = []
        for plot_dict in config.additional_plots:
            plot_configs.append(PlotConfig(
                type=plot_dict.get("type"),
                output=plot_dict.get("output"),
                options=plot_dict.get("options"),
                export_as=plot_dict.get("export_as"),
            ))
        
        # Run plots
        results_dir = config.results_dir or Path("results")
        generated = run_plots(
            state=state,
            config=config,
            results_dir=results_dir,
            plot_configs=plot_configs,
            corpus_index=self._build_corpus_index(),
            export_dir=config.export_dir,
            export_plots_dir=config.report.export_plots_dir,
        )
        
        if generated:
            print(f"✅ Generated {len(generated)} plots")
        else:
            print("⚠️ No plots generated")
    
    def _build_corpus_index(self) -> Dict[str, dict]:
        """Build corpus index from loaded corpora.
        
        Returns dict mapping doc_id to metadata for all documents.
        """
        index = {}
        for doc_id in self.total_doc_ids:
            # Parse global doc_id to get corpus name and local id
            # Format is typically "corpus_name::local_id" 
            if "::" in doc_id:
                corpus_name, local_id = doc_id.split("::", 1)
            else:
                # Single corpus - use first corpus
                corpus_name = self.corpus_names[0] if self.corpus_names else None
                local_id = doc_id
            
            if corpus_name and corpus_name in self.corpora_map:
                corpus = self.corpora_map[corpus_name].corpus
                meta = corpus.get_metadata(local_id) or {}
                index[doc_id] = meta
            else:
                index[doc_id] = {}
        
        return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Narrative framing induction + application workflow")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML configuration file",
        required=True
    )
    return parser.parse_args()


def run_workflow(config: NarrativeFramingConfig) -> None:
    """Backward-compatible wrapper to run the narrative framing workflow."""
    workflow = NarrativeFramingWorkflow(config)
    workflow.run()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    workflow = NarrativeFramingWorkflow(config)
    workflow.run()


if __name__ == "__main__":
    main()
