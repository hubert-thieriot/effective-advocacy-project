#!/usr/bin/env python3
"""Command-line entry point for the narrative framing workflow."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

from apps.narrative_framing.aggregation import (
    DocumentFrameAggregate,
    FrameAggregationStrategy,
    LengthWeightedFrameAggregator,
    build_weighted_time_series,
    compute_global_frame_share,
    time_series_to_records,
)
from apps.narrative_framing.config import ClassifierSettings, NarrativeFramingConfig, load_config
from apps.narrative_framing.plots import image_to_base64, render_frame_area_chart
from apps.narrative_framing.report import write_html_report

from efi_analyser.chunkers.sentence_chunker import SentenceChunker
from efi_analyser.embedders import SentenceTransformerEmbedder
from efi_analyser.frames import Frame, FrameAssignment, FrameInducer, FrameSchema, LLMFrameApplicator
from efi_analyser.frames.classifier import (
    CorpusSampler,
    FrameClassifierSpec,
    FrameClassifierTrainer,
    FrameLabelSet,
    SamplerConfig,
)
from efi_analyser.frames.classifier.model import FrameClassifierModel
from efi_analyser.scorers.openai_interface import OpenAIConfig, OpenAIInterface
from efi_corpus.embedded.embedded_corpus import EmbeddedCorpus

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
    document_aggregates: Optional[Path] = None
    frame_timeseries: Optional[Path] = None
    frame_area_chart: Optional[Path] = None
    html: Optional[Path] = None


@dataclass
class ClassifierRun:
    predictions: List[Dict[str, object]]
    model: Optional[FrameClassifierModel]


def create_chunker(target_words: int, max_chars: int) -> object:
    """Return a chunker compatible with the embedded corpus layout."""
    if TextChunker and TextChunkerConfig:
        try:
            return TextChunker(TextChunkerConfig(max_words=target_words))
        except Exception:
            pass  # Fall back to regex chunker if spaCy model is unavailable.
    if _TEXT_CHUNKER_ERROR:
        print(
            "⚠️ Falling back to regex sentence chunker; spaCy-based chunker unavailable:",
            _TEXT_CHUNKER_ERROR,
        )
    overlap_chars = max(0, min(max_chars - 1, int(max_chars * 0.15)))
    return SentenceChunker(max_chunk_size=max_chars, overlap=overlap_chars)


def create_embedder() -> SentenceTransformerEmbedder:
    """Instantiate a lazy-loading embedder to avoid heavyweight startup."""
    return SentenceTransformerEmbedder(lazy_load=True)


def resolve_result_paths(results_dir: Optional[Path]) -> ResultPaths:
    if not results_dir:
        return ResultPaths()
    results_dir.mkdir(parents=True, exist_ok=True)
    return ResultPaths(
        schema=results_dir / "frame_schema.json",
        assignments=results_dir / "frame_assignments.json",
        classifier_predictions=results_dir / "frame_classifier_predictions.json",
        document_aggregates=results_dir / "frame_document_aggregates.json",
        frame_timeseries=results_dir / "frame_timeseries.json",
        frame_area_chart=results_dir / "frame_area_chart.png",
        html=results_dir / "frame_report.html",
    )


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
                or (item.get("name", "").split()[0] if item.get("name") else item.get("frame_id", ""))
            )[:12],
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


def save_document_aggregates(path: Path, aggregates: Sequence[DocumentFrameAggregate]) -> None:
    payload = [aggregate.to_dict() for aggregate in aggregates]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_document_aggregates(path: Path) -> List[DocumentFrameAggregate]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    aggregates: List[DocumentFrameAggregate] = []
    for item in payload:
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


def save_frame_timeseries(path: Path, records: Sequence[Dict[str, object]]) -> None:
    path.write_text(json.dumps(list(records), indent=2, ensure_ascii=False), encoding="utf-8")


def load_frame_timeseries(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload:
        return pd.DataFrame(columns=["date", "frame_id", "avg_score", "share"])
    df = pd.DataFrame.from_records(payload)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def enrich_assignments_with_links(
    assignments: Sequence[FrameAssignment],
    embedded_corpus: EmbeddedCorpus,
) -> None:
    url_cache: Dict[str, Dict[str, str]] = {}
    for assignment in assignments:
        doc_id = assignment.metadata.get("doc_id") or assignment.passage_id.split(":", 1)[0]
        assignment.metadata["doc_id"] = doc_id

        if doc_id not in url_cache:
            index_entry = embedded_corpus.corpus.get_index_entry(doc_id) or {}
            meta = embedded_corpus.corpus.get_metadata(doc_id)
            fetch_info = embedded_corpus.corpus.get_fetch_info(doc_id)
            merged_meta: Dict[str, str] = {}
            for source in (index_entry, meta, fetch_info):
                if not isinstance(source, dict):
                    continue
                for key, value in source.items():
                    if key not in merged_meta and isinstance(value, str):
                        merged_meta[key] = value

            doc_folder_path = embedded_corpus.corpus.layout.doc_dir(doc_id)
            # Convert to absolute path for proper file:// URLs
            doc_folder_path_abs = doc_folder_path.resolve() if doc_folder_path.exists() else doc_folder_path
            url_cache[doc_id] = {
                "url": merged_meta.get("url", ""),
                "title": merged_meta.get("title", ""),
                "published_at": merged_meta.get("published_at", ""),
                "doc_folder_path": str(doc_folder_path_abs),
            }

        cache_entry = url_cache[doc_id]
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

    spec = FrameClassifierSpec(model_name=settings.model_name, output_dir=str(settings.output_dir))
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    trainer = FrameClassifierTrainer(spec)
    print(f"→ Training classifier model: {settings.model_name}")
    model = trainer.train(label_set)
    print("→ Classifier training complete.")

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

    model.save(settings.output_dir)
    return ClassifierRun(predictions=predictions, model=model)


def classify_corpus_documents(
    model: FrameClassifierModel,
    embedded_corpus: EmbeddedCorpus,
    batch_size: int,
    aggregator: Optional[FrameAggregationStrategy] = None,
    sample_size: Optional[int] = None,
    seed: Optional[int] = None,
) -> Sequence[DocumentFrameAggregate]:
    frame_ids = [frame.frame_id for frame in model.schema.frames]
    active_aggregator = aggregator or LengthWeightedFrameAggregator(frame_ids)

    doc_ids = embedded_corpus.corpus.list_ids()
    if not doc_ids:
        return []

    if sample_size is not None:
        limited = max(0, min(sample_size, len(doc_ids)))
        rng = random.Random(seed)
        rng.shuffle(doc_ids)
        doc_ids = doc_ids[:limited]

    iterator = tqdm(
        doc_ids,
        desc="Classifying documents",
        unit="doc",
        leave=False,
    )

    for idx, doc_id in enumerate(iterator, start=1):
        doc = embedded_corpus.corpus.get_document(doc_id)
        if doc is None:
            continue

        # Extract published_at from metadata if not available on document
        published_at = doc.published_at
        if not published_at:
            metadata = embedded_corpus.corpus.get_metadata(doc_id)
            if metadata and isinstance(metadata, dict):
                published_at = metadata.get('published_at')

        chunks = embedded_corpus.get_chunks(doc_id, materialize_if_necessary=True) or []
        texts: List[str] = []
        chunk_text_pairs: List[Tuple[str, str]] = []
        for chunk in chunks:
            text = (chunk.text or "").strip()
            if not text:
                continue
            chunk_id = f"{doc_id}:chunk{int(chunk.chunk_id):03d}"
            texts.append(text)
            chunk_text_pairs.append((chunk_id, text))

        if not texts:
            continue

        probabilities = model.predict_proba_batch(texts, batch_size=batch_size)
        for (chunk_id, passage_text), probs in zip(chunk_text_pairs, probabilities):
            active_aggregator.accumulate(
                doc_id=doc_id,
                passage_text=passage_text,
                probabilities=probs,
                published_at=published_at,
                title=doc.title,
                url=doc.url,
            )

    iterator.close()

    return active_aggregator.finalize()


def _select_document_highlights(
    aggregates: Sequence[DocumentFrameAggregate],
    schema: FrameSchema,
    limit: int = 10,
) -> List[Dict[str, object]]:
    if not aggregates:
        return []
    frame_labels = {
        frame.frame_id: frame.short_name or (frame.name.split()[0] if frame.name else frame.frame_id)
        for frame in schema.frames
    }
    sorted_aggs = sorted(aggregates, key=lambda agg: float(agg.total_weight), reverse=True)
    highlights: List[Dict[str, object]] = []
    for aggregate in sorted_aggs[: max(limit, 0)]:
        ordered = sorted(aggregate.frame_scores.items(), key=lambda item: item[1], reverse=True)
        top_frames = [
            {
                "frame_id": frame_id,
                "label": frame_labels.get(frame_id, frame_id),
                "score": float(score),
            }
            for frame_id, score in ordered[:3]
        ]
        highlights.append(
            {
                "doc_id": aggregate.doc_id,
                "title": aggregate.title or aggregate.doc_id,
                "url": aggregate.url,
                "published_at": aggregate.published_at,
                "top_frames": top_frames,
            }
        )
    return highlights


def run_workflow(config: NarrativeFramingConfig) -> None:
    # ============================================================================
    # INITIALIZATION AND SETUP
    # ============================================================================
    
    corpus_path = config.corpora_root / config.corpus
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found at {corpus_path}")
    config.workspace_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading embedded corpus from {corpus_path}...")
    chunker = create_chunker(config.target_words, config.max_chars)
    embedder = create_embedder()

    embedded_corpus = EmbeddedCorpus(
        corpus_path=corpus_path,
        workspace_path=config.workspace_root,
        chunker=chunker,
        embedder=embedder,
    )

    print(
        "Using chunker key",
        embedded_corpus.chunker_spec.key(),
        "and embedder",
        embedded_corpus.embedder_spec.model_name,
    )

    sampler = CorpusSampler(embedded_corpus)
    keywords = config.filter_keywords

    paths = resolve_result_paths(config.results_dir)

    # ============================================================================
    # VARIABLE INITIALIZATION
    # ============================================================================
    
    schema: Optional[FrameSchema] = None
    assignments: List[FrameAssignment] = []
    application_samples: List[Tuple[str, str]] = []
    classifier_predictions: List[Dict[str, object]] = []
    classifier_model: Optional[FrameClassifierModel] = None
    document_aggregates: List[DocumentFrameAggregate] = []
    frame_timeseries_df = pd.DataFrame(columns=["date", "frame_id", "avg_score", "share"])
    frame_timeseries_records: List[Dict[str, object]] = []
    frame_area_chart_b64: Optional[str] = None
    global_frame_share: Dict[str, float] = {}
    document_highlights: List[Dict[str, object]] = []

    induction_samples: List[Tuple[str, str]] = []
    induction_reused = False
    assignments_reused = False

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
        ignore_cache=True,
        verbose=True,  # Enable warnings for temperature overrides
    )

    # ============================================================================
    # FRAME INDUCTION (OR RELOAD FROM CACHE)
    # ============================================================================
    
    if config.reload_induction and paths.schema and paths.schema.exists():
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
            )
        )
        print(f"Collected {len(induction_samples)} passages for induction.")
        induction_passages = [text for _, text in induction_samples]
        inducer_client = OpenAIInterface(name="frame_induction", config=openai_config)
        inducer = FrameInducer(
            llm_client=inducer_client,
            domain=config.domain,
            frame_target="between 5 and 10",
            max_passages_per_call=max(20, min(config.induction_sample_size, 80)),
            max_total_passages=config.induction_sample_size * 2,
            frame_guidance=config.induction_guidance,
        )
        schema = inducer.induce(induction_passages)
        if not schema.schema_id:
            schema.schema_id = config.domain.replace(" ", "_")
        print(f"Induction produced {len(schema.frames)} frames.")

        if paths.schema:
            save_schema(paths.schema, schema)

    if schema is None:
        raise RuntimeError("Frame schema is required to proceed. Ensure induction completes successfully.")

    # ============================================================================
    # FRAME APPLICATION (OR RELOAD FROM CACHE)
    # ============================================================================
    
    if config.skip_application:
        application_samples = induction_samples
    else:
        if (
            config.reload_application
            and induction_reused
            and paths.assignments
            and paths.assignments.exists()
        ):
            assignments = load_assignments(paths.assignments)
            assignments_reused = True
            application_samples = [(item.passage_id, item.passage_text) for item in assignments]
            if config.application_sample_size > len(application_samples):
                remaining = config.application_sample_size - len(application_samples)
                exclude_ids = [pid for pid, _ in application_samples]
                if remaining > 0:
                    extra_samples = sampler.collect(
                        SamplerConfig(
                            sample_size=remaining,
                            seed=config.seed + 1,
                            keywords=keywords,
                            exclude_passage_ids=exclude_ids,
                        )
                    )
                    application_samples.extend(extra_samples)
                    if extra_samples:
                        print(
                            f"Sampled {len(extra_samples)} additional passages for classifier evaluation display."
                        )
            print(f"Reloaded {len(assignments)} LLM frame assignments from cache.")
        else:
            exclude_ids = [pid for pid, _ in induction_samples] if induction_samples else []
            print("Sampling passages for frame application...")
            application_samples = sampler.collect(
                SamplerConfig(
                    sample_size=config.application_sample_size,
                    seed=config.seed + 1,
                    keywords=keywords,
                    exclude_passage_ids=exclude_ids or None,
                )
            )
            applicator_client = OpenAIInterface(name="frame_application", config=applicator_config)
            applicator = LLMFrameApplicator(
                llm_client=applicator_client,
                batch_size=config.application_batch_size,
                max_chars_per_passage=config.application_max_chars,
                chunk_overlap_chars=int(config.application_max_chars * 0.1),
            )
            print(f"Applying frames to {len(application_samples)} passages...")
            from tqdm import tqdm
            
            # Create a progress bar for frame application
            with tqdm(total=len(application_samples), desc="Applying frames", unit="passages", 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                # We'll update the progress bar manually by processing in smaller batches
                batch_size = applicator.batch_size
                assignments = []
                
                for i in range(0, len(application_samples), batch_size):
                    batch = application_samples[i:i + batch_size]
                    batch_assignments = applicator.batch_assign(
                        schema,
                        batch,
                        top_k=config.application_top_k,
                    )
                    assignments.extend(batch_assignments)
                    pbar.update(len(batch))
            print(f"Received {len(assignments)} assignments from LLM.")
            if paths.assignments:
                save_assignments(paths.assignments, assignments)

    # ============================================================================
    # POST-PROCESS ASSIGNMENTS AND DISPLAY RESULTS
    # ============================================================================
    
    if assignments:
        enrich_assignments_with_links(assignments, embedded_corpus)
        if paths.assignments:
            save_assignments(paths.assignments, assignments)
        print("Assignments enriched with document metadata.")

    print_schema(schema)

    if assignments:
        preview_assignments(assignments, limit=config.assignment_preview)

    # ============================================================================
    # LOAD CACHED RESULTS (CLASSIFIER, AGGREGATES, TIME SERIES)
    # ============================================================================
    
    if (
        config.classifier.enabled
        and config.reload_classifier
        and paths.classifier_predictions
        and paths.classifier_predictions.exists()
        and assignments_reused
    ):
        classifier_predictions = load_classifier_predictions(paths.classifier_predictions)
        print(f"Reloaded {len(classifier_predictions)} classifier predictions from cache.")

    if config.classifier.enabled and config.reload_classifier and classifier_model is None:
        try:
            classifier_model = FrameClassifierModel.load(config.classifier.output_dir)
            print(f"Reloaded classifier model from {config.classifier.output_dir}.")
        except FileNotFoundError:
            classifier_model = None
        except Exception as exc:  # pragma: no cover - informational only
            print(f"⚠️ Failed to load classifier model from {config.classifier.output_dir}: {exc}")
            classifier_model = None

    # Load cached document aggregates if available
    if config.reload_document_aggregates and paths.document_aggregates and paths.document_aggregates.exists():
        try:
            document_aggregates = load_document_aggregates(paths.document_aggregates)
            print(f"Reloaded {len(document_aggregates)} document aggregates from cache.")
        except Exception as exc:
            print(f"⚠️ Failed to load document aggregates from cache: {exc}")
            document_aggregates = []

    # Load cached time series if available
    if config.reload_time_series and paths.frame_timeseries and paths.frame_timeseries.exists():
        try:
            time_series_data = json.loads(paths.frame_timeseries.read_text(encoding="utf-8"))
            frame_timeseries_records = time_series_data
            # Convert back to DataFrame for processing
            if time_series_data:
                frame_timeseries_df = pd.DataFrame(time_series_data)
                frame_timeseries_df['date'] = pd.to_datetime(frame_timeseries_df['date']).dt.date
                print(f"Reloaded time series with {len(frame_timeseries_df)} records from cache.")
            else:
                frame_timeseries_df = pd.DataFrame(columns=["date", "frame_id", "avg_score", "share"])
        except Exception as exc:
            print(f"⚠️ Failed to load time series from cache: {exc}")
            frame_timeseries_df = pd.DataFrame(columns=["date", "frame_id", "avg_score", "share"])
            frame_timeseries_records = []

    # ============================================================================
    # TRAIN CLASSIFIER (IF NEEDED)
    # ============================================================================
    
    if schema and assignments and config.classifier.enabled:
        need_training = not classifier_predictions or classifier_model is None
        if need_training:
            print("Training classifier on LLM-labeled passages...")
            classifier_run = train_and_apply_classifier(
                settings=config.classifier,
                schema=schema,
                assignments=assignments,
                samples=application_samples
                if application_samples
                else [(a.passage_id, a.passage_text) for a in assignments],
            )
            classifier_predictions = classifier_run.predictions
            classifier_model = classifier_run.model
            if classifier_predictions:
                print(f"Generated classifier predictions for {len(classifier_predictions)} passages.")
            if classifier_predictions and paths.classifier_predictions:
                save_classifier_predictions(paths.classifier_predictions, classifier_predictions)

    # ============================================================================
    # PREPARE DISPLAY ASSIGNMENTS AND CLASSIFY CORPUS
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

    if schema and classifier_model and not document_aggregates:
        target_docs = config.classifier_corpus_sample_size or embedded_corpus.corpus.get_document_count()
        print(f"Applying classifier across corpus sample (target {target_docs} documents)...")
        inference_batch = max(1, config.classifier.inference_batch_size or config.classifier.batch_size)
        document_aggregates = list(
            classify_corpus_documents(
                model=classifier_model,
                embedded_corpus=embedded_corpus,
                batch_size=inference_batch,
                sample_size=config.classifier_corpus_sample_size,
                seed=config.seed,
            )
        )
        if paths.document_aggregates:
            save_document_aggregates(paths.document_aggregates, document_aggregates)
    
    # ============================================================================
    # BUILD TIME SERIES AND VISUALIZATIONS
    # ============================================================================
    
    # Build time series if we have document aggregates but no cached time series
    if document_aggregates and frame_timeseries_df.empty:
        print("Building time series from document aggregates...")
        frame_timeseries_df = build_weighted_time_series(document_aggregates)
        frame_timeseries_records = time_series_to_records(frame_timeseries_df)
        if paths.frame_timeseries:
            save_frame_timeseries(paths.frame_timeseries, frame_timeseries_records)
    
    # Set up frame order and labels for visualization
    if schema:
        frame_order = [frame.frame_id for frame in schema.frames]
        frame_labels = {
            frame.frame_id: frame.short_name or (frame.name.split()[0] if frame.name else frame.frame_id)
            for frame in schema.frames
        }
        if paths.frame_area_chart:
            result_path = render_frame_area_chart(
                frame_timeseries_df,
                frame_order=frame_order,
                frame_labels=frame_labels,
                output_path=paths.frame_area_chart,
                rolling_window=30,  # 30-day running average
            )
            if result_path:
                frame_area_chart_b64 = image_to_base64(result_path)
        global_frame_share = compute_global_frame_share(document_aggregates)
        document_highlights = _select_document_highlights(document_aggregates, schema)
        classified_docs = len(document_aggregates)
        sampled_suffix = (
            f" (sample of {config.classifier_corpus_sample_size})"
            if config.classifier_corpus_sample_size
            else ""
        )
        print(f"Classifier applied to {classified_docs} documents{sampled_suffix}.")
    else:
        if not schema:
            print("⚠️ Skipping corpus classification because no schema is available.")
        elif not classifier_model:
            print("⚠️ Skipping corpus classification because the classifier model is missing.")

    # ============================================================================
    # GENERATE FINAL HTML REPORT
    # ============================================================================
    
    if paths.html and schema:
        classifier_lookup = {item["passage_id"]: item for item in classifier_predictions} if classifier_predictions else None
        write_html_report(
            schema=schema,
            assignments=display_assignments,
            output_path=paths.html,
            classifier_lookup=classifier_lookup,
            global_frame_share=global_frame_share,
            timeseries_records=frame_timeseries_records,
            document_highlights=document_highlights,
            classified_documents=len(document_aggregates),
            classifier_sample_limit=config.classifier_corpus_sample_size,
            area_chart_b64=frame_area_chart_b64,
            include_classifier_plots=True,
        )
        print(f"\nHTML report written to {paths.html}")


def apply_overrides(config: NarrativeFramingConfig, args: argparse.Namespace) -> NarrativeFramingConfig:
    if args.results_dir:
        config.results_dir = args.results_dir
    if args.reload_results:
        config.reload_results = True
        config.reload_induction = True
        config.reload_application = True
        config.reload_classifier = True
        config.reload_document_aggregates = True
        config.reload_time_series = True
    if args.reload_induction:
        config.reload_induction = True
    if args.reload_application:
        config.reload_application = True
    if args.reload_classifier:
        config.reload_classifier = True
    if hasattr(args, 'reload_document_aggregates') and args.reload_document_aggregates:
        config.reload_document_aggregates = True
    if hasattr(args, 'reload_time_series') and args.reload_time_series:
        config.reload_time_series = True
    if args.skip_application:
        config.skip_application = True
    if args.train_classifier:
        config.classifier.enabled = True
    if args.assignment_preview is not None:
        config.assignment_preview = args.assignment_preview
    if args.model:
        config.llm_model = args.model
        # For backward compatibility, set both models to the same value if only --model is specified
        if not args.induction_model:
            config.induction_model = args.model
        if not args.application_model:
            config.application_model = args.model
    if args.induction_model:
        config.induction_model = args.induction_model
    if args.application_model:
        config.application_model = args.application_model
    if args.induction_temperature is not None:
        config.induction_temperature = args.induction_temperature
    if args.application_temperature is not None:
        config.application_temperature = args.application_temperature
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
    parser.add_argument("--reload-document-aggregates", action="store_true", help="Reuse cached document aggregates")
    parser.add_argument("--reload-time-series", action="store_true", help="Reuse cached time series data")
    parser.add_argument("--skip-application", action="store_true", help="Skip the frame application step")
    parser.add_argument("--train-classifier", action="store_true", help="Force-enable classifier training")
    parser.add_argument("--assignment-preview", type=int, help="Number of assignments to preview in stdout")
    parser.add_argument("--model", type=str, help="Override the LLM model name at runtime (deprecated - use --induction-model and --application-model)")
    parser.add_argument("--induction-model", type=str, help="Override the induction model name at runtime")
    parser.add_argument("--application-model", type=str, help="Override the application model name at runtime")
    parser.add_argument("--induction-temperature", type=float, help="Override the induction temperature at runtime")
    parser.add_argument("--application-temperature", type=float, help="Override the application temperature at runtime")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args)
    run_workflow(config)


if __name__ == "__main__":
    main()
