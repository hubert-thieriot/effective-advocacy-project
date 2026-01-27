"""High-level orchestration for frame induction, annotation, training, and classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import json
import random
import logging

from efi_analyser.frames.induction import FrameInducer
from efi_analyser.frames.types import FrameSchema, FrameAssignments, Candidate
from efi_analyser.frames.annotator import LLMFrameAnnotator
from efi_analyser.frames.classifier import (
    FrameClassifierTrainer,
    FrameClassifierModel,
    FrameClassifier,
    FrameLabelSet,
    SamplerConfig,
    DocumentClassifications,
    FrameClassifierArtifacts,
)
from efi_analyser.frames.corpora import EmbeddedCorpora
from efi_analyser.frames.classifier import EmbeddedCorporaSampler
from efi_analyser.frames.identifiers import split_passage_id, split_global_doc_id
from efi_analyser.frames.filter import DocumentFilter
from efi_analyser.scorers.openai_interface import OpenAIInterface, OpenAIConfig


@dataclass
class FramerPaths:
    """Filesystem paths for framing artifacts."""

    results_dir: Path
    schema_path: Path
    assignments_path: Path
    classifier_dir: Path
    classifications_dir: Path


@dataclass
class FramingResult:
    """Result bundle from framing orchestration."""

    schema: FrameSchema
    assignments: FrameAssignments
    classifications: DocumentClassifications
    induction_samples: List[Tuple[str, str]]
    annotation_candidates: List[Candidate]


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
    from efi_analyser.frames.types import Frame

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


class Framer:
    """Orchestrate frame induction, annotation, training, and classification."""

    def __init__(
        self,
        config: object,
        paths: FramerPaths,
        *,
        prompts: Optional[Dict[str, str]] = None,
    ) -> None:
        self.config = config
        self.paths = paths
        self.prompts = prompts or {}
        self._induction_samples: Optional[List[Tuple[str, str]]] = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(
        self,
        *,
        corpora: EmbeddedCorpora,
        sampler: EmbeddedCorporaSampler,
        allow_new_work: bool = True,
        filter_kwargs: Optional[Dict[str, object]] = None,
        relevance_keywords: Optional[Sequence[str]] = None,
        doc_keywords: Optional[Sequence[str]] = None,
        domain_whitelist: Optional[Sequence[str]] = None,
        date_from: Optional[str] = None,
        date_windows: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> FramingResult:
        schema = self._get_or_create_schema(
            sampler,
            allow_new_work=allow_new_work,
            filter_kwargs=filter_kwargs,
            relevance_keywords=relevance_keywords,
            doc_keywords=doc_keywords,
            domain_whitelist=domain_whitelist,
            date_from=date_from,
            date_windows=date_windows,
        )

        assignments, annotation_candidates = self._annotate(
            sampler,
            corpora,
            schema,
            allow_new_work=allow_new_work,
            filter_kwargs=filter_kwargs,
            relevance_keywords=relevance_keywords,
            doc_keywords=doc_keywords,
            domain_whitelist=domain_whitelist,
            date_from=date_from,
            date_windows=date_windows,
        )

        model = self._train(schema, assignments, allow_new_work=allow_new_work)

        classifications = self._classify(
            corpora,
            sampler,
            schema,
            assignments,
            model,
            allow_new_work=allow_new_work,
            doc_keywords=doc_keywords,
            domain_whitelist=domain_whitelist,
            date_from=date_from,
            date_windows=date_windows,
        )

        return FramingResult(
            schema=schema,
            assignments=assignments,
            classifications=classifications,
            induction_samples=list(self._induction_samples or []),
            annotation_candidates=list(annotation_candidates),
        )

    def _get_or_create_schema(
        self,
        sampler: EmbeddedCorporaSampler,
        *,
        allow_new_work: bool,
        filter_kwargs: Optional[Dict[str, object]],
        relevance_keywords: Optional[Sequence[str]],
        doc_keywords: Optional[Sequence[str]],
        domain_whitelist: Optional[Sequence[str]],
        date_from: Optional[str],
        date_windows: Optional[Sequence[Tuple[str, str]]],
    ) -> FrameSchema:
        schema_cfg = self.config.framing.schema

        if schema_cfg.source == "file":
            if not schema_cfg.schema_path:
                raise ValueError("framing.schema.schema_path is required when source='file'")
            return load_schema(Path(schema_cfg.schema_path))

        # Induction flow
        schema_path = self.paths.schema_path
        if (
            getattr(self.config, "regenerate_report_only", False)
            or getattr(self.config, "reload_framing_schema", False)
        ) and schema_path.exists():
            return load_schema(schema_path)

        if not allow_new_work and schema_path.exists():
            return load_schema(schema_path)
        if not allow_new_work:
            raise RuntimeError("Framing schema missing and new work is disabled.")

        sampler_kwargs = dict(filter_kwargs or {})
        if relevance_keywords:
            sampler_kwargs["keywords"] = list(relevance_keywords)
        sampler_config = SamplerConfig(
            sample_size=schema_cfg.induction_size,
            seed=self.config.seed,
            date_from=date_from,
            date_windows=date_windows,
            require_document_keywords=doc_keywords,
            domain_whitelist=domain_whitelist,
            **sampler_kwargs,
        )
        self._induction_samples = sampler.collect_chunks(sampler_config)
        passages = [text for _, text in self._induction_samples]

        llm_config = OpenAIConfig(
            model=schema_cfg.induction_model,
            temperature=schema_cfg.induction_temperature or 0.0,
            flex_processing=schema_cfg.flex_processing or False,
        )
        llm_client = OpenAIInterface(name="frame_induction", config=llm_config)

        max_per_call = schema_cfg.induction_batch_size or min(schema_cfg.induction_size, 40)
        inducer = FrameInducer(
            llm_client=llm_client,
            domain=self.config.framing.domain or "",
            frame_target=schema_cfg.frame_target,
            max_passages_per_call=max_per_call,
            max_total_passages=schema_cfg.induction_size * 2,
            induction_guidance=schema_cfg.induction_guidance,
            system_template=self.prompts.get("induction_system"),
            user_template=self.prompts.get("induction_user"),
        )
        schema = inducer.induce(passages)
        if not schema.schema_id and self.config.framing.domain:
            schema.schema_id = str(self.config.framing.domain).replace(" ", "_")

        schema_path.parent.mkdir(parents=True, exist_ok=True)
        save_schema(schema_path, schema)

        if inducer.emitted_messages and self.paths.results_dir:
            resolved_dir = self.paths.results_dir / "prompts" / "induction" / "resolved"
            self._save_resolved_messages(resolved_dir, "induction_call", inducer.emitted_messages)

        return schema

    def _annotate(
        self,
        sampler: EmbeddedCorporaSampler,
        corpora: EmbeddedCorpora,
        schema: FrameSchema,
        *,
        allow_new_work: bool,
        filter_kwargs: Optional[Dict[str, object]],
        relevance_keywords: Optional[Sequence[str]],
        doc_keywords: Optional[Sequence[str]],
        domain_whitelist: Optional[Sequence[str]],
        date_from: Optional[str],
        date_windows: Optional[Sequence[Tuple[str, str]]],
    ) -> Tuple[FrameAssignments, List["Candidate"]]:
        annotation_cfg = self.config.framing.annotation
        assignments = FrameAssignments()

        if annotation_cfg.size <= 0:
            return assignments, []

        # Build document filter for cached data
        doc_filter = DocumentFilter(
            keywords=list(doc_keywords) if doc_keywords else None,
            domain_whitelist=list(domain_whitelist) if domain_whitelist else None,
            date_from=date_from,
            date_windows=list(date_windows) if date_windows else None,
        )

        assignments_path = self.paths.assignments_path
        if (
            getattr(self.config, "regenerate_report_only", False)
            or getattr(self.config, "reload_framing_annotation", False)
        ) and assignments_path.exists():
            try:
                assignments = FrameAssignments.load(assignments_path)
                assignments = self._filter_assignments(assignments, doc_filter, corpora)
                assignments = self._limit_assignments(assignments, annotation_cfg.size)
                self.logger.info(
                    "Loaded %s frame annotations from cache (target %s).",
                    assignments.count,
                    annotation_cfg.size,
                )
                if assignments.count < annotation_cfg.size and allow_new_work:
                    missing = annotation_cfg.size - assignments.count
                    self.logger.info(
                        "Cached frame annotations short by %s; sampling additional passages.",
                        missing,
                    )
                    sampler_kwargs = dict(filter_kwargs or {})
                    sampler_config = SamplerConfig(
                        sample_size=missing,
                        seed=self.config.seed + 3,
                        date_from=date_from,
                        date_windows=date_windows,
                        require_document_keywords=doc_keywords,
                        domain_whitelist=domain_whitelist,
                        exclude_passage_ids=[item.passage_id for item in assignments],
                        **sampler_kwargs,
                    )
                    extra_candidates = sampler.collect_candidates(sampler_config)
                    if extra_candidates:
                        self.logger.info(
                            "Annotating %s additional frame candidates.",
                            len(extra_candidates),
                        )
                        extra_assignments = self._run_annotation(
                            schema,
                            extra_candidates,
                            annotation_cfg,
                            relevance_keywords,
                        )
                        if extra_assignments:
                            assignments.extend(extra_assignments)
                            assignments = self._limit_assignments(assignments, annotation_cfg.size)
                            assignments_path.parent.mkdir(parents=True, exist_ok=True)
                            assignments.save(assignments_path)
                return assignments, self._assignments_to_candidates(assignments)
            except Exception:
                assignments = FrameAssignments()

        if not allow_new_work and assignments_path.exists():
            assignments = FrameAssignments.load(assignments_path)
            assignments = self._filter_assignments(assignments, doc_filter, corpora)
            assignments = self._limit_assignments(assignments, annotation_cfg.size)
            self.logger.info(
                "Loaded %s frame annotations from cache (target %s).",
                assignments.count,
                annotation_cfg.size,
            )
            return assignments, self._assignments_to_candidates(assignments)
        if not allow_new_work:
            raise RuntimeError("Framing assignments missing and new work is disabled.")

        sampler_kwargs = dict(filter_kwargs or {})
        sampler_config = SamplerConfig(
            sample_size=annotation_cfg.size,
            seed=self.config.seed + 1,
            date_from=date_from,
            date_windows=date_windows,
            require_document_keywords=doc_keywords,
            domain_whitelist=domain_whitelist,
            **sampler_kwargs,
        )
        candidates = sampler.collect_candidates(sampler_config)
        self.logger.info(
            "Annotating %s frame candidates (target %s).",
            len(candidates),
            annotation_cfg.size,
        )

        assignments = self._run_annotation(
            schema,
            candidates,
            annotation_cfg,
            relevance_keywords,
        )

        assignments = self._limit_assignments(assignments, annotation_cfg.size)
        self.logger.info(
            "Frame annotation complete: %s passages.",
            assignments.count,
        )

        assignments_path.parent.mkdir(parents=True, exist_ok=True)
        assignments.save(assignments_path)

        return assignments, self._assignments_to_candidates(assignments)

    def _run_annotation(
        self,
        schema: FrameSchema,
        candidates: Sequence[Candidate],
        annotation_cfg: object,
        relevance_keywords: Optional[Sequence[str]],
    ) -> FrameAssignments:
        if not candidates:
            return FrameAssignments()
        llm_config = OpenAIConfig(
            model=annotation_cfg.model,
            temperature=annotation_cfg.temperature or 0.0,
            flex_processing=annotation_cfg.flex_processing or False,
        )
        llm_client = OpenAIInterface(name="frame_annotation", config=llm_config)

        resolved_dir = self.paths.results_dir / "prompts" / "frame_annotation" / "resolved"
        annotator = LLMFrameAnnotator(
            llm_client=llm_client,
            batch_size=annotation_cfg.batch_size,
            max_chars_per_passage=None,
            chunk_overlap_chars=0,
            system_template=self.prompts.get("application_system"),
            user_template=self.prompts.get("application_user"),
            resolved_messages_dir=resolved_dir,
            resolved_messages_prefix="frame_annotation_batch",
        )

        return FrameAssignments(
            annotator.batch_assign(
                schema,
                candidates,
                top_k=annotation_cfg.top_k,
                show_progress=True,
                progress_desc=f"Annotating frames ({llm_config.model})",
                relevance_keywords=annotation_cfg.force_zero_if_no_keywords or relevance_keywords,
                guidance=annotation_cfg.guidance,
            )
        )

    @staticmethod
    def _assignments_to_candidates(assignments: FrameAssignments) -> List[Candidate]:
        return [
            Candidate(
                item.passage_id,
                item.passage_text,
                meta=dict(item.metadata or {}),
            )
            for item in assignments
        ]

    def _limit_assignments(self, assignments: FrameAssignments, target_size: int) -> FrameAssignments:
        if target_size and assignments.count > target_size:
            return assignments.select_random(target_size, self.config.seed + 97)
        return assignments

    def _filter_assignments(
        self,
        assignments: FrameAssignments,
        doc_filter: DocumentFilter,
        corpora: EmbeddedCorpora,
    ) -> FrameAssignments:
        """Filter assignments using document filter applied to corpus metadata."""
        if doc_filter.is_empty():
            return assignments
        filtered = FrameAssignments()
        for assignment in assignments:
            corpus_name, local_doc_id, _ = split_passage_id(assignment.passage_id)
            try:
                embedded = corpora.get_embedded(corpus_name)
                if doc_filter.matches(local_doc_id, embedded.corpus):
                    filtered.append(assignment)
            except Exception:
                # Skip assignments where corpus lookup fails
                continue
        return filtered

    def _filter_classifications(
        self,
        classifications: DocumentClassifications,
        doc_filter: DocumentFilter,
        corpora: EmbeddedCorpora,
    ) -> DocumentClassifications:
        """Filter classifications using document filter applied to corpus metadata."""
        if doc_filter.is_empty():
            return classifications
        filtered = DocumentClassifications()
        for doc in classifications:
            doc_id = doc.doc_id
            if not doc_id:
                continue
            corpus_name, local_doc_id = split_global_doc_id(doc_id)
            try:
                embedded = corpora.get_embedded(corpus_name)
                if doc_filter.matches(local_doc_id, embedded.corpus):
                    filtered.append(doc)
            except Exception:
                # Skip classifications where corpus lookup fails
                continue
        return filtered

    def _train(
        self,
        schema: FrameSchema,
        assignments: FrameAssignments,
        *,
        allow_new_work: bool,
    ) -> Optional[FrameClassifierModel]:
        classifier_cfg = self.config.framing.classifier
        if not classifier_cfg.enabled:
            return None

        classifier_dir = self.paths.classifier_dir
        if (
            getattr(self.config, "regenerate_report_only", False)
            or getattr(self.config, "reload_framing_classifier", False)
        ) and classifier_dir.exists():
            model = FrameClassifierModel.load(classifier_dir)
            self._maybe_save_classifier_predictions(model, assignments)
            return model

        if not allow_new_work and classifier_dir.exists():
            model = FrameClassifierModel.load(classifier_dir)
            self._maybe_save_classifier_predictions(model, assignments)
            return model
        if not allow_new_work:
            raise RuntimeError("Frame classifier missing and new work is disabled.")

        label_set = FrameLabelSet.from_assignments(schema, assignments, source="llm")
        if not label_set.passages:
            return None
        self.logger.info(
            "Training frame classifier on %s annotated passages (from %s assignments).",
            len(label_set.passages),
            assignments.count,
        )

        spec = classifier_cfg.to_spec()
        spec.output_dir = str(classifier_dir)
        spec.run_name = f"{self.config.corpus}-frame-cls"

        trainer = FrameClassifierTrainer(spec)
        artifacts = trainer.run(
            label_set=label_set,
            assignments=assignments,
            cv_folds=classifier_cfg.cv_folds if classifier_cfg.cv_folds and classifier_cfg.cv_folds >= 2 else None,
        )
        if artifacts.predictions:
            predictions_path = self.paths.classifier_dir.parent / "classifier_predictions.json"
            predictions_path.parent.mkdir(parents=True, exist_ok=True)
            artifacts.save_predictions(predictions_path)
        if artifacts.model:
            classifier_dir.mkdir(parents=True, exist_ok=True)
            artifacts.model.save(classifier_dir)
            return artifacts.model
        return None

    def _classify(
        self,
        corpora: EmbeddedCorpora,
        sampler: EmbeddedCorporaSampler,
        schema: FrameSchema,
        assignments: FrameAssignments,
        model: Optional[FrameClassifierModel],
        *,
        allow_new_work: bool,
        doc_keywords: Optional[Sequence[str]],
        domain_whitelist: Optional[Sequence[str]],
        date_from: Optional[str],
        date_windows: Optional[Sequence[Tuple[str, str]]],
    ) -> DocumentClassifications:
        classification_cfg = self.config.framing.classification
        if not classification_cfg.size:
            return DocumentClassifications()

        # Build document filter for cached data
        doc_filter = DocumentFilter(
            keywords=list(doc_keywords) if doc_keywords else None,
            domain_whitelist=list(domain_whitelist) if domain_whitelist else None,
            date_from=date_from,
            date_windows=list(date_windows) if date_windows else None,
        )

        classifications_dir = self.paths.classifications_dir
        doc_ids: Optional[List[str]] = None
        desired_doc_ids: Optional[List[str]] = None
        cached: Optional[DocumentClassifications] = None
        fill_gap = False
        if (
            getattr(self.config, "regenerate_report_only", False)
            or getattr(self.config, "reload_framing_classifications", False)
        ) and classifications_dir.exists():
            cached = DocumentClassifications.from_folder(classifications_dir)
            cached = self._filter_classifications(cached, doc_filter, corpora)
            if cached.n_docs > 0:
                if allow_new_work:
                    desired_doc_ids = sampler.collect_docs(
                        sample_size=classification_cfg.size,
                        seed=self.config.seed,
                        date_from=date_from,
                        date_windows=date_windows,
                        require_keywords=doc_keywords,
                        domain_whitelist=domain_whitelist,
                    )
                    cached = DocumentClassifications.from_folder(
                        classifications_dir, doc_ids=desired_doc_ids
                    )
                    cached = self._filter_classifications(cached, doc_filter, corpora)
                    cached_doc_ids = {doc.doc_id for doc in cached if doc.doc_id}
                    missing = [doc_id for doc_id in desired_doc_ids if doc_id not in cached_doc_ids]
                    if not missing:
                        self.logger.info(
                            f"Using cached classifications ({cached.n_docs}) "
                            f"for {len(desired_doc_ids)} sampled documents."
                        )
                        return self._limit_classifications(cached, classification_cfg.size)
                    self.logger.info(
                        f"Cached classifications ({cached.n_docs}) < target "
                        f"({len(desired_doc_ids)}); classifying {len(missing)} missing documents."
                    )
                    doc_ids = missing
                    fill_gap = True
                else:
                    return self._limit_classifications(cached, classification_cfg.size)

        if not allow_new_work and classifications_dir.exists():
            cached = DocumentClassifications.from_folder(classifications_dir)
            cached = self._filter_classifications(cached, doc_filter, corpora)
            return self._limit_classifications(cached, classification_cfg.size)
        if not allow_new_work:
            raise RuntimeError("Frame classifications missing and new work is disabled.")

        if self.config.framing.classifier.enabled and model is not None:
            classifier = FrameClassifier(
                model=model,
                corpora=corpora,
                batch_size=max(1, self.config.framing.classifier.batch_size),
                seed=self.config.seed,
                require_keywords=doc_keywords,
                exclude_regex=self.config.filter.chunk.exclude_regex,
                exclude_min_hits=self.config.filter.chunk.exclude_min_hits,
                trim_after_markers=self.config.filter.chunk.trim_after_markers,
            )
            if doc_ids is None:
                if desired_doc_ids is None:
                    desired_doc_ids = sampler.collect_docs(
                        sample_size=classification_cfg.size,
                        seed=self.config.seed,
                        date_from=date_from,
                        date_windows=date_windows,
                        require_keywords=doc_keywords,
                        domain_whitelist=domain_whitelist,
                    )
                doc_ids = desired_doc_ids
            output_dir = None if fill_gap else classifications_dir
            new_classifications = DocumentClassifications(
                classifier.run(
                    sample_size=None,
                    doc_ids=doc_ids,
                    output_dir=output_dir,
                    desc="Classifying documents (framing)",
                )
            )
            if fill_gap and cached is not None:
                classifications = DocumentClassifications(list(cached) + list(new_classifications))
            else:
                classifications = new_classifications
        else:
            classifications = assignments.to_classifications()

        classifications = self._limit_classifications(classifications, classification_cfg.size)
        classifications_dir.mkdir(parents=True, exist_ok=True)
        classifications.to_folder(classifications_dir)
        return classifications

    def _limit_classifications(
        self,
        classifications: DocumentClassifications,
        target_size: Optional[int],
    ) -> DocumentClassifications:
        if not target_size or classifications.n_docs <= target_size:
            return classifications
        docs = list(classifications)
        rng = random.Random(self.config.seed + 29)
        rng.shuffle(docs)
        return DocumentClassifications(docs[:target_size])

    def _maybe_save_classifier_predictions(
        self,
        model: FrameClassifierModel,
        assignments: FrameAssignments,
    ) -> None:
        if not assignments:
            return
        predictions_path = self.paths.classifier_dir.parent / "classifier_predictions.json"
        if predictions_path.exists():
            return
        texts = [a.passage_text for a in assignments]
        if not texts:
            return
        probs = model.predict_proba_batch(texts, batch_size=max(1, self.config.framing.classifier.batch_size))
        predictions: List[Dict[str, object]] = []
        for assignment, prob_dict in zip(assignments, probs):
            ordered = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
            predictions.append(
                {
                    "passage_id": assignment.passage_id,
                    "probabilities": prob_dict,
                    "top_frames": [fid for fid, _ in ordered[:3]],
                }
            )
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        FrameClassifierArtifacts(model=None, predictions=predictions).save_predictions(predictions_path)

    def _save_resolved_messages(self, output_dir: Path, prefix: str, messages_by_call: list) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, messages in enumerate(messages_by_call, start=1):
            output_file = output_dir / f"{prefix}_{i:03d}.json"
            output_file.write_text(json.dumps(messages, indent=2, ensure_ascii=False), encoding="utf-8")


__all__ = ["Framer", "FramerPaths", "FramingResult", "load_schema", "save_schema"]
