"""
Stance detection stage for discourse analysis pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import random

from efi_core.pipeline import PipelineStage
from efi_analyser.frames.identifiers import make_global_passage_id, split_global_doc_id
from efi_analyser.frames.types import Candidate
from efi_analyser.frames.corpora import EmbeddedCorpora
from efi_analyser.frames.classifier import SamplerConfig
from efi_analyser.scorers.openai_interface import OpenAIInterface, OpenAIConfig
from efi_analyser.stance import (
    StanceAssignments,
    StanceDetector,
    StancePaths,
    LLMStanceAnnotator,
)
from efi_analyser.stance.trainer import StanceClassifierSpec, StanceClassifierTrainer, StanceClassifierModel

from apps.narrative_framing.filtering import filter_text

from .base import StageContext


@dataclass
class StanceDetectionResult:
    assignments: StanceAssignments
    classifications: StanceAssignments


class StanceDetectionStage(PipelineStage[StageContext, StanceDetectionResult]):
    """Annotate/train stance detector and classify corpus chunks."""

    def should_run(self, input_data: Optional[StageContext]) -> bool:
        if input_data is None:
            return False
        return bool(input_data.config.stance.enabled)

    def execute(self, input_data: StageContext) -> StanceDetectionResult:
        config = input_data.config
        state = input_data.state
        paths = input_data.paths

        if not state.sampler or not state.corpora:
            raise ValueError("Corpora and sampler must be initialized before stance detection.")

        stance_paths = StancePaths(
            annotation_path=paths.stance_annotation_path,
            classifier_dir=paths.stance_classifier_dir,
            classifications_dir=paths.stance_classifications_dir,
        )

        assignments = StanceAssignments()
        if config.stance.mode == "trained":
            assignments = self._load_or_annotate(input_data, stance_paths)
            self._train_if_needed(input_data, assignments, stance_paths)

        classifications = self._load_or_classify(input_data, stance_paths)

        return StanceDetectionResult(assignments=assignments, classifications=classifications)

    def _load_or_annotate(
        self,
        input_data: StageContext,
        paths: StancePaths,
    ) -> StanceAssignments:
        config = input_data.config
        if (
            config.stance.reload_annotation or config.regenerate_report_only
        ) and paths.annotation_path.exists():
            try:
                return StanceAssignments.load(paths.annotation_path)
            except Exception:
                pass

        if not input_data.allow_new_work and paths.annotation_path.exists():
            return StanceAssignments.load(paths.annotation_path)
        if not input_data.allow_new_work:
            raise RuntimeError("Stance annotations missing and new work is disabled.")

        candidates = self._select_annotation_candidates(input_data)
        llm_config = OpenAIConfig(
            model=config.stance.annotation.model,
            temperature=config.stance.annotation.temperature or 0.0,
        )
        llm_client = OpenAIInterface(name="stance_annotation", config=llm_config)
        resolved_dir = input_data.paths.results_dir / "prompts" / "stance_annotation" / "resolved"

        annotator = LLMStanceAnnotator(
            llm_client=llm_client,
            resolved_messages_dir=resolved_dir,
            resolved_messages_prefix="stance_annotation_call",
        )
        assignments = annotator.annotate(
            candidates,
            config.stance.targets,
            show_progress=True,
        )

        paths.annotation_path.parent.mkdir(parents=True, exist_ok=True)
        assignments.save(paths.annotation_path)
        return assignments

    def _train_if_needed(
        self,
        input_data: StageContext,
        assignments: StanceAssignments,
        paths: StancePaths,
    ) -> Optional[StanceClassifierModel]:
        config = input_data.config
        classifier_dir = paths.classifier_dir

        if (config.stance.reload_classifier or config.regenerate_report_only) and classifier_dir.exists():
            return StanceClassifierModel.load(classifier_dir)
        if not input_data.allow_new_work and classifier_dir.exists():
            return StanceClassifierModel.load(classifier_dir)
        if not input_data.allow_new_work:
            raise RuntimeError("Stance classifier missing and new work is disabled.")

        label_set = assignments.to_label_set()
        if not label_set.passages:
            return None

        spec = self._build_stance_spec(config.stance.classifier, classifier_dir)
        spec.run_name = f"{config.corpus}-stance-cls"
        trainer = StanceClassifierTrainer(spec, label_set.label_order)
        model = trainer.train(label_set)
        classifier_dir.mkdir(parents=True, exist_ok=True)
        model.save(classifier_dir)
        return model

    def _load_or_classify(
        self,
        input_data: StageContext,
        paths: StancePaths,
    ) -> StanceAssignments:
        config = input_data.config
        classifications_path = paths.classifications_dir / "classifications.json"

        if (
            config.stance.reload_classifications or config.regenerate_report_only
        ) and classifications_path.exists():
            cached = StanceAssignments.load(classifications_path)
            expected = self._expected_stance_rows(config)
            if expected and len(cached) < expected and input_data.allow_new_work:
                self.logger.info(
                    f"Cached stance rows ({len(cached)}) < target ({expected}); will re-run stance detection."
                )
            else:
                return cached

        if not input_data.allow_new_work and classifications_path.exists():
            return StanceAssignments.load(classifications_path)
        if not input_data.allow_new_work:
            raise RuntimeError("Stance classifications missing and new work is disabled.")

        if not config.stance.classification.size:
            return StanceAssignments()

        if config.stance.mode == "trained" and not paths.classifier_dir.exists():
            raise RuntimeError("Trained stance classifier not found for classification.")

        detector = StanceDetector.create(config.stance, paths)
        corpora: EmbeddedCorpora = input_data.state.corpora

        filter_spec = input_data.filter_spec
        results = StanceAssignments()

        chunk_pairs = self._collect_chunk_pairs(input_data, filter_spec)
        if chunk_pairs:
            self.logger.info(
                f"Stance detection using {len(chunk_pairs)} chunks from framing classifications "
                f"(doc sample target: {config.stance.classification.size})"
            )
            results.extend(detector.detect(chunk_pairs))
        else:
            doc_ids = input_data.state.sampler.collect_docs(
                sample_size=config.stance.classification.size,
                seed=config.seed,
                date_from=config.filter.date_from,
                date_windows=config.filter.date_windows,
                require_keywords=config.filter.document.keywords,
                domain_whitelist=config.filter.document.domain_whitelist,
            )
            self.logger.info(
                f"Stance detection using {len(doc_ids)} sampled documents "
                f"(target: {config.stance.classification.size})"
            )
            pairs: List[Tuple[str, str]] = []
            for doc_id in doc_ids:
                pairs.extend(self._extract_chunks(corpora, doc_id, filter_spec))
            pairs = self._limit_chunk_pairs(pairs, config.stance.chunk_limit, config.seed)
            if pairs:
                results.extend(detector.detect(pairs))

        paths.classifications_dir.mkdir(parents=True, exist_ok=True)
        results.save(classifications_path)
        return results

    def _collect_chunk_pairs(self, input_data: StageContext, filter_spec: object) -> List[Tuple[str, str]]:
        """Use framing classifications to build stance samples when available."""
        frame_classifications = input_data.state.frame_classifications
        if not frame_classifications:
            return []

        docs = list(frame_classifications)
        if not docs:
            return []

        # Ensure chunk_id is present; otherwise we cannot align stance to frames.
        has_chunk_id = False
        for doc in docs:
            chunks = doc.payload.get("chunks", []) if isinstance(doc.payload, dict) else []
            for chunk in chunks or []:
                if isinstance(chunk, dict) and chunk.get("chunk_id"):
                    has_chunk_id = True
                    break
            if has_chunk_id:
                break
        if not has_chunk_id:
            return []

        # Match stance sample size to framing doc sample when possible.
        config_sample = input_data.config.stance.classification.size
        if config_sample and len(docs) > config_sample:
            rng = random.Random(input_data.config.seed + 17)
            rng.shuffle(docs)
            docs = docs[:config_sample]

        pairs: List[Tuple[str, str]] = []
        for doc in docs:
            chunks = doc.payload.get("chunks", []) if isinstance(doc.payload, dict) else []
            for chunk in chunks or []:
                if not isinstance(chunk, dict):
                    continue
                chunk_id = str(chunk.get("chunk_id", "")).strip()
                if not chunk_id:
                    continue
                text = str(chunk.get("text", "")).strip()
                if not text:
                    continue
                if filter_spec is not None:
                    text = filter_text(text, filter_spec)
                    if not text:
                        continue
                pairs.append((chunk_id, text))

        # If stance sample size exceeds framing doc sample, add more docs.
        if config_sample and len(docs) < config_sample:
            needed = config_sample - len(docs)
            exclude_ids = [doc.doc_id for doc in docs if doc.doc_id]
            extra_doc_ids = input_data.state.sampler.collect_docs(
                sample_size=needed,
                seed=input_data.config.seed + 23,
                date_from=input_data.config.filter.date_from,
                date_windows=input_data.config.filter.date_windows,
                exclude_doc_ids=exclude_ids or None,
                require_keywords=input_data.config.filter.document.keywords,
                domain_whitelist=input_data.config.filter.document.domain_whitelist,
            )
            for doc_id in extra_doc_ids:
                pairs.extend(self._extract_chunks(input_data.state.corpora, doc_id, filter_spec))

        return self._limit_chunk_pairs(pairs, input_data.config.stance.chunk_limit, input_data.config.seed)

    def _limit_chunk_pairs(
        self,
        pairs: List[Tuple[str, str]],
        chunk_limit: Optional[int],
        seed: int,
    ) -> List[Tuple[str, str]]:
        if not chunk_limit or len(pairs) <= chunk_limit:
            return pairs
        rng = random.Random(seed + 31)
        rng.shuffle(pairs)
        return pairs[:chunk_limit]

    def _extract_chunks(
        self,
        corpora: EmbeddedCorpora,
        doc_id: str,
        filter_spec: object,
    ) -> List[Tuple[str, str]]:
        corpus_name, local_doc_id = split_global_doc_id(doc_id)
        embedded = corpora.get_embedded(corpus_name)
        doc = embedded.corpus.get_document(local_doc_id)
        if doc is None:
            return []
        chunks = embedded.get_chunks(local_doc_id, materialize_if_necessary=True) or []
        results: List[Tuple[str, str]] = []
        for chunk in chunks:
            raw_text = (chunk.text or "")
            if filter_spec is None:
                text = raw_text.strip()
            else:
                text = filter_text(raw_text, filter_spec)
            if not text:
                continue
            local_passage_id = f"{local_doc_id}:chunk{int(chunk.chunk_id):03d}"
            chunk_id = make_global_passage_id(corpus_name if len(corpora) > 1 else None, local_passage_id)
            results.append((chunk_id, text))
        return results

    def _select_annotation_candidates(self, input_data: StageContext) -> List[Candidate]:
        config = input_data.config
        target_size = max(0, config.stance.annotation.size)
        if target_size == 0:
            return []

        base_candidates: List[Candidate] = list(input_data.state.frame_annotation_candidates or [])
        if base_candidates:
            if target_size <= len(base_candidates):
                rng = random.Random(config.seed + 99)
                indices = list(range(len(base_candidates)))
                rng.shuffle(indices)
                return [base_candidates[i] for i in indices[:target_size]]

            # Need more candidates beyond framing sample
            remaining = target_size - len(base_candidates)
            exclude_ids = [c.item_id for c in base_candidates]
            extra = self._collect_candidates(input_data, remaining, exclude_ids)
            return base_candidates + extra

        return self._collect_candidates(input_data, target_size, None)

    def _collect_candidates(
        self,
        input_data: StageContext,
        sample_size: int,
        exclude_ids: Optional[Sequence[str]],
    ) -> List[Candidate]:
        sampler_kwargs = {}
        if input_data.filter:
            sampler_kwargs = input_data.filter.sampler_kwargs(
                domain_whitelist=input_data.config.filter.document.domain_whitelist
            )
        sampler_config = SamplerConfig(
            sample_size=sample_size,
            seed=input_data.config.seed + 3,
            exclude_passage_ids=exclude_ids or None,
            date_from=input_data.config.filter.date_from,
            date_windows=input_data.config.filter.date_windows,
            require_document_keywords=input_data.config.filter.document.keywords,
            **sampler_kwargs,
        )
        return input_data.state.sampler.collect_candidates(sampler_config)

    def _build_stance_spec(self, classifier_cfg, classifier_dir: Path) -> StanceClassifierSpec:
        spec_fields = set(StanceClassifierSpec.__dataclass_fields__.keys())
        cfg_dict = classifier_cfg.model_dump() if hasattr(classifier_cfg, "model_dump") else dict(classifier_cfg.__dict__)
        filtered = {k: v for k, v in cfg_dict.items() if k in spec_fields}
        spec = StanceClassifierSpec(**filtered)
        spec.output_dir = str(classifier_dir)
        return spec

    def _expected_stance_rows(self, config) -> Optional[int]:
        targets = list(getattr(config.stance, "targets", []) or [])
        if not targets:
            return None
        chunk_limit = getattr(config.stance, "chunk_limit", None)
        if chunk_limit:
            return int(chunk_limit) * len(targets)
        return None

    def get_metadata(self) -> dict:
        return {"stage": self.name}
