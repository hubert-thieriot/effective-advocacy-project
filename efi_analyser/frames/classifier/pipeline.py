"""High-level pipeline for inducing, labelling, and training frame classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from efi_analyser.frames import FrameAssignment, FrameInducer, FrameSchema, LLMFrameApplicator
from efi_analyser.frames.classifier.dataset import FrameLabelSet
from efi_analyser.frames.classifier.model import FrameClassifierModel, FrameClassifierSpec
from efi_analyser.frames.classifier.sampler import CorpusSampler, SamplerConfig
from efi_analyser.frames.classifier.trainer import FrameClassifierTrainer


@dataclass
class InductionConfig:
    """Configuration for LLM frame induction."""

    sample_size: int = 100
    seed: int = 42
    keywords: Optional[Sequence[str]] = None
    frame_target: int | str = "between 6 and 10"
    guidance: Optional[str] = None
    max_passages_per_call: Optional[int] = None
    max_total_passages: Optional[int] = None


@dataclass
class ApplicationConfig:
    """Configuration for LLM frame application to generate labels."""

    sample_size: int = 1000
    seed: int = 43
    keywords: Optional[Sequence[str]] = None
    batch_size: int = 8
    top_k: int = 3
    max_chars_per_passage: int = 1200
    chunk_overlap_chars: Optional[int] = None
    exclude_induction_passages: bool = True


@dataclass
class SplitConfig:
    """Train/dev/test split ratios for the weakly labelled dataset."""

    train_ratio: float = 0.8
    dev_ratio: float = 0.1
    seed: int = 13

    def validate(self) -> None:
        if self.train_ratio <= 0:
            raise ValueError("train_ratio must be positive")
        if self.dev_ratio < 0:
            raise ValueError("dev_ratio must be ≥ 0")
        if self.train_ratio + self.dev_ratio > 1:
            raise ValueError("train_ratio + dev_ratio must be ≤ 1")


@dataclass
class FrameClassifierArtifacts:
    """Artifacts produced by the end-to-end frame classifier pipeline."""

    schema: FrameSchema
    induction_passages: List[Tuple[str, str]]
    application_passages: List[Tuple[str, str]]
    assignments: List[FrameAssignment]
    label_set: FrameLabelSet
    train_set: FrameLabelSet
    dev_set: Optional[FrameLabelSet]
    test_set: Optional[FrameLabelSet]
    model: FrameClassifierModel


class FrameClassifierPipeline:
    """Coordinate schema induction, weak labelling, and classifier training."""

    def __init__(
        self,
        *,
        embedded_corpus,
        domain: str,
        inducer_client,
        applicator_client,
        classifier_spec: FrameClassifierSpec,
        induction_config: InductionConfig = InductionConfig(),
        application_config: ApplicationConfig = ApplicationConfig(),
        split_config: SplitConfig = SplitConfig(),
        label_source: str = "llm",
        compute_metrics: Optional[Callable[[Dict[str, Any]], Dict[str, float]]] = None,
        sampler: Optional[CorpusSampler] = None,
        trainer: Optional[FrameClassifierTrainer] = None,
    ) -> None:
        self.embedded_corpus = embedded_corpus
        self.domain = domain
        self.inducer_client = inducer_client
        self.applicator_client = applicator_client
        self.classifier_spec = classifier_spec
        self.induction_config = induction_config
        self.application_config = application_config
        self.split_config = split_config
        self.label_source = label_source
        self.compute_metrics = compute_metrics

        self.sampler = sampler or CorpusSampler(embedded_corpus)
        self.trainer = trainer or FrameClassifierTrainer(classifier_spec)

    def run(
        self,
        *,
        schema_override: Optional[FrameSchema] = None,
        induction_passages_override: Optional[Sequence[Tuple[str, str]]] = None,
        application_passages_override: Optional[Sequence[Tuple[str, str]]] = None,
        assignments_override: Optional[Sequence[FrameAssignment]] = None,
    ) -> FrameClassifierArtifacts:
        schema, induction_passages = self._build_schema(
            schema_override=schema_override,
            passages_override=induction_passages_override,
        )
        application_passages, assignments = self._label_passages(
            schema,
            induction_passages,
            passages_override=application_passages_override,
            assignments_override=assignments_override,
        )
        label_set = FrameLabelSet.from_assignments(
            schema,
            assignments,
            source=self.label_source,
            metadata={
                "domain": schema.domain,
                "induction_sample_size": len(induction_passages),
                "application_sample_size": len(application_passages),
            },
        )

        train_set, dev_set, test_set = self._split_label_set(label_set)
        model = self.trainer.train(
            label_set=train_set,
            eval_set=dev_set,
            compute_metrics=self.compute_metrics,
        )

        return FrameClassifierArtifacts(
            schema=schema,
            induction_passages=induction_passages,
            application_passages=application_passages,
            assignments=list(assignments),
            label_set=label_set,
            train_set=train_set,
            dev_set=dev_set,
            test_set=test_set,
            model=model,
        )

    # ------------------------------------------------------------------ helpers
    def _build_schema(
        self,
        *,
        schema_override: Optional[FrameSchema],
        passages_override: Optional[Sequence[Tuple[str, str]]],
    ) -> Tuple[FrameSchema, List[Tuple[str, str]]]:
        if schema_override is not None:
            schema = schema_override
            induction_passages = list(passages_override or [])
        else:
            induction_passages = self._collect_passages(
                SamplerConfig(
                    sample_size=self.induction_config.sample_size,
                    seed=self.induction_config.seed,
                    keywords=self.induction_config.keywords,
                )
            )
            passages_text = [text for _, text in induction_passages]

            max_passages_per_call = self.induction_config.max_passages_per_call
            if max_passages_per_call is None:
                max_passages_per_call = max(20, min(len(passages_text), 80))

            max_total_passages = self.induction_config.max_total_passages
            if max_total_passages is None:
                max_total_passages = max(len(passages_text), max_passages_per_call) * 2

            inducer = FrameInducer(
                llm_client=self.inducer_client,
                domain=self.domain,
                frame_target=self.induction_config.frame_target,
                max_passages_per_call=max_passages_per_call,
                max_total_passages=max_total_passages,
                frame_guidance=self.induction_config.guidance,
            )
            schema = inducer.induce(passages_text)

        if not schema.schema_id:
            schema.schema_id = self.domain.replace(" ", "_")
        return schema, induction_passages

    def _label_passages(
        self,
        schema: FrameSchema,
        induction_passages: Sequence[Tuple[str, str]],
        *,
        passages_override: Optional[Sequence[Tuple[str, str]]],
        assignments_override: Optional[Sequence[FrameAssignment]],
    ) -> Tuple[List[Tuple[str, str]], List[FrameAssignment]]:
        if assignments_override is not None:
            return list(passages_override or []), list(assignments_override)

        if passages_override is not None:
            application_passages = list(passages_override)
        else:
            if self.application_config.sample_size <= 0:
                raise ValueError("application_config.sample_size must be positive when passages are not provided")
            exclude_ids = None
            if self.application_config.exclude_induction_passages and induction_passages:
                exclude_ids = [pid for pid, _ in induction_passages]
            application_passages = self._collect_passages(
                SamplerConfig(
                    sample_size=self.application_config.sample_size,
                    seed=self.application_config.seed,
                    keywords=self.application_config.keywords,
                    exclude_passage_ids=exclude_ids,
                )
            )

        chunk_overlap = self.application_config.chunk_overlap_chars
        if chunk_overlap is None:
            chunk_overlap = int(self.application_config.max_chars_per_passage * 0.1)

        applicator = LLMFrameApplicator(
            llm_client=self.applicator_client,
            batch_size=self.application_config.batch_size,
            max_chars_per_passage=self.application_config.max_chars_per_passage,
            chunk_overlap_chars=chunk_overlap,
        )
        assignments = applicator.batch_assign(
            schema,
            application_passages,
            top_k=self.application_config.top_k,
        )
        return application_passages, assignments

    def _collect_passages(self, config: SamplerConfig) -> List[Tuple[str, str]]:
        return self.sampler.collect(config)

    def _split_label_set(
        self,
        label_set: FrameLabelSet,
    ) -> Tuple[FrameLabelSet, Optional[FrameLabelSet], Optional[FrameLabelSet]]:
        self.split_config.validate()
        train_set, dev_set, test_set = label_set.split(
            train_ratio=self.split_config.train_ratio,
            dev_ratio=self.split_config.dev_ratio,
            seed=self.split_config.seed,
        )
        if len(dev_set.passages) == 0:
            dev_set = None
        if len(test_set.passages) == 0:
            test_set = None
        return train_set, dev_set, test_set


__all__ = [
    "InductionConfig",
    "ApplicationConfig",
    "SplitConfig",
    "FrameClassifierArtifacts",
    "FrameClassifierPipeline",
]
