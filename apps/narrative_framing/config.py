"""Configuration helpers for the narrative framing application."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

DEFAULT_CORPUS_NAME = "mediacloud_india_coal"
DEFAULT_CORPORA_ROOT = Path("corpora")
DEFAULT_WORKSPACE_ROOT = Path("workspace")
DEFAULT_INDUCTION_SAMPLE = 100
DEFAULT_APPLICATION_SAMPLE = 1000
DEFAULT_SEED = 42
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_INDUCTION_MODEL = "gpt-4o"
DEFAULT_APPLICATION_MODEL = "gpt-4o-mini"
DEFAULT_TARGET_WORDS = 200
DEFAULT_MAX_CHARS = 900
DEFAULT_APPLICATION_BATCH = 8
DEFAULT_APPLICATION_TOP_K = 3
DEFAULT_APPLICATION_MAX_CHARS = 1200
DEFAULT_ASSIGNMENT_PREVIEW = 5


@dataclass
class ClassifierSettings:
    """Configuration for the optional frame classifier."""

    enabled: bool = False
    model_name: str = "microsoft/deberta-v3-base"
    output_dir: Path = DEFAULT_WORKSPACE_ROOT / "frame_classifier_model"
    batch_size: int = 8
    inference_batch_size: int = 8


@dataclass
class NarrativeFramingConfig:
    """Typed configuration for the narrative framing workflow."""

    corpus: str = DEFAULT_CORPUS_NAME
    corpora_root: Path = DEFAULT_CORPORA_ROOT
    workspace_root: Path = DEFAULT_WORKSPACE_ROOT
    domain: str = "India coal media coverage"
    llm_model: str = DEFAULT_LLM_MODEL  # Deprecated: use induction_model and application_model
    induction_model: str = DEFAULT_INDUCTION_MODEL
    application_model: str = DEFAULT_APPLICATION_MODEL
    induction_temperature: Optional[float] = None  # None = use model default
    application_temperature: Optional[float] = None  # None = use model default
    seed: int = DEFAULT_SEED
    filter_keywords: Optional[List[str]] = field(default_factory=lambda: ["coal"])
    target_words: int = DEFAULT_TARGET_WORDS
    max_chars: int = DEFAULT_MAX_CHARS
    induction_sample_size: int = DEFAULT_INDUCTION_SAMPLE
    application_sample_size: int = DEFAULT_APPLICATION_SAMPLE
    application_batch_size: int = DEFAULT_APPLICATION_BATCH
    application_max_chars: int = DEFAULT_APPLICATION_MAX_CHARS
    application_top_k: int = DEFAULT_APPLICATION_TOP_K
    skip_application: bool = False
    reload_results: bool = False
    reload_induction: bool = False
    reload_application: bool = False
    reload_classifier: bool = False
    reload_document_aggregates: bool = False
    reload_time_series: bool = False
    results_dir: Optional[Path] = None
    assignment_preview: int = DEFAULT_ASSIGNMENT_PREVIEW
    classifier: ClassifierSettings = field(default_factory=ClassifierSettings)
    induction_guidance: Optional[str] = None
    classifier_corpus_sample_size: Optional[int] = None

    def normalize(self) -> None:
        """Normalize derived fields after loading from disk."""

        if self.filter_keywords is not None and len(self.filter_keywords) == 0:
            self.filter_keywords = None
        if self.induction_guidance is not None:
            stripped = self.induction_guidance.strip()
            self.induction_guidance = stripped if stripped else None
        if self.classifier_corpus_sample_size is not None and self.classifier_corpus_sample_size <= 0:
            self.classifier_corpus_sample_size = None


def _as_path(value: Optional[Any]) -> Optional[Path]:
    if value in (None, ""):
        return None
    return Path(value)


def _load_classifier_settings(data: Dict[str, Any]) -> ClassifierSettings:
    settings = ClassifierSettings()
    if not data:
        return settings
    if "enabled" in data:
        settings.enabled = bool(data["enabled"])
    if "model_name" in data:
        settings.model_name = str(data["model_name"])
    if "output_dir" in data:
        settings.output_dir = Path(data["output_dir"])
    if "batch_size" in data:
        settings.batch_size = int(data["batch_size"])
    if "inference_batch_size" in data:
        settings.inference_batch_size = int(data["inference_batch_size"])
    return settings


def load_config(path: Path) -> NarrativeFramingConfig:
    """Load configuration from a YAML file."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    data: Dict[str, Any] = payload or {}

    config = NarrativeFramingConfig()
    if "corpus" in data:
        config.corpus = str(data["corpus"])
    if "corpora_root" in data:
        config.corpora_root = Path(data["corpora_root"])
    if "workspace_root" in data:
        config.workspace_root = Path(data["workspace_root"])
    if "domain" in data:
        config.domain = str(data["domain"])
    if "induction_model" in data:
        config.induction_model = str(data["induction_model"])
    if "application_model" in data:
        config.application_model = str(data["application_model"])
    if "induction_temperature" in data:
        config.induction_temperature = float(data["induction_temperature"]) if data["induction_temperature"] is not None else None
    if "application_temperature" in data:
        config.application_temperature = float(data["application_temperature"]) if data["application_temperature"] is not None else None
    if "seed" in data:
        config.seed = int(data["seed"])
    if "filter_keywords" in data:
        keywords = data["filter_keywords"]
        if keywords is None:
            config.filter_keywords = None
        else:
            config.filter_keywords = [str(item) for item in keywords]
    if "target_words" in data:
        config.target_words = int(data["target_words"])
    if "max_chars" in data:
        config.max_chars = int(data["max_chars"])
    if "induction_sample_size" in data:
        config.induction_sample_size = int(data["induction_sample_size"])
    if "application_sample_size" in data:
        config.application_sample_size = int(data["application_sample_size"])
    if "application_batch_size" in data:
        config.application_batch_size = int(data["application_batch_size"])
    if "application_max_chars" in data:
        config.application_max_chars = int(data["application_max_chars"])
    if "application_top_k" in data:
        config.application_top_k = int(data["application_top_k"])
    if "skip_application" in data:
        config.skip_application = bool(data["skip_application"])
    if "reload_results" in data:
        config.reload_results = bool(data["reload_results"])
    if "reload_induction" in data:
        config.reload_induction = bool(data["reload_induction"])
    if "reload_application" in data:
        config.reload_application = bool(data["reload_application"])
    if "reload_classifier" in data:
        config.reload_classifier = bool(data["reload_classifier"])
    if "reload_document_aggregates" in data:
        config.reload_document_aggregates = bool(data["reload_document_aggregates"])
    if "reload_time_series" in data:
        config.reload_time_series = bool(data["reload_time_series"])
    if "results_dir" in data:
        config.results_dir = _as_path(data["results_dir"])
    if "assignment_preview" in data:
        config.assignment_preview = int(data["assignment_preview"])

    if "classifier" in data:
        config.classifier = _load_classifier_settings(data["classifier"])

    if "induction_guidance" in data:
        config.induction_guidance = str(data["induction_guidance"])
    if "classifier_corpus_sample_size" in data:
        value = data["classifier_corpus_sample_size"]
        config.classifier_corpus_sample_size = int(value) if value is not None else None

    if config.reload_results:
        if "reload_induction" not in data:
            config.reload_induction = True
        if "reload_application" not in data:
            config.reload_application = True
        if "reload_classifier" not in data:
            config.reload_classifier = True
        if "reload_document_aggregates" not in data:
            config.reload_document_aggregates = True
        if "reload_time_series" not in data:
            config.reload_time_series = True

    config.normalize()
    return config


__all__ = [
    "ClassifierSettings",
    "NarrativeFramingConfig",
    "load_config",
]
