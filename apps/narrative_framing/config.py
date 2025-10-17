"""Configuration helpers for the narrative framing application."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

DEFAULT_CORPORA_ROOT = Path("corpora")
DEFAULT_WORKSPACE_ROOT = Path("workspace")
DEFAULT_INDUCTION_SAMPLE = 100
DEFAULT_APPLICATION_SAMPLE = 1000
DEFAULT_SEED = 42
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_INDUCTION_MODEL = "gpt-4o"
DEFAULT_APPLICATION_MODEL = "gpt-4o-mini"
DEFAULT_TARGET_WORDS = 200
DEFAULT_APPLICATION_BATCH = 8
DEFAULT_APPLICATION_TOP_K = 3

@dataclass
class ClassifierSettings:
    """Configuration for the optional frame classifier."""

    enabled: bool = False
    model_name: str = "microsoft/deberta-v3-base"
    batch_size: int = 8
    inference_batch_size: int = 8
    
    # Training parameters
    num_train_epochs: float = 3.0
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 384
    # Logging/reporting
    report_to: List[str] = field(default_factory=list)  # e.g., ["tensorboard", "wandb"]
    logging_dir: Optional[str] = None
    # Evaluation options
    eval_threshold: float = 0.5
    eval_top_k: Optional[int] = None
    eval_steps: Optional[int] = None


@dataclass
class NarrativeFramingConfig:
    """Typed configuration for the narrative framing workflow."""

    corpus: str = None
    corpora: Optional[List[str]] = None
    corpora_root: Path = DEFAULT_CORPORA_ROOT
    workspace_root: Path = DEFAULT_WORKSPACE_ROOT
    domain: str = "India coal media coverage"
    llm_model: str = DEFAULT_LLM_MODEL  # Deprecated: use induction_model and application_model
    induction_model: str = DEFAULT_INDUCTION_MODEL
    application_model: str = DEFAULT_APPLICATION_MODEL
    induction_frame_target: str = "between 5 and 10"
    induction_temperature: Optional[float] = None  # None = use model default
    application_temperature: Optional[float] = None  # None = use model default
    seed: int = DEFAULT_SEED
    filter_keywords: Optional[List[str]] = field(default_factory=lambda: [])
    # Content filtering (optional)
    filter_exclude_regex: Optional[List[str]] = None
    filter_exclude_min_hits: Optional[Dict[str, int]] = None  # e.g., {"share price": 3}
    filter_trim_after_markers: Optional[List[str]] = None  # strings after which to trim tail
    target_words: int = DEFAULT_TARGET_WORDS
    induction_sample_size: int = DEFAULT_INDUCTION_SAMPLE
    application_sample_size: int = DEFAULT_APPLICATION_SAMPLE
    application_batch_size: int = DEFAULT_APPLICATION_BATCH
    application_top_k: int = DEFAULT_APPLICATION_TOP_K
    reload_results: bool = False
    reload_induction: bool = False
    reload_application: bool = False
    reload_classifier: bool = False
    reload_chunk_classifications: bool = False
    reload_document_aggregates: bool = False
    reload_time_series: bool = False
    reset_chunk_classifications: bool = False
    reset_document_aggregates: bool = False
    results_dir: Optional[Path] = None
    classifier: ClassifierSettings = field(default_factory=ClassifierSettings)
    induction_guidance: Optional[str] = None
    classifier_corpus_sample_size: Optional[int] = None
    sampling_policy: str = "equal"

    def normalize(self) -> None:
        """Normalize derived fields after loading from disk."""

        if self.filter_keywords is not None and len(self.filter_keywords) == 0:
            self.filter_keywords = None
        if self.induction_guidance is not None:
            stripped = self.induction_guidance.strip()
            self.induction_guidance = stripped if stripped else None
        # Normalize filtering fields
        if self.filter_exclude_regex is not None:
            self.filter_exclude_regex = [str(p).strip() for p in self.filter_exclude_regex if str(p).strip()]
            if not self.filter_exclude_regex:
                self.filter_exclude_regex = None
        if self.filter_exclude_min_hits is not None:
            cleaned: Dict[str, int] = {}
            for k, v in self.filter_exclude_min_hits.items():
                key = str(k).strip()
                try:
                    val = int(v)
                except Exception:
                    continue
                if key and val >= 1:
                    cleaned[key] = val
            self.filter_exclude_min_hits = cleaned or None
        if self.filter_trim_after_markers is not None:
            self.filter_trim_after_markers = [str(m).strip() for m in self.filter_trim_after_markers if str(m).strip()]
            if not self.filter_trim_after_markers:
                self.filter_trim_after_markers = None
        if self.classifier_corpus_sample_size is not None and self.classifier_corpus_sample_size <= 0:
            self.classifier_corpus_sample_size = None
        if self.corpora is not None:
            self.corpora = [str(item).strip() for item in self.corpora if str(item).strip()]
            if not self.corpora:
                self.corpora = None
        self.sampling_policy = (self.sampling_policy or "equal").strip().lower()
        if self.sampling_policy not in {"equal", "proportional"}:
            raise ValueError(
                f"Unsupported sampling_policy '{self.sampling_policy}'. Expected 'equal' or 'proportional'."
            )

    def iter_corpus_names(self) -> Iterable[str]:
        if self.corpora:
            for name in self.corpora:
                yield name
        else:
            yield self.corpus


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
    if "batch_size" in data:
        settings.batch_size = int(data["batch_size"])
    if "inference_batch_size" in data:
        settings.inference_batch_size = int(data["inference_batch_size"])
    # Training parameters
    if "num_train_epochs" in data:
        settings.num_train_epochs = float(data["num_train_epochs"])
    if "learning_rate" in data:
        settings.learning_rate = float(data["learning_rate"])
    if "weight_decay" in data:
        settings.weight_decay = float(data["weight_decay"])
    if "warmup_ratio" in data:
        settings.warmup_ratio = float(data["warmup_ratio"])
    if "max_length" in data:
        settings.max_length = int(data["max_length"])
    # Logging/reporting
    if "report_to" in data:
        raw = data["report_to"]
        if isinstance(raw, (list, tuple)):
            settings.report_to = [str(x).strip() for x in raw if str(x).strip()]
        elif isinstance(raw, str) and raw.strip():
            settings.report_to = [s.strip() for s in raw.split(",") if s.strip()]
    if "logging_dir" in data:
        settings.logging_dir = str(data["logging_dir"]) if data["logging_dir"] else None
    # Evaluation options
    if "eval_threshold" in data:
        settings.eval_threshold = float(data["eval_threshold"]) if data["eval_threshold"] is not None else settings.eval_threshold
    if "eval_top_k" in data:
        val = data["eval_top_k"]
        try:
            parsed = int(val) if val is not None else None
        except Exception:
            parsed = None
        settings.eval_top_k = parsed if (parsed is None or parsed > 0) else None
    if "eval_steps" in data:
        try:
            settings.eval_steps = int(data["eval_steps"]) if data["eval_steps"] is not None else None
        except Exception:
            settings.eval_steps = None
    return settings


def load_config(path: Path) -> NarrativeFramingConfig:
    """Load configuration from a YAML file."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    data: Dict[str, Any] = payload or {}

    config = NarrativeFramingConfig()
    if "corpus" in data:
        raw_corpus = data["corpus"]
        if isinstance(raw_corpus, (list, tuple)):
            config.corpora = [str(item) for item in raw_corpus]
            if config.corpora:
                config.corpus = config.corpora[0]
        else:
            config.corpus = str(raw_corpus)
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
    if "induction_frame_target" in data:
        # Accept int or string in YAML, store as string
        raw = data["induction_frame_target"]
        config.induction_frame_target = str(raw) if raw is not None else config.induction_frame_target
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
    if "filter_exclude_regex" in data:
        patterns = data["filter_exclude_regex"]
        if patterns is None:
            config.filter_exclude_regex = None
        else:
            config.filter_exclude_regex = [str(item) for item in patterns]
    if "filter_exclude_min_hits" in data:
        mapping = data["filter_exclude_min_hits"] or {}
        if isinstance(mapping, dict):
            parsed: Dict[str, int] = {}
            for k, v in mapping.items():
                try:
                    parsed[str(k)] = int(v)
                except Exception:
                    continue
            config.filter_exclude_min_hits = parsed
    if "filter_trim_after_markers" in data:
        markers = data["filter_trim_after_markers"]
        if markers is None:
            config.filter_trim_after_markers = None
        else:
            config.filter_trim_after_markers = [str(item) for item in markers]
    if "target_words" in data:
        config.target_words = int(data["target_words"])
    if "induction_sample_size" in data:
        config.induction_sample_size = int(data["induction_sample_size"])
    if "application_sample_size" in data:
        config.application_sample_size = int(data["application_sample_size"])
    if "application_batch_size" in data:
        config.application_batch_size = int(data["application_batch_size"])
    if "application_top_k" in data:
        config.application_top_k = int(data["application_top_k"])
    if "reload_results" in data:
        config.reload_results = bool(data["reload_results"])
    if "reload_induction" in data:
        config.reload_induction = bool(data["reload_induction"])
    if "reload_application" in data:
        config.reload_application = bool(data["reload_application"])
    if "reload_classifier" in data:
        config.reload_classifier = bool(data["reload_classifier"])
    if "reload_chunk_classifications" in data:
        config.reload_chunk_classifications = bool(data["reload_chunk_classifications"])
    if "reload_document_aggregates" in data:
        config.reload_document_aggregates = bool(data["reload_document_aggregates"])
    if "reload_time_series" in data:
        config.reload_time_series = bool(data["reload_time_series"])
    if "reset_chunk_classifications" in data:
        config.reset_chunk_classifications = bool(data["reset_chunk_classifications"])
    if "reset_document_aggregates" in data:
        config.reset_document_aggregates = bool(data["reset_document_aggregates"])
    if "results_dir" in data:
        config.results_dir = _as_path(data["results_dir"])

    if "classifier" in data:
        config.classifier = _load_classifier_settings(data["classifier"])

    if "induction_guidance" in data:
        config.induction_guidance = str(data["induction_guidance"])
    if "classifier_corpus_sample_size" in data:
        value = data["classifier_corpus_sample_size"]
        config.classifier_corpus_sample_size = int(value) if value is not None else None
    if "sampling_policy" in data:
        config.sampling_policy = str(data["sampling_policy"]).strip().lower()

    if config.reload_results:
        if "reload_induction" not in data:
            config.reload_induction = True
        if "reload_application" not in data:
            config.reload_application = True
        if "reload_classifier" not in data:
            config.reload_classifier = True
        if "reload_chunk_classifications" not in data:
            config.reload_chunk_classifications = True
        if "reload_document_aggregates" not in data:
            config.reload_document_aggregates = True
        if "reload_time_series" not in data:
            config.reload_time_series = True
        if "reset_chunk_classifications" not in data:
            config.reset_chunk_classifications = False
        if "reset_document_aggregates" not in data:
            config.reset_document_aggregates = False

    config.normalize()
    return config


__all__ = [
    "ClassifierSettings",
    "NarrativeFramingConfig",
    "load_config",
]
