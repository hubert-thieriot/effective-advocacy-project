"""
Pydantic-based configuration models for narrative framing application.

Replaces the 470-line load_config() function with declarative validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator
import yaml


# ============================================================================
# Utility functions
# ============================================================================

def normalize_string_list(items: Optional[List[str]]) -> Optional[List[str]]:
    """Normalize a list of strings (strip, remove empty)"""
    if items is None:
        return None
    normalized = [s.strip() for s in items if s and str(s).strip()]
    return normalized or None


def normalize_keyword_dict(
    keywords: Optional[Union[List[str], Dict[str, List[str]]]]
) -> Optional[Union[List[str], Dict[str, List[str]]]]:
    """Normalize keywords - can be flat list or dict by language"""
    if keywords is None:
        return None

    if isinstance(keywords, dict):
        # Dict by language code: {english: [...], german: [...], ...}
        normalized = {
            lang: [str(item).lower().strip() for item in kw_list if item]
            for lang, kw_list in keywords.items()
            if kw_list
        }
        return normalized or None
    else:
        # Flat list (language-agnostic)
        result = [str(item).lower().strip() for item in keywords if item]
        return result or None


def parse_date_windows(data: Any) -> Optional[List[Tuple[str, str]]]:
    """Parse date windows from config data.

    Supports two formats:
    1. Explicit ranges: [{from: "2020-01-01", to: "2020-04-04"}, ...]
    2. Pattern-based yearly windows: {pattern: "yearly", from_month_day: "01-01", to_month_day: "04-04", years: [2020, 2021, ...]}

    Returns a list of (from_date, to_date) tuples in YYYY-MM-DD format.
    """
    if data is None:
        return None

    if isinstance(data, dict):
        # Pattern-based format
        if "pattern" in data and data["pattern"] == "yearly":
            from_month_day = data.get("from_month_day", "")
            to_month_day = data.get("to_month_day", "")
            years = data.get("years", [])

            if not from_month_day or not to_month_day or not years:
                return None

            # Validate month-day format (MM-DD)
            try:
                datetime.strptime(f"2000-{from_month_day}", "%Y-%m-%d")
                datetime.strptime(f"2000-{to_month_day}", "%Y-%m-%d")
            except ValueError:
                return None

            windows = []
            for year in years:
                year_str = str(year)
                from_date = f"{year_str}-{from_month_day}"
                to_date = f"{year_str}-{to_month_day}"
                windows.append((from_date, to_date))

            return windows if windows else None

        # Single explicit range: {from: "...", to: "..."}
        if "from" in data and "to" in data:
            from_date = str(data["from"]).strip()
            to_date = str(data["to"]).strip()
            if from_date and to_date:
                return [(from_date, to_date)]

    elif isinstance(data, list):
        # List of explicit ranges
        windows = []
        for item in data:
            if isinstance(item, dict) and "from" in item and "to" in item:
                from_date = str(item["from"]).strip()
                to_date = str(item["to"]).strip()
                if from_date and to_date:
                    windows.append((from_date, to_date))
        return windows if windows else None

    return None


# ============================================================================
# Filter Configuration
# ============================================================================

class DocumentFilter(BaseModel):
    """Document-level filtering configuration"""
    keywords: Optional[List[str]] = None
    domain_whitelist: Optional[List[str]] = None
    exclude_regex: Optional[List[str]] = None
    exclude_min_hits: Optional[Dict[str, int]] = None
    trim_after_markers: Optional[List[str]] = None

    @field_validator('keywords', 'exclude_regex', 'trim_after_markers')
    @classmethod
    def normalize_string_lists(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        return normalize_string_list(v)

    @field_validator('domain_whitelist')
    @classmethod
    def normalize_domains(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return None
        normalized = [s.lower().strip() for s in v if s and str(s).strip()]
        return normalized or None

    @field_validator('exclude_min_hits')
    @classmethod
    def validate_min_hits(cls, v: Optional[Dict[str, int]]) -> Optional[Dict[str, int]]:
        if v is None:
            return None
        cleaned = {}
        for k, val in v.items():
            key = str(k).strip()
            try:
                int_val = int(val)
                if key and int_val >= 1:
                    cleaned[key] = int_val
            except (ValueError, TypeError):
                continue
        return cleaned or None


class ChunkFilter(BaseModel):
    """Chunk-level filtering configuration"""
    keywords: Optional[List[str]] = None
    exclude_regex: Optional[List[str]] = None
    exclude_min_hits: Optional[Dict[str, int]] = None
    trim_after_markers: Optional[List[str]] = None

    @field_validator('keywords', 'exclude_regex', 'trim_after_markers')
    @classmethod
    def normalize_string_lists(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        return normalize_string_list(v)

    @field_validator('exclude_min_hits')
    @classmethod
    def validate_min_hits(cls, v: Optional[Dict[str, int]]) -> Optional[Dict[str, int]]:
        if v is None:
            return None
        cleaned = {}
        for k, val in v.items():
            key = str(k).strip()
            try:
                int_val = int(val)
                if key and int_val >= 1:
                    cleaned[key] = int_val
            except (ValueError, TypeError):
                continue
        return cleaned or None


class FilterConfig(BaseModel):
    """Combined filtering configuration"""
    date_from: Optional[str] = None
    date_windows: Optional[List[Tuple[str, str]]] = None
    document: DocumentFilter = Field(default_factory=DocumentFilter)
    chunk: ChunkFilter = Field(default_factory=ChunkFilter)

    @field_validator('date_from')
    @classmethod
    def normalize_date_from(cls, v: Optional[str]) -> Optional[str]:
        if v is None or not v:
            return None
        stripped = str(v).strip()
        return stripped if stripped else None


# ============================================================================
# Chunking Configuration
# ============================================================================

class ChunkingConfig(BaseModel):
    """Chunking configuration"""
    target_words: int = 200
    chunker_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# ============================================================================
# Induction Configuration
# ============================================================================

class InductionConfig(BaseModel):
    """Frame induction configuration"""
    size: int = 100
    model: str = "claude-sonnet-4"
    frame_target: str = "10"
    temperature: Optional[float] = Field(default=0.0, ge=0.0, le=2.0)
    guidance: Optional[str] = None
    batch_size: Optional[int] = None

    @field_validator('guidance')
    @classmethod
    def normalize_guidance(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        stripped = str(v).strip()
        return stripped if stripped else None

    @field_validator('batch_size')
    @classmethod
    def validate_batch_size(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        return v if v > 0 else None


# ============================================================================
# Annotation Configuration
# ============================================================================

class AnnotationConfig(BaseModel):
    """Frame annotation configuration"""
    size: int = 500
    model: str = "claude-sonnet-4"
    batch_size: int = 5
    top_k: int = 3
    temperature: Optional[float] = Field(default=0.0, ge=0.0, le=2.0)
    force_zero_if_no_keywords: Optional[Union[List[str], Dict[str, List[str]]]] = None
    guidance: Optional[str] = None
    verbose: Optional[bool] = None
    export_annotated_html: bool = False
    annotated_html_subfolder_fields: Optional[List[str]] = None

    @field_validator('guidance')
    @classmethod
    def normalize_guidance(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        stripped = str(v).strip()
        return stripped if stripped else None

    @field_validator('force_zero_if_no_keywords')
    @classmethod
    def normalize_keywords(cls, v):
        return normalize_keyword_dict(v)


# ============================================================================
# Classification Configuration
# ============================================================================

class ClassificationConfig(BaseModel):
    """Classification/application configuration"""
    size: Optional[int] = None

    @field_validator('size')
    @classmethod
    def validate_size(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        return v if v > 0 else None


# ============================================================================
# Classifier Settings
# ============================================================================

class ClassifierSettings(BaseModel):
    """Classifier training and inference settings.

    Compatible with efi_analyser.frames.classifier.FrameClassifierSpec.
    """
    # App-specific settings
    enabled: bool = True
    cv_folds: Optional[int] = None

    # Core FrameClassifierSpec fields
    model_name: str = "distilbert-base-multilingual-cased"
    max_length: int = 512
    learning_rate: float = 2e-5
    num_train_epochs: float = 3.0
    batch_size: int = 8
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    fp16: bool = False
    seed: int = 13
    output_dir: str = "frame_classifier_runs"
    freeze_base_model: bool = False

    # Logging/reporting
    report_to: List[str] = Field(default_factory=lambda: ["none"])
    logging_dir: Optional[str] = None
    run_name: Optional[str] = None

    # Evaluation options
    eval_threshold: float = 0.5
    eval_top_k: Optional[int] = None
    eval_steps: Optional[int] = None

    @field_validator('report_to')
    @classmethod
    def normalize_report_to(cls, v: Union[List[str], str]) -> List[str]:
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return [str(x).strip() for x in v if str(x).strip()]

    @field_validator('eval_top_k', 'cv_folds')
    @classmethod
    def validate_positive_optional(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        return v if v > 0 else None

    @field_validator('cv_folds')
    @classmethod
    def validate_cv_folds(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        return v if v >= 2 else None

    def to_spec(self):
        """Convert to FrameClassifierSpec for use with efi_analyser.

        Returns a dictionary containing only the fields needed by FrameClassifierSpec,
        excluding app-specific fields like 'enabled' and 'cv_folds'.
        """
        from efi_analyser.frames.classifier import FrameClassifierSpec

        # Get all fields except app-specific ones
        spec_dict = self.model_dump(exclude={'enabled', 'cv_folds'})

        return FrameClassifierSpec(**spec_dict)


# ============================================================================
# Report Configuration
# ============================================================================

class PlotSettings(BaseModel):
    """Plot settings for reports"""
    title: Optional[str] = None
    subtitle: Optional[str] = None
    note: Optional[str] = None


class CustomPlotSettings(BaseModel):
    """Custom plot configuration"""
    type: str
    title: Optional[str] = None
    subtitle: Optional[str] = None
    caption: Optional[str] = None


class ReportConfig(BaseModel):
    """Report generation configuration"""
    hide_empty_passages: bool = False
    plot: PlotSettings = Field(default_factory=PlotSettings)
    custom_plots: List[CustomPlotSettings] = Field(default_factory=list)
    export_plots_dir: Optional[Path] = None
    export_includes_dir: Optional[Path] = None
    export_plot_formats: List[str] = Field(default_factory=lambda: ["png"])
    n_min_per_media: Optional[int] = None
    domain_mapping_max_domains: int = 20
    include_domain_yearly_bar_charts: bool = False
    domain_yearly_top_domains: int = 10
    domain_yearly_version: str = "weighted"
    domain_yearly_include_empty: bool = False
    include_yearly_bar_charts: bool = True

    @field_validator('export_plots_dir', 'export_includes_dir')
    @classmethod
    def expand_path(cls, v: Optional[Path]) -> Optional[Path]:
        if v is None:
            return None
        return Path(v).expanduser().resolve()

    @field_validator('export_plot_formats')
    @classmethod
    def normalize_formats(cls, v: Union[List[str], str]) -> List[str]:
        if isinstance(v, str):
            return [f.strip().lower() for f in v.split(",") if f.strip()]
        return [str(f).strip().lower() for f in v if str(f).strip()]

    @field_validator('domain_yearly_version')
    @classmethod
    def validate_version(cls, v: str) -> str:
        v_lower = v.lower()
        if v_lower not in ("weighted", "occurrence"):
            return "weighted"
        return v_lower

    @field_validator('n_min_per_media')
    @classmethod
    def validate_n_min(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        return v if v > 0 else None


# ============================================================================
# Main Configuration
# ============================================================================

class NarrativeFramingConfig(BaseModel):
    """Complete narrative framing configuration"""

    # Core paths and settings
    corpus: str
    corpora: Optional[List[str]] = None
    corpora_root: Path = Path.home() / "corpora"
    workspace_root: Path = Path.home() / "workspace"
    domain: Optional[str] = None

    # Legacy top-level settings (deprecated but supported for backward compatibility)
    induction_model: Optional[str] = None
    application_model: Optional[str] = None
    induction_frame_target: Optional[str] = None
    induction_temperature: Optional[float] = None
    application_temperature: Optional[float] = None
    target_words: Optional[int] = None
    chunker_model: Optional[str] = None
    induction_sample_size: Optional[int] = None
    application_sample_size: Optional[int] = None
    application_batch_size: Optional[int] = None
    application_top_k: Optional[int] = None
    induction_guidance: Optional[str] = None

    # Legacy filter settings (deprecated)
    filter_keywords: Optional[List[str]] = None
    filter_exclude_regex: Optional[List[str]] = None
    filter_exclude_min_hits: Optional[Dict[str, int]] = None
    filter_trim_after_markers: Optional[List[str]] = None
    date_from: Optional[str] = None

    # Seed for reproducibility
    seed: int = 42

    # Reload flags
    reload_induction: bool = False
    reload_annotation: bool = False
    reload_classifier: bool = False
    reload_classifications: bool = False
    reload_aggregates: bool = False
    reload_results: bool = False
    reload_application: bool = False
    regenerate_report_only: bool = False

    # Allow new work flags
    allow_new_induction: bool = True
    allow_new_annotation: bool = True
    allow_new_training: bool = True
    allow_new_classification: bool = True
    allow_new_aggregation: bool = True

    # Top-level relevance keywords (used for both induction and annotation)
    relevance_keywords: Optional[Union[List[str], Dict[str, List[str]]]] = None

    # Sampling configuration
    sampling_policy: str = "equal"
    classifier_corpus_sample_size: Optional[int] = None
    corpus_aliases: Optional[Dict[str, str]] = None

    # Aggregation settings (deprecated)
    agg_min_threshold_weighted: Optional[float] = None
    agg_normalize_weighted: Optional[bool] = None
    agg_min_threshold_occurrence: Optional[float] = None

    # Nested configuration sections
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    filter: FilterConfig = Field(default_factory=FilterConfig)
    induction: InductionConfig = Field(default_factory=InductionConfig)
    annotation: AnnotationConfig = Field(default_factory=AnnotationConfig)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    classifier: ClassifierSettings = Field(default_factory=ClassifierSettings)
    report: ReportConfig = Field(default_factory=ReportConfig)

    # Output directories
    results_dir: Optional[Path] = None
    export_dir: Optional[Path] = None

    # Additional plots
    additional_plots: List[Dict[str, Any]] = Field(default_factory=list)

    # Source config path (for provenance)
    source_config_path: Optional[Path] = None

    @field_validator('corpora_root', 'workspace_root', 'results_dir', 'export_dir')
    @classmethod
    def expand_paths(cls, v: Optional[Path]) -> Optional[Path]:
        if v is None:
            return None
        return Path(v).expanduser().resolve()

    @field_validator('sampling_policy')
    @classmethod
    def validate_sampling_policy(cls, v: str) -> str:
        v_lower = v.strip().lower()
        if v_lower not in {"equal", "proportional"}:
            raise ValueError(
                f"Unsupported sampling_policy '{v}'. Expected 'equal' or 'proportional'."
            )
        return v_lower

    @field_validator('relevance_keywords')
    @classmethod
    def normalize_keywords(cls, v):
        return normalize_keyword_dict(v)

    @field_validator('corpus_aliases')
    @classmethod
    def normalize_aliases(cls, v: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
        if v is None:
            return None
        cleaned = {}
        for k, val in v.items():
            key = str(k).strip()
            value = str(val).strip()
            if key and value:
                cleaned[key] = value
        return cleaned or None

    @field_validator('corpora')
    @classmethod
    def normalize_corpora(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return None
        normalized = [str(item).strip() for item in v if str(item).strip()]
        return normalized or None

    @model_validator(mode='after')
    def handle_reload_results(self):
        """If reload_results is True, set all reload flags to True (unless explicitly set)"""
        if self.reload_results:
            # Note: In YAML loading, we need to check if values were explicitly set
            # For now, just set them all to True
            self.reload_induction = True
            self.reload_application = True
            self.reload_classifier = True
            self.reload_classifications = True
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> "NarrativeFramingConfig":
        """Load configuration from a YAML file"""
        if not path.exists():
            # Return default config if file doesn't exist
            return cls(corpus="default")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # Handle corpus/corpora field
        if "corpus" in data:
            raw_corpus = data["corpus"]
            if isinstance(raw_corpus, (list, tuple)):
                data["corpora"] = [str(item) for item in raw_corpus]
                if data["corpora"]:
                    data["corpus"] = data["corpora"][0]
            else:
                data["corpus"] = str(raw_corpus)

        # Handle date_windows parsing
        if "filter" in data and isinstance(data["filter"], dict):
            if "date_windows" in data["filter"]:
                data["filter"]["date_windows"] = parse_date_windows(data["filter"]["date_windows"])

        # Handle backward compatibility for rebuild_aggregates (inverted logic)
        if "rebuild_aggregates" in data:
            rebuild = bool(data["rebuild_aggregates"])
            data["reload_aggregates"] = not rebuild

        # Create config instance
        config = cls(**data)

        # Set source path
        try:
            config.source_config_path = path
        except Exception:
            config.source_config_path = None

        return config

    def iter_corpus_names(self):
        """Iterate over corpus names (either from corpora list or single corpus)"""
        if self.corpora:
            for name in self.corpora:
                yield name
        else:
            yield self.corpus


__all__ = [
    "NarrativeFramingConfig",
    "FilterConfig",
    "DocumentFilter",
    "ChunkFilter",
    "ChunkingConfig",
    "InductionConfig",
    "AnnotationConfig",
    "ClassificationConfig",
    "ClassifierSettings",
    "ReportConfig",
    "PlotSettings",
    "CustomPlotSettings",
]
