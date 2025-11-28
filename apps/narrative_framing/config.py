"""Configuration helpers for the narrative framing application."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import yaml

from efi_analyser.frames.classifier import FrameClassifierSpec

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
class ClassifierSettings(FrameClassifierSpec):
    """Application-level classifier settings.

    Inherits all hyper-parameters from FrameClassifierSpec and adds only
    app-specific flags such as `enabled` and `cv_folds`.
    """

    enabled: bool = False
    cv_folds: Optional[int] = None


@dataclass
class PlotSettings:
    """Configuration for plot titles and descriptions."""
    title: Optional[str] = None
    subtitle: Optional[str] = None
    note: Optional[str] = None

@dataclass
class CustomPlotSettings:
    """Configuration for a custom plot in the report."""
    type: str  # e.g., "global_occurrence_with_zeros", "year_occurrence_with_zeros"
    title: Optional[str] = None
    subtitle: Optional[str] = None
    caption: Optional[str] = None

@dataclass
class ReportSettings:
    """Configuration for report generation and display."""
    plot: PlotSettings = field(default_factory=PlotSettings)
    hide_empty_passages: bool = False
    custom_plots: Optional[List[CustomPlotSettings]] = None
    export_plots_dir: Optional[Path] = None  # Optional directory to export plots to (in addition to results/plots)
    export_includes_dir: Optional[Path] = None  # Optional directory to export Jekyll includes (e.g., docs/_includes/narrative_framing)
    export_plot_formats: List[str] = field(default_factory=lambda: ["png"])  # List of formats to export: ["png"], ["svg"], ["png", "svg"], etc. HTML is always exported.
    n_min_per_media: Optional[int] = None  # Minimum number of articles per media/domain to show in domain mapping
    domain_mapping_max_domains: int = 20  # Maximum number of domains to show in domain mapping chart
    include_yearly_bar_charts: bool = True  # Whether to include yearly bar charts in the aggregation section
    include_domain_yearly_bar_charts: bool = False  # Whether to include per-domain yearly bar charts
    domain_yearly_top_domains: int = 5  # Number of top domains (by article count) to display in yearly charts


@dataclass
class DocumentFilter:
    """Document-level filtering configuration."""
    keywords: Optional[List[str]] = None  # Documents must contain at least one chunk with these keywords
    trim_after_markers: Optional[List[str]] = None  # Trim document text after these markers
    exclude_min_hits: Optional[Dict[str, int]] = None  # Exclude documents with too many hits of certain phrases
    exclude_regex: Optional[List[str]] = None  # Exclude documents matching these regex patterns
    domain_whitelist: Optional[List[str]] = None  # Only include documents from these domains (extracted from URL)


@dataclass
class ChunkFilter:
    """Chunk-level filtering configuration."""
    keywords: Optional[List[str]] = None  # Chunks must contain these keywords
    exclude_regex: Optional[List[str]] = None  # Exclude chunks matching these regex patterns
    exclude_min_hits: Optional[Dict[str, int]] = None  # Exclude chunks with too many hits of certain phrases
    trim_after_markers: Optional[List[str]] = None  # Trim chunk text after these markers


@dataclass
class FilterConfig:
    """Comprehensive filtering configuration for documents and chunks."""
    date_from: Optional[str] = None  # Only consider documents on/after this date (YYYY-MM-DD)
    document: DocumentFilter = field(default_factory=DocumentFilter)
    chunk: ChunkFilter = field(default_factory=ChunkFilter)




@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    target_words: int = DEFAULT_TARGET_WORDS
    chunker_model: str = "en_core_web_sm"  # spaCy model for text chunking


@dataclass
class InductionConfig:
    """Configuration for frame induction."""
    size: int = DEFAULT_INDUCTION_SAMPLE
    model: str = DEFAULT_INDUCTION_MODEL
    frame_target: str = "between 5 and 10"
    temperature: Optional[float] = None  # None = use model default
    guidance: Optional[str] = None
    batch_size: Optional[int] = None  # Max passages per LLM call (default: auto-sized)


@dataclass
class AnnotationConfig:
    """Configuration for frame annotation/application."""
    size: int = DEFAULT_APPLICATION_SAMPLE
    model: str = DEFAULT_APPLICATION_MODEL
    batch_size: int = DEFAULT_APPLICATION_BATCH
    top_k: int = DEFAULT_APPLICATION_TOP_K
    temperature: Optional[float] = None  # None = use model default
    # Bypass LLM with zero scores if chunk lacks these keywords.
    # Can be a flat list (applies to all languages) or dict by language code:
    #   force_zero_if_no_keywords: [animal, welfare, ...]  # flat
    #   force_zero_if_no_keywords:  # by language
    #     en: [animal, welfare, ...]
    #     de: [tier, tierschutz, ...]
    force_zero_if_no_keywords: Optional[Union[List[str], Dict[str, List[str]]]] = None
    guidance: Optional[str] = None  # Additional guidance for the annotator to reduce false positives


@dataclass
class ClassificationConfig:
    """Configuration for classifier-based classification."""
    size: Optional[int] = None  # None = use all available documents


@dataclass
class NarrativeFramingConfig:
    """Typed configuration for the narrative framing workflow."""

    # For provenance: original YAML file path (set at load time)
    source_config_path: Optional[Path] = None

    corpus: str = None
    corpora: Optional[List[str]] = None
    corpora_root: Path = DEFAULT_CORPORA_ROOT
    workspace_root: Path = DEFAULT_WORKSPACE_ROOT
    domain: str = "India coal media coverage"
    seed: int = DEFAULT_SEED
    
    # Reload settings (at top level as requested)
    reload_induction: bool = False
    reload_annotation: bool = False  # Renamed from reload_application
    reload_classifier: bool = False
    reload_classifications: bool = False
    reload_aggregates: bool = True
    
    # Top-level relevance keywords (used for both induction sampling and annotation bypass).
    # Can be a flat list (applies to all languages) or dict by language code:
    #   relevance_keywords: [animal, welfare, ...]  # flat
    #   relevance_keywords:  # by language
    #     english: [animal, welfare, ...]
    #     german: [tier, tierschutz, ...]
    relevance_keywords: Optional[Union[List[str], Dict[str, List[str]]]] = None
    
    # Nested configuration sections
    filter: FilterConfig = field(default_factory=FilterConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    induction: InductionConfig = field(default_factory=InductionConfig)
    annotation: AnnotationConfig = field(default_factory=AnnotationConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    
    results_dir: Optional[Path] = None
    classifier: ClassifierSettings = field(default_factory=ClassifierSettings)
    report: ReportSettings = field(default_factory=ReportSettings)
    
    # Legacy fields (deprecated, kept for backward compatibility)
    llm_model: str = DEFAULT_LLM_MODEL  # Deprecated: use induction.model and annotation.model
    induction_model: str = None  # Deprecated: use induction.model
    application_model: str = None  # Deprecated: use annotation.model
    induction_frame_target: str = None  # Deprecated: use induction.frame_target
    induction_temperature: Optional[float] = None  # Deprecated: use induction.temperature
    application_temperature: Optional[float] = None  # Deprecated: use annotation.temperature
    filter_keywords: Optional[List[str]] = None  # Deprecated: use filter.document.keywords
    filter_exclude_regex: Optional[List[str]] = None  # Deprecated: use filter.chunk.exclude_regex
    filter_exclude_min_hits: Optional[Dict[str, int]] = None  # Deprecated: use filter.chunk.exclude_min_hits
    filter_trim_after_markers: Optional[List[str]] = None  # Deprecated: use filter.chunk.trim_after_markers
    date_from: Optional[str] = None  # Deprecated: use filter.date_from
    target_words: int = None  # Deprecated: use chunking.target_words
    chunker_model: str = None  # Deprecated: use chunking.chunker_model
    induction_sample_size: int = None  # Deprecated: use induction.size
    application_sample_size: int = None  # Deprecated: use annotation.size
    application_batch_size: int = None  # Deprecated: use annotation.batch_size
    application_top_k: int = None  # Deprecated: use annotation.top_k
    reload_results: bool = False  # Deprecated
    reload_application: bool = None  # Deprecated: use reload_annotation
    induction_guidance: Optional[str] = None  # Deprecated: use induction.guidance
    classifier_corpus_sample_size: Optional[int] = None  # Deprecated: use classification.size
    sampling_policy: str = "equal"
    # Optional mapping to display human-friendly labels for corpora in reports
    corpus_aliases: Optional[Dict[str, str]] = None
    # Utility: when true, skip analysis and rebuild report from cached artifacts only
    regenerate_report_only: bool = False
    # Per-step controls for creating new work beyond cached artifacts
    allow_new_induction: bool = True
    allow_new_annotation: bool = True
    allow_new_training: bool = True
    allow_new_classification: bool = True
    allow_new_aggregation: bool = True
    # Aggregation controls
    agg_min_threshold_weighted: float = 0.2
    agg_normalize_weighted: bool = True
    agg_min_threshold_occurrence: float = 0.2

    def normalize(self) -> None:
        """Normalize derived fields after loading from disk."""
        
        # Normalize filter.date_from
        if self.filter.date_from is not None:
            self.filter.date_from = str(self.filter.date_from).strip() or None
        
        # Normalize filter.document fields
        if self.filter.document.keywords is not None:
            normalized = [kw.strip() for kw in self.filter.document.keywords if kw and kw.strip()]
            self.filter.document.keywords = normalized or None
        if self.filter.document.exclude_regex is not None:
            normalized = [str(p).strip() for p in self.filter.document.exclude_regex if str(p).strip()]
            self.filter.document.exclude_regex = normalized or None
        if self.filter.document.exclude_min_hits is not None:
            cleaned: Dict[str, int] = {}
            for k, v in self.filter.document.exclude_min_hits.items():
                key = str(k).strip()
                try:
                    val = int(v)
                except Exception:
                    continue
                if key and val >= 1:
                    cleaned[key] = val
            self.filter.document.exclude_min_hits = cleaned or None
        if self.filter.document.trim_after_markers is not None:
            normalized = [str(m).strip() for m in self.filter.document.trim_after_markers if str(m).strip()]
            self.filter.document.trim_after_markers = normalized or None
        
        # Normalize filter.chunk fields
        if self.filter.chunk.keywords is not None:
            normalized = [kw.strip() for kw in self.filter.chunk.keywords if kw and kw.strip()]
            self.filter.chunk.keywords = normalized or None
        if self.filter.chunk.exclude_regex is not None:
            normalized = [str(p).strip() for p in self.filter.chunk.exclude_regex if str(p).strip()]
            self.filter.chunk.exclude_regex = normalized or None
        if self.filter.chunk.exclude_min_hits is not None:
            cleaned: Dict[str, int] = {}
            for k, v in self.filter.chunk.exclude_min_hits.items():
                key = str(k).strip()
                try:
                    val = int(v)
                except Exception:
                    continue
                if key and val >= 1:
                    cleaned[key] = val
            self.filter.chunk.exclude_min_hits = cleaned or None
        if self.filter.chunk.trim_after_markers is not None:
            normalized = [str(m).strip() for m in self.filter.chunk.trim_after_markers if str(m).strip()]
            self.filter.chunk.trim_after_markers = normalized or None
        
        # Normalize induction.guidance
        if self.induction.guidance is not None:
            stripped = self.induction.guidance.strip()
            self.induction.guidance = stripped if stripped else None
        # Normalize classification.size
        if self.classification.size is not None and self.classification.size <= 0:
            self.classification.size = None
        if self.corpora is not None:
            self.corpora = [str(item).strip() for item in self.corpora if str(item).strip()]
            if not self.corpora:
                self.corpora = None
        # Normalize corpus_aliases mapping (strip keys/values)
        if self.corpus_aliases:
            cleaned_aliases: Dict[str, str] = {}
            for k, v in self.corpus_aliases.items():
                key = str(k).strip()
                val = str(v).strip()
                if key and val:
                    cleaned_aliases[key] = val
            self.corpus_aliases = cleaned_aliases or None
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
    # Allow older configs to specify inference_batch_size as a synonym
    if "batch_size" in data:
        settings.batch_size = int(data["batch_size"])
    elif "inference_batch_size" in data:
        settings.batch_size = int(data["inference_batch_size"])
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
    if "cv_folds" in data:
        try:
            v = int(data["cv_folds"]) if data["cv_folds"] is not None else None
        except Exception:
            v = None
        settings.cv_folds = v if (v is None or v >= 2) else None
    return settings


def load_config(path: Path) -> NarrativeFramingConfig:
    """Load configuration from a YAML file."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    data: Dict[str, Any] = payload or {}

    config = NarrativeFramingConfig()
    # Record provenance of the config file for later snapshotting
    try:
        config.source_config_path = path
    except Exception:
        config.source_config_path = None
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
    
    # Parse reload settings (at top level)
    if "reload_induction" in data:
        config.reload_induction = bool(data["reload_induction"])
    if "reload_annotation" in data:
        config.reload_annotation = bool(data["reload_annotation"])
    if "reload_classifier" in data:
        config.reload_classifier = bool(data["reload_classifier"])
    if "reload_classifications" in data:
        config.reload_classifications = bool(data["reload_classifications"])
    if "reload_aggregates" in data:
        config.reload_aggregates = bool(data["reload_aggregates"])
    
    # Parse top-level relevance_keywords (used for both induction and annotation)
    if "relevance_keywords" in data:
        keywords = data["relevance_keywords"]
        if keywords is None:
            config.relevance_keywords = None
        elif isinstance(keywords, dict):
            # Dict by language code: {english: [...], german: [...], ...}
            config.relevance_keywords = {
                lang: [str(item).lower().strip() for item in kw_list if item]
                for lang, kw_list in keywords.items()
                if kw_list
            }
        else:
            # Flat list (language-agnostic)
            config.relevance_keywords = [str(item).lower().strip() for item in keywords if item]
    
    # Parse nested chunking config
    if "chunking" in data and isinstance(data["chunking"], dict):
        chunking_data = data["chunking"]
        if "target_words" in chunking_data:
            config.chunking.target_words = int(chunking_data["target_words"])
        if "chunker_model" in chunking_data:
            config.chunking.chunker_model = str(chunking_data["chunker_model"])
    
    # Parse nested induction config
    if "induction" in data and isinstance(data["induction"], dict):
        induction_data = data["induction"]
        if "size" in induction_data:
            config.induction.size = int(induction_data["size"])
        if "model" in induction_data:
            config.induction.model = str(induction_data["model"])
        if "frame_target" in induction_data:
            raw = induction_data["frame_target"]
            config.induction.frame_target = str(raw) if raw is not None else config.induction.frame_target
        if "temperature" in induction_data:
            config.induction.temperature = float(induction_data["temperature"]) if induction_data["temperature"] is not None else None
        if "guidance" in induction_data:
            config.induction.guidance = str(induction_data["guidance"]).strip() if induction_data["guidance"] else None
        if "batch_size" in induction_data:
            config.induction.batch_size = int(induction_data["batch_size"]) if induction_data["batch_size"] else None
    
    # Parse nested annotation config
    if "annotation" in data and isinstance(data["annotation"], dict):
        annotation_data = data["annotation"]
        if "size" in annotation_data:
            config.annotation.size = int(annotation_data["size"])
        if "model" in annotation_data:
            config.annotation.model = str(annotation_data["model"])
        if "batch_size" in annotation_data:
            config.annotation.batch_size = int(annotation_data["batch_size"])
        if "top_k" in annotation_data:
            config.annotation.top_k = int(annotation_data["top_k"])
        if "temperature" in annotation_data:
            config.annotation.temperature = float(annotation_data["temperature"]) if annotation_data["temperature"] is not None else None
        if "force_zero_if_no_keywords" in annotation_data:
            keywords = annotation_data["force_zero_if_no_keywords"]
            if keywords is None:
                config.annotation.force_zero_if_no_keywords = None
            elif isinstance(keywords, dict):
                # Dict by language code: {en: [...], de: [...], ...}
                config.annotation.force_zero_if_no_keywords = {
                    lang: [str(item).lower().strip() for item in kw_list if item]
                    for lang, kw_list in keywords.items()
                    if kw_list
                }
            else:
                # Flat list (language-agnostic)
                config.annotation.force_zero_if_no_keywords = [str(item).lower().strip() for item in keywords if item]
        if "guidance" in annotation_data:
            config.annotation.guidance = str(annotation_data["guidance"]).strip() if annotation_data["guidance"] else None
    
    # Parse nested classification config
    if "classification" in data and isinstance(data["classification"], dict):
        classification_data = data["classification"]
        if "size" in classification_data:
            val = classification_data["size"]
            config.classification.size = int(val) if val is not None else None
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
    
    # Parse new filter structure
    if "filter" in data:
        filter_data = data["filter"]
        if isinstance(filter_data, dict):
            # Parse date_from
            if "date_from" in filter_data:
                config.filter.date_from = str(filter_data["date_from"]).strip() if filter_data["date_from"] else None
            
            # Parse document filter
            if "document" in filter_data and isinstance(filter_data["document"], dict):
                doc_filter = filter_data["document"]
                if "keywords" in doc_filter:
                    keywords = doc_filter["keywords"]
                    if keywords is None:
                        config.filter.document.keywords = None
                    else:
                        config.filter.document.keywords = [str(item) for item in keywords]
                if "domain_whitelist" in doc_filter:
                    domains = doc_filter["domain_whitelist"]
                    if domains is None:
                        config.filter.document.domain_whitelist = None
                    else:
                        config.filter.document.domain_whitelist = [str(item).lower().strip() for item in domains]
                if "exclude_regex" in doc_filter:
                    patterns = doc_filter["exclude_regex"]
                    if patterns is None:
                        config.filter.document.exclude_regex = None
                    else:
                        config.filter.document.exclude_regex = [str(item) for item in patterns]
                if "exclude_min_hits" in doc_filter:
                    mapping = doc_filter["exclude_min_hits"] or {}
                    if isinstance(mapping, dict):
                        parsed: Dict[str, int] = {}
                        for k, v in mapping.items():
                            try:
                                parsed[str(k)] = int(v)
                            except Exception:
                                continue
                        config.filter.document.exclude_min_hits = parsed
                if "trim_after_markers" in doc_filter:
                    markers = doc_filter["trim_after_markers"]
                    if markers is None:
                        config.filter.document.trim_after_markers = None
                    else:
                        config.filter.document.trim_after_markers = [str(item) for item in markers]
            
            # Parse chunk filter
            if "chunk" in filter_data and isinstance(filter_data["chunk"], dict):
                chunk_filter = filter_data["chunk"]
                if "keywords" in chunk_filter:
                    keywords = chunk_filter["keywords"]
                    if keywords is None:
                        config.filter.chunk.keywords = None
                    else:
                        config.filter.chunk.keywords = [str(item) for item in keywords]
                if "exclude_regex" in chunk_filter:
                    patterns = chunk_filter["exclude_regex"]
                    if patterns is None:
                        config.filter.chunk.exclude_regex = None
                    else:
                        config.filter.chunk.exclude_regex = [str(item) for item in patterns]
                if "exclude_min_hits" in chunk_filter:
                    mapping = chunk_filter["exclude_min_hits"] or {}
                    if isinstance(mapping, dict):
                        parsed: Dict[str, int] = {}
                        for k, v in mapping.items():
                            try:
                                parsed[str(k)] = int(v)
                            except Exception:
                                continue
                        config.filter.chunk.exclude_min_hits = parsed
                if "trim_after_markers" in chunk_filter:
                    markers = chunk_filter["trim_after_markers"]
                    if markers is None:
                        config.filter.chunk.trim_after_markers = None
                    else:
                        config.filter.chunk.trim_after_markers = [str(item) for item in markers]
    
    if "target_words" in data:
        config.target_words = int(data["target_words"])
    if "chunker_model" in data:
        config.chunker_model = str(data["chunker_model"])
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
    if "reload_classifications" in data:
        config.reload_classifications = bool(data["reload_classifications"])
    if "results_dir" in data:
        config.results_dir = _as_path(data["results_dir"])
    if "regenerate_report_only" in data:
        try:
            config.regenerate_report_only = bool(data["regenerate_report_only"])  # type: ignore[attr-defined]
        except Exception:
            config.regenerate_report_only = False
    if "reload_aggregates" in data:
        try:
            config.reload_aggregates = bool(data["reload_aggregates"])  # type: ignore[attr-defined]
        except Exception:
            config.reload_aggregates = True
    # Backward compatibility: rebuild_aggregates (inverted logic)
    if "rebuild_aggregates" in data:
        try:
            rebuild = bool(data["rebuild_aggregates"])
            config.reload_aggregates = not rebuild  # type: ignore[attr-defined]
        except Exception:
            pass
    # Allow toggling per-step creation of new work
    if "allow_new_induction" in data:
        try:
            config.allow_new_induction = bool(data["allow_new_induction"])
        except Exception:
            pass
    if "allow_new_annotation" in data:
        try:
            config.allow_new_annotation = bool(data["allow_new_annotation"])
        except Exception:
            pass
    if "allow_new_training" in data:
        try:
            config.allow_new_training = bool(data["allow_new_training"])
        except Exception:
            pass
    if "allow_new_classification" in data:
        try:
            config.allow_new_classification = bool(data["allow_new_classification"])
        except Exception:
            pass
    if "allow_new_aggregation" in data:
        try:
            config.allow_new_aggregation = bool(data["allow_new_aggregation"])
        except Exception:
            pass
    # Back-compat: allow top-level hide_empty_passages to set report flag
    if "hide_empty_passages" in data:
        try:
            config.report.hide_empty_passages = bool(data["hide_empty_passages"])  # prefer report-scoped flag
        except Exception:
            pass
    if "agg_min_threshold_weighted" in data:
        try:
            config.agg_min_threshold_weighted = float(data["agg_min_threshold_weighted"])  # type: ignore[attr-defined]
        except Exception:
            pass
    if "agg_normalize_weighted" in data:
        try:
            config.agg_normalize_weighted = bool(data["agg_normalize_weighted"])  # type: ignore[attr-defined]
        except Exception:
            pass
    if "agg_min_threshold_occurrence" in data:
        try:
            config.agg_min_threshold_occurrence = float(data["agg_min_threshold_occurrence"])  # type: ignore[attr-defined]
        except Exception:
            pass

    if "classifier" in data:
        config.classifier = _load_classifier_settings(data["classifier"])

    if "induction_guidance" in data:
        config.induction_guidance = str(data["induction_guidance"])
    if "classifier_corpus_sample_size" in data:
        value = data["classifier_corpus_sample_size"]
        config.classifier_corpus_sample_size = int(value) if value is not None else None
    if "sampling_policy" in data:
        config.sampling_policy = str(data["sampling_policy"]).strip().lower()
    if "corpus_aliases" in data and isinstance(data["corpus_aliases"], dict):
        # Map from corpus folder name to human-friendly alias for charts
        config.corpus_aliases = {str(k): str(v) for k, v in data["corpus_aliases"].items()}
    if "date_from" in data:
        val = data["date_from"]
        config.date_from = str(val).strip() if val is not None else None

    if config.reload_results:
        if "reload_induction" not in data:
            config.reload_induction = True
        if "reload_application" not in data:
            config.reload_application = True
        if "reload_classifier" not in data:
            config.reload_classifier = True
        if "reload_classifications" not in data:
            config.reload_classifications = True

    # Handle report section
    if "report" in data:
        report_data = data["report"]
        if isinstance(report_data, dict):
            if "hide_empty_passages" in report_data:
                # Report-level override takes precedence over top-level
                config.report.hide_empty_passages = bool(report_data["hide_empty_passages"])
            
            # Handle plot settings
            if "plot" in report_data and isinstance(report_data["plot"], dict):
                plot_data = report_data["plot"]
                if "title" in plot_data:
                    config.report.plot.title = str(plot_data["title"]) if plot_data["title"] else None
                if "subtitle" in plot_data:
                    config.report.plot.subtitle = str(plot_data["subtitle"]) if plot_data["subtitle"] else None
                if "note" in plot_data:
                    config.report.plot.note = str(plot_data["note"]) if plot_data["note"] else None
            
            # Handle custom plots
            if "custom_plots" in report_data and isinstance(report_data["custom_plots"], list):
                custom_plots_list = []
                for plot_item in report_data["custom_plots"]:
                    if isinstance(plot_item, dict) and "type" in plot_item:
                        custom_plot = CustomPlotSettings(
                            type=str(plot_item["type"]),
                            title=str(plot_item["title"]) if plot_item.get("title") else None,
                            subtitle=str(plot_item["subtitle"]) if plot_item.get("subtitle") else None,
                            caption=str(plot_item["caption"]) if plot_item.get("caption") else None,
                        )
                        custom_plots_list.append(custom_plot)
                if custom_plots_list:
                    config.report.custom_plots = custom_plots_list
            
            # Handle export_plots_dir
            if "export_plots_dir" in report_data:
                config.report.export_plots_dir = _as_path(report_data["export_plots_dir"])
            # Handle export_includes_dir
            if "export_includes_dir" in report_data:
                config.report.export_includes_dir = _as_path(report_data["export_includes_dir"])
            # Handle export_plot_formats
            if "export_plot_formats" in report_data:
                formats = report_data["export_plot_formats"]
                if isinstance(formats, (list, tuple)):
                    config.report.export_plot_formats = [str(f).strip().lower() for f in formats if str(f).strip()]
                elif isinstance(formats, str):
                    # Allow comma-separated string
                    config.report.export_plot_formats = [f.strip().lower() for f in formats.split(",") if f.strip()]
                else:
                    config.report.export_plot_formats = ["png"]  # Default fallback
            # Handle n_min_per_media
            if "n_min_per_media" in report_data:
                try:
                    config.report.n_min_per_media = int(report_data["n_min_per_media"]) if report_data["n_min_per_media"] is not None else None
                except Exception:
                    config.report.n_min_per_media = None
            # Handle domain_mapping_max_domains
            if "domain_mapping_max_domains" in report_data:
                try:
                    config.report.domain_mapping_max_domains = int(report_data["domain_mapping_max_domains"]) if report_data["domain_mapping_max_domains"] is not None else 20
                except Exception:
                    config.report.domain_mapping_max_domains = 20
            # Handle include_domain_yearly_bar_charts
            if "include_domain_yearly_bar_charts" in report_data:
                try:
                    config.report.include_domain_yearly_bar_charts = bool(report_data["include_domain_yearly_bar_charts"])
                except Exception:
                    config.report.include_domain_yearly_bar_charts = False
            # Handle domain_yearly_top_domains
            if "domain_yearly_top_domains" in report_data:
                try:
                    value = int(report_data["domain_yearly_top_domains"])
                except Exception:
                    value = config.report.domain_yearly_top_domains
                config.report.domain_yearly_top_domains = max(0, value)
            # Handle include_yearly_bar_charts
            if "include_yearly_bar_charts" in report_data:
                try:
                    config.report.include_yearly_bar_charts = bool(report_data["include_yearly_bar_charts"])
                except Exception:
                    config.report.include_yearly_bar_charts = True

    config.normalize()
    return config


__all__ = [
    "ClassifierSettings",
    "NarrativeFramingConfig",
    "load_config",
]
