"""
Pydantic configuration models for the discourse analysis application.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator
import yaml

from apps.narrative_framing.config_models import (
    FilterConfig,
    ChunkingConfig,
    ClassifierSettings,
    ClassificationConfig,
    normalize_keyword_dict,
    parse_date_windows,
)


class FramingSchemaConfig(BaseModel):
    source: str = "induction"  # "induction" or "file"
    schema_path: Optional[Path] = None
    induction_size: int = 100
    induction_model: str = "claude-sonnet-4"
    induction_temperature: Optional[float] = Field(default=0.0, ge=0.0, le=2.0)
    frame_target: str = "10"
    induction_guidance: Optional[str] = None
    induction_batch_size: Optional[int] = None
    flex_processing: Optional[bool] = False

    @field_validator("schema_path")
    @classmethod
    def expand_schema_path(cls, v: Optional[Path]) -> Optional[Path]:
        if v is None:
            return None
        return Path(v).expanduser().resolve()

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        v_lower = str(v).strip().lower()
        if v_lower not in {"induction", "file"}:
            return "induction"
        return v_lower


class FramingAnnotationConfig(BaseModel):
    size: int = 500
    model: str = "claude-sonnet-4"
    batch_size: int = 5
    top_k: int = 3
    temperature: Optional[float] = Field(default=0.0, ge=0.0, le=2.0)
    force_zero_if_no_keywords: Optional[Union[List[str], Dict[str, List[str]]]] = None
    guidance: Optional[str] = None
    flex_processing: Optional[bool] = False

    @field_validator("force_zero_if_no_keywords")
    @classmethod
    def normalize_keywords(cls, v):
        return normalize_keyword_dict(v)


class FramingConfig(BaseModel):
    enabled: bool = True
    domain: Optional[str] = None
    schema: FramingSchemaConfig = Field(default_factory=FramingSchemaConfig)
    annotation: FramingAnnotationConfig = Field(default_factory=FramingAnnotationConfig)
    classifier: ClassifierSettings = Field(default_factory=ClassifierSettings)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)


class StanceAnnotationConfig(BaseModel):
    size: int = 500
    model: str = "claude-sonnet-4"
    batch_size: int = 5
    temperature: Optional[float] = Field(default=0.0, ge=0.0, le=2.0)
    flex_processing: Optional[bool] = False


class StanceConfig(BaseModel):
    enabled: bool = True
    mode: str = "zero_shot"  # "zero_shot" or "trained"
    targets: List[str] = Field(default_factory=list)
    zero_shot_model: str = "facebook/bart-large-mnli"
    chunk_limit: Optional[int] = None
    annotation: StanceAnnotationConfig = Field(default_factory=StanceAnnotationConfig)
    classifier: ClassifierSettings = Field(default_factory=ClassifierSettings)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)

    reload_annotation: bool = False
    reload_classifier: bool = False
    reload_classifications: bool = False

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        v_lower = str(v).strip().lower()
        if v_lower not in {"zero_shot", "trained"}:
            return "zero_shot"
        return v_lower

    @field_validator("chunk_limit")
    @classmethod
    def validate_chunk_limit(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        return v if v > 0 else None


class ReportConfig(BaseModel):
    title: str = "Discourse Analysis Report"
    subtitle: Optional[str] = None


class DiscourseAnalysisConfig(BaseModel):
    # Core paths/settings
    corpus: str
    corpora: Optional[List[str]] = None
    corpora_root: Path = Path.home() / "corpora"
    workspace_root: Path = Path.home() / "workspace"
    results_dir: Optional[Path] = None

    # Seed
    seed: int = 42

    # Reload flags
    reload_framing_schema: bool = False
    reload_framing_annotation: bool = False
    reload_framing_classifier: bool = False
    reload_framing_classifications: bool = False
    reload_stance: bool = False
    regenerate_report_only: bool = False

    # Shared filters/chunking
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    filter: FilterConfig = Field(default_factory=FilterConfig)
    relevance_keywords: Optional[Union[List[str], Dict[str, List[str]]]] = None

    # Sections
    framing: FramingConfig = Field(default_factory=FramingConfig)
    stance: StanceConfig = Field(default_factory=StanceConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)

    source_config_path: Optional[Path] = None

    @field_validator("corpora_root", "workspace_root", "results_dir")
    @classmethod
    def expand_paths(cls, v: Optional[Path]) -> Optional[Path]:
        if v is None:
            return None
        return Path(v).expanduser().resolve()

    @field_validator("relevance_keywords")
    @classmethod
    def normalize_keywords(cls, v):
        return normalize_keyword_dict(v)

    @field_validator("corpora")
    @classmethod
    def normalize_corpora(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return None
        normalized = [str(item).strip() for item in v if str(item).strip()]
        return normalized or None

    @model_validator(mode="after")
    def handle_reload_report_only(self) -> "DiscourseAnalysisConfig":
        if self.reload_stance:
            self.stance.reload_annotation = True
            self.stance.reload_classifier = True
            self.stance.reload_classifications = True
        if self.regenerate_report_only:
            self.reload_framing_schema = True
            self.reload_framing_annotation = True
            self.reload_framing_classifier = True
            self.reload_framing_classifications = True
            self.stance.reload_annotation = True
            self.stance.reload_classifier = True
            self.stance.reload_classifications = True
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> "DiscourseAnalysisConfig":
        if not path.exists():
            return cls(corpus="default")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if "corpus" in data:
            raw_corpus = data["corpus"]
            if isinstance(raw_corpus, (list, tuple)):
                data["corpora"] = [str(item) for item in raw_corpus]
                if data["corpora"]:
                    data["corpus"] = data["corpora"][0]
            else:
                data["corpus"] = str(raw_corpus)

        if "filter" in data and isinstance(data["filter"], dict):
            if "date_windows" in data["filter"]:
                data["filter"]["date_windows"] = parse_date_windows(data["filter"]["date_windows"])

        config = cls(**data)
        try:
            config.source_config_path = path
        except Exception:
            config.source_config_path = None
        return config

    def iter_corpus_names(self):
        if self.corpora:
            for name in self.corpora:
                yield name
        else:
            yield self.corpus


__all__ = [
    "DiscourseAnalysisConfig",
    "FramingConfig",
    "FramingSchemaConfig",
    "FramingAnnotationConfig",
    "StanceConfig",
    "StanceAnnotationConfig",
    "ReportConfig",
    "FilterConfig",
    "ChunkingConfig",
    "ClassifierSettings",
    "ClassificationConfig",
]
