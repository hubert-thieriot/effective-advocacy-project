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

    # Reload flags
    reload_schema: bool = False
    reload_annotation: bool = False
    reload_classifier: bool = False
    reload_classifications: bool = False


# ---------------------------------------------------------------------------
# Claims Analysis Configuration (replaces Stance for DNA-style analysis)
# ---------------------------------------------------------------------------


class StatementExtractionConfig(BaseModel):
    """Configuration for statement extraction from NER chunks."""

    model: str = "gpt-4o-mini"
    batch_size: int = 10  # Chunks per LLM call
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_statements_per_chunk: int = 5
    min_statement_length: int = 20
    flex_processing: bool = True
    chunk_limit: Optional[int] = None  # Limit number of chunks to process (for testing)


class ClaimSchemaConfig(BaseModel):
    """Configuration for claim schema (induction or file)."""

    source: str = "induction"  # "induction" or "file"
    schema_path: Optional[Path] = None

    # Induction settings
    induction_size: int = 200  # Statements to sample for induction
    induction_model: str = "claude-sonnet-4"
    induction_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    claim_target: str = "10"  # Can be "10" or "8-12 claims"
    induction_guidance: Optional[str] = None  # Seed claims/hints
    induction_batch_size: int = 10
    flex_processing: bool = False

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        v_lower = str(v).strip().lower()
        if v_lower not in {"induction", "file"}:
            return "induction"
        return v_lower

    @field_validator("schema_path")
    @classmethod
    def expand_schema_path(cls, v: Optional[Path]) -> Optional[Path]:
        if v is None:
            return None
        return Path(v).expanduser().resolve()


class AgreementScoringConfig(BaseModel):
    """Configuration for statement-claim agreement scoring."""

    model: str = "gpt-4o-mini"  # Fast model for many calls
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    flex_processing: bool = True
    batch_size: int = Field(default=1, ge=1)  # Statements per LLM call

    # Thresholds for labels
    support_threshold: float = 0.3  # >= 0.3 -> "supports"
    oppose_threshold: float = -0.3  # <= -0.3 -> "opposes"


class DNAConfig(BaseModel):
    """Configuration for Discourse Network Analysis (DNA).

    Controls actor-actor network building and coalition clustering,
    both of which run from a shared actor set for consistency.
    """

    enabled: bool = True
    min_statements: int = 2  # Minimum supports/opposes agreements per actor
    max_actors: int = 50  # Maximum actors to include
    n_clusters_range: Tuple[int, int] = (2, 8)  # K-means N range to test
    layout: str = "spring"  # Network layout: "spring", "community", "kamada_kawai"
    reload_dna: bool = False  # True: load saved results from cache; False: rebuild


class ClaimsAnalysisConfig(BaseModel):
    """Configuration for the Claims Analysis stage.

    This stage:
    1. Extracts statements from chunks with actors (from NER)
    2. Induces/loads claims (specific policy positions)
    3. Scores agreement between statements and claims
    """

    enabled: bool = False
    domain: Optional[str] = None  # E.g., "Grand National horse racing debate"

    statement: StatementExtractionConfig = Field(default_factory=StatementExtractionConfig)
    claims: ClaimSchemaConfig = Field(default_factory=ClaimSchemaConfig)
    scoring: AgreementScoringConfig = Field(default_factory=AgreementScoringConfig)
    dna: DNAConfig = Field(default_factory=DNAConfig)

    # Reload flags
    reload_statements: bool = False
    reload_claims: bool = False
    reload_agreements: bool = False


class NERConsolidationConfig(BaseModel):
    """Configuration for NER consolidation (entity grouping and enrichment)."""

    enabled: bool = False
    model: str = "gpt-4o-mini"  # LLM model for consolidation
    batch_size: int = 50  # Entities per LLM call
    min_count: int = 2  # Only consolidate entities appearing >= min_count times
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    flex_processing: bool = True  # OpenAI flex processing (50% cost reduction)
    guidance: Optional[str] = None  # User-provided context/guidance for LLM
    entity_types: List[str] = Field(
        default_factory=lambda: ["PERSON", "ORG", "WORK_OF_ART", "EVENT", "FAC", "PRODUCT"]
    )


class NERConfig(BaseModel):
    """Configuration for Named Entity Recognition stage."""

    enabled: bool = False
    language: Union[str, List[str]] = "en"  # Stanza language code(s)
    frame_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    entity_types: Optional[List[str]] = None  # Filter to specific types (None = all)
    batch_size: int = 32

    consolidation: NERConsolidationConfig = Field(default_factory=NERConsolidationConfig)
    reload_ner: bool = False

    @field_validator("language")
    @classmethod
    def validate_language(cls, v):
        if v is None:
            return "en"
        if isinstance(v, (list, tuple, set)):
            raw = [str(item).strip().lower() for item in v if str(item).strip()]
            seen = set()
            langs = []
            for lang in raw:
                if lang not in seen:
                    seen.add(lang)
                    langs.append(lang)
            if not langs:
                return "en"
            if len(langs) == 1:
                return langs[0]
            return langs
        text = str(v).strip()
        if not text:
            return "en"
        if "," in text:
            raw = [part.strip().lower() for part in text.split(",") if part.strip()]
            seen = set()
            langs = []
            for lang in raw:
                if lang not in seen:
                    seen.add(lang)
                    langs.append(lang)
            if not langs:
                return "en"
            if len(langs) == 1:
                return langs[0]
            return langs
        return text.lower()


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
    regenerate_report_only: bool = False

    # Shared filters/chunking
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    filter: FilterConfig = Field(default_factory=FilterConfig)
    relevance_keywords: Optional[Union[List[str], Dict[str, List[str]]]] = None

    # Sections
    framing: FramingConfig = Field(default_factory=FramingConfig)
    ner: NERConfig = Field(default_factory=NERConfig)
    claims_analysis: ClaimsAnalysisConfig = Field(default_factory=ClaimsAnalysisConfig)
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
        if self.regenerate_report_only:
            self.framing.reload_schema = True
            self.framing.reload_annotation = True
            self.framing.reload_classifier = True
            self.framing.reload_classifications = True
            self.ner.reload_ner = True
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
    "NERConfig",
    "NERConsolidationConfig",
    "ClaimsAnalysisConfig",
    "StatementExtractionConfig",
    "ClaimSchemaConfig",
    "AgreementScoringConfig",
    "DNAConfig",
    "ReportConfig",
    "FilterConfig",
    "ChunkingConfig",
    "ClassifierSettings",
    "ClassificationConfig",
]
