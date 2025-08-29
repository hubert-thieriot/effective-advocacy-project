"""
Shared domain types and model specifications.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from datetime import datetime
import json
import hashlib


@dataclass
class Document:
    """Canonical document type used across corpus and analyzer layers.

    Matches the existing efi_corpus Document shape for compatibility.
    """
    doc_id: str
    url: str
    title: Optional[str]
    text: str
    published_at: Optional[str]
    language: Optional[str]
    meta: Dict[str, Any]


@dataclass
class Finding:
    """A single key finding extracted from a document or source."""
    text: str
    finding_id: Optional[str] = None
    published_at: Optional[str] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = None
    confidence: Optional[float] = None
    category: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    
    @classmethod
    def generate_doc_id(cls, url: str) -> str:
        """Generate a unique document ID from URL"""
        return f"{hash(url) % 1000000:06d}"
    
    @classmethod
    def generate_id(cls, url: str, index: int) -> str:
        """Generate a unique finding ID from URL and index"""
        doc_id = cls.generate_doc_id(url)
        return f"{doc_id}_{index:03d}"
    
    @classmethod
    def extract_doc_id_from_finding_id(cls, finding_id: str) -> str:
        """Extract document ID from finding ID (first part before underscore)"""
        if '_' in finding_id:
            return finding_id.split('_')[0]
        return finding_id


@dataclass
class Chunk:
    """Represents a text chunk with metadata"""
    chunk_id: int
    text: str
    doc_id: str


@dataclass
class LibraryDocument:
    """Library document metadata without findings"""
    doc_id: str
    url: str
    title: Optional[str] = None
    published_at: Optional[datetime] = None
    language: Optional[str] = None
    extraction_date: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Ensure published_at is always a datetime object or None
        from .utils import normalize_date
        self.published_at = normalize_date(self.published_at)


@dataclass
class LibraryDocumentWFindings(LibraryDocument):
    """Library document with findings included"""
    findings: List[Finding] = field(default_factory=list)





@dataclass(frozen=True)
class ChunkerSpec:
    name: str
    params: Dict[str, Any]

    def key(self) -> str:
        s = json.dumps({"name": self.name, "params": self.params}, sort_keys=True)
        return hashlib.sha1(s.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class EmbedderSpec:
    model_name: str
    dim: int
    revision: Optional[str] = None

    def key(self) -> str:
        s = json.dumps({"model": self.model_name, "dim": self.dim, "rev": self.revision}, sort_keys=True)
        return hashlib.sha1(s.encode()).hexdigest()[:12]


@dataclass
class DocState:
    """Tracks the processing state of a document for incremental rebuilds."""
    document_id: str
    fingerprint: str
    last_built_ts: float
    chunker_key: str
    embedder_key: str


@dataclass
class FindingState:
    """Tracks the processing state of a finding embedding for incremental rebuilds."""
    finding_id: str
    fingerprint: str
    last_built_ts: float
    embedder_key: str


# Document Matching Pipeline Types
@dataclass
class FindingFilters:
    """Configurable filters for findings."""
    include_keywords: List[str] = field(default_factory=list)
    exclude_keywords: List[str] = field(default_factory=list)
    date_range: Optional[Tuple[datetime, datetime]] = None
    confidence_threshold: Optional[float] = None


@dataclass
class DocumentMatch:
    """A single document match."""
    chunk_id: str
    chunk_text: str
    cosine_score: float
    rescorer_scores: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class FindingResults:
    """Results for a single finding."""
    finding_id: str
    finding_text: str
    matches: List[DocumentMatch]
    rescorer_scores: Dict[str, List[float]]
    timing: Dict[str, float]


@dataclass
class DocumentMatchingResults:
    """Container for all matching results."""
    findings_processed: int
    total_matches: int
    results_by_finding: Dict[str, FindingResults]
    timing_stats: Dict[str, Any]
    score_stats: Dict[str, Any]
    metadata: Dict[str, Any]


