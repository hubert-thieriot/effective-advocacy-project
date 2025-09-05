"""
Shared domain types and model specifications.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
from datetime import datetime
import json
import hashlib
import numpy as np


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


# Scoring and Reranking Types

class Task(Enum):
    """Types of scoring tasks."""
    NLI = "nli"
    STANCE = "stance"


class LabelSpace(Enum):
    """Canonical label spaces for different tasks."""
    NLI = "entails|contradicts|neutral"
    STANCE = "pro|anti|neutral|uncertain"


@dataclass
class Candidate:
    """A candidate result from retrieval, enriched with scores and utilities."""
    item_id: str                    # e.g. f"{doc_id}:{chunk_id}"
    ann_score: float                # raw retrieval score (cosine/BM25)
    text: str                       # passage text
    meta: Dict[str, Any] = field(default_factory=dict)    # doc_id, offsets, etc.
    scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # e.g. scores["nli_roberta"] = {"entails": 0.73, "contradicts": 0.06, "neutral": 0.21}
    #      scores["stance_tata"] = {"pro": 0.19, "anti": 0.64, "neutral": 0.17}
    utilities: Dict[str, float] = field(default_factory=dict)
    # utilities["nli_entails"] = 0.73; utilities["stance_pro"] = 0.19


class PairScorer(ABC):
    """Base class for scorers that evaluate target-passage pairs."""

    def __init__(self, name: str, task: Task):
        self.name = name
        self.task = task

    @abstractmethod
    def batch_score(self, targets: List[str], passages: List[str]) -> List[Dict[str, float]]:
        """Return calibrated label→prob dict per (target, passage) pair."""
        raise NotImplementedError


class RescoreEngine:
    """Applies multiple scorers to candidates and adds their scores."""

    def __init__(self, scorers: List[PairScorer]):
        self.scorers = scorers

    def rescore(self, candidates: List[Candidate], target: str, premise_is_target: bool = False) -> List[Candidate]:
        """Apply all scorers to candidates for the given target.

        Args:
            candidates: List of candidate documents/chunks
            target: The target text (finding, hypothesis, etc.)
            premise_is_target: If True, target is premise and candidates are hypotheses.
                              If False, candidates are premises and target is hypothesis.
        """
        passages = [c.text for c in candidates]

        # For each scorer, batch score all candidates
        for scorer in self.scorers:
            if premise_is_target:
                # Finding Document Matching: premise=target (finding), hypothesis=passage (document)
                probs_list = scorer.batch_score([target] * len(passages), passages)
            else:
                # Coal Analysis: premise=passage (document), hypothesis=target (config statement)
                probs_list = scorer.batch_score(passages, [target] * len(passages))

            # Attach results to each candidate
            for i, (cand, probs) in enumerate(zip(candidates, probs_list)):
                cand.scores[scorer.name] = probs

        return candidates


class Retriever(ABC):
    """Abstract base class for vector retrievers."""

    @abstractmethod
    def query(
        self,
        query_vector: Union[str, np.ndarray],
        top_k: int = 10
    ) -> List[Candidate]:
        """
        Query for similar items.

        Args:
            query_vector: Either text string or pre-computed embedding vector
            top_k: Number of top results to return

        Returns:
            List of Candidate objects sorted by score (highest first)
        """
        pass





class RerankPolicy:
    """Policy for converting label probabilities to utility scores."""

    def __init__(self, positive_labels: Dict[str, float]):
        """Initialize with label→weight mapping for positive contributions."""
        self.positive_labels = positive_labels

    def to_utility(self, label_probs: Dict[str, float]) -> float:
        """Convert label probabilities to utility score."""
        return sum(self.positive_labels.get(lbl, 0.0) * prob for lbl, prob in label_probs.items())


class RerankingEngine:
    """Applies reranking policies to compute utilities and sort candidates."""

    def __init__(self, policies: Dict[str, RerankPolicy]):
        """Initialize with scorer.name -> policy mapping."""
        self.policies = policies

    def rerank(self, candidates: List[Candidate]) -> List[Candidate]:
        """Compute utilities for all candidates and return sorted list."""
        for cand in candidates:
            for scorer_name, policy in self.policies.items():
                if scorer_name in cand.scores:
                    utility = policy.to_utility(cand.scores[scorer_name])
                    cand.utilities[scorer_name] = utility

        # Sort by the first utility score (can be made configurable later)
        if candidates and candidates[0].utilities:
            first_utility_key = next(iter(candidates[0].utilities.keys()))
            candidates.sort(key=lambda c: c.utilities.get(first_utility_key, 0), reverse=True)

        return candidates

