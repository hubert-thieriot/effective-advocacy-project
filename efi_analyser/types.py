"""
Data models and configuration for the EFI Analyser layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path


# ============================================================================
# Document Matching Types
# ============================================================================

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
    """Container for all document matching results."""
    findings_processed: int
    total_matches: int
    results_by_finding: Dict[str, FindingResults]
    timing_stats: Dict[str, Any]
    score_stats: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class DocumentMatchingConfig:
    """Configuration for document matching pipeline."""
    corpus_path: Path
    library_path: Path
    workspace_path: Path
    cosine_threshold: float = 0.1
    top_k: int = 5
    rescorers: List[str] = field(default_factory=lambda: ["nli"])
    output_formats: List[str] = field(default_factory=lambda: ["csv", "json", "html"])


# ============================================================================
# Claim Supporting Types
# ============================================================================

@dataclass
class ClaimSupportingResult:
    """A single claim supporting result."""
    claim: str
    chunk_id: str
    document_id: str
    classification: str  # 'entailment', 'contradiction', 'neutral'
    entailment_score: float
    neutral_score: float
    contradiction_score: float
    cosine_score: float
    media_source: str
    date: str
    url: str
    title: str


@dataclass
class ClaimSupportingResults:
    """Results for a single claim."""
    claim: str
    results: List[ClaimSupportingResult]
    total_chunks: int
    entailment_count: int
    contradiction_count: int
    neutral_count: int
    media_breakdown: Dict[str, Dict[str, int]]


@dataclass
class ClaimSupportingConfig:
    """Configuration for claim supporting pipeline."""
    corpus_path: Path
    workspace_path: Path
    cosine_threshold: float = 0.1
    top_k_retrieval: int = 1000
    nli_model: str = "typeform/distilbert-base-uncased-mnli"
    classification_threshold: float = 0.7
    output_formats: List[str] = field(default_factory=lambda: ["csv", "json", "html"])


# ============================================================================
# Word Occurrence Types
# ============================================================================

@dataclass
class WordOccurrenceResult:
    """A single word occurrence result."""
    document_id: str
    url: str
    title: str
    date: str
    keyword_counts: Dict[str, int]
    keyword_positions: Dict[str, int]
    total_keywords: int
    metadata: Dict[str, Any]


@dataclass
class WordOccurrenceResults:
    """Container for all word occurrence results."""
    keywords: List[str]
    total_documents: int
    results: List[WordOccurrenceResult]
    keyword_counts: Dict[str, int]
    keyword_percentages: Dict[str, float]
    timing_stats: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class WordOccurrenceConfig:
    """Configuration for word occurrence pipeline."""
    corpus_path: Path
    workspace_path: Path
    keywords: List[str]
    case_sensitive: bool = False
    whole_word_only: bool = True
    allow_hyphenation: bool = True
    output_formats: List[str] = field(default_factory=lambda: ["csv", "json", "html"])


# ============================================================================
# Base Classes
# ============================================================================

class Processor:
    """Base class for document processors."""
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
    
    def process(self, document) -> Dict[str, Any]:
        """
        Process a document and return results.
        
        Args:
            document: Document to process
            
        Returns:
            Dictionary containing processing results
        """
        raise NotImplementedError("Subclasses must implement process method")


class Filter:
    """Base class for document filters."""
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
    
    def filter(self, document) -> bool:
        """
        Filter a document and return whether it passes.
        
        Args:
            document: Document to filter
            
        Returns:
            True if document passes filter, False otherwise
        """
        raise NotImplementedError("Subclasses must implement filter method")


# ============================================================================
# Pipeline Types
# ============================================================================

@dataclass
class AnalysisResult:
    """Result of processing a single document."""
    doc_id: str
    url: str
    passed_filters: bool
    filter_results: Dict[str, Any]
    processing_results: Dict[str, Any]
    meta: Dict[str, Any]


@dataclass
class AggregatedResult:
    """Result of aggregating analysis results."""
    aggregator_name: str
    aggregated_data: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class PipelineResult:
    """Result of running a pipeline."""
    pipeline_name: str
    results: List[AnalysisResult]
    aggregated_result: AggregatedResult
    metadata: Dict[str, Any]


@dataclass
class AnalysisPipelineResult:
    """Legacy result format for backward compatibility."""
    data: List[AnalysisResult]
    metadata: Dict[str, Any]


# ============================================================================
# Common Types
# ============================================================================

@dataclass
class PipelineConfig:
    """Base configuration for all pipelines."""
    workspace_path: Path
    output_formats: List[str] = field(default_factory=lambda: ["csv", "json", "html"])


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    output_dir: Path
    formats: List[str] = field(default_factory=lambda: ["csv", "json", "html"])
    include_metadata: bool = True
    include_timing: bool = True
