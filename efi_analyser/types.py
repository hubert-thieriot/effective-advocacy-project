"""
Type definitions for the analysis system
"""

from typing import Protocol, Dict, Any, List, Optional
from dataclasses import dataclass
from efi_corpus.types import Document


@dataclass
class AnalysisResult:
    """Result of analyzing a single document"""
    doc_id: str
    url: str
    passed_filters: bool
    filter_results: Dict[str, bool]
    processing_results: Dict[str, Any]
    meta: Dict[str, Any]


@dataclass
class AggregatedResult:
    """Result of aggregating multiple analysis results"""
    aggregator_name: str
    aggregated_data: Dict[str, Any]
    metadata: Dict[str, Any] = None


@dataclass
class AnalysisPipelineResult:
    """Result of running the complete analysis pipeline"""
    processed_results: Optional[List[AnalysisResult]] = None
    aggregated_results: List[AggregatedResult] = None
    stats: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.aggregated_results is None:
            self.aggregated_results = []
        if self.stats is None:
            self.stats = {}


class Filter(Protocol):
    """Protocol for document filters"""
    def apply(self, document: Document) -> bool:
        """Apply filter to document, return True if document passes"""
        ...


class Processor(Protocol):
    """Protocol for document processors"""
    def process(self, document: Document) -> Dict[str, Any]:
        """Process document and return results"""
        ...


class Aggregator(Protocol):
    """Protocol for result aggregators"""
    def aggregate(self, results: List[AnalysisResult]) -> AggregatedResult:
        """Aggregate a list of analysis results into a single aggregated result"""
        ...
