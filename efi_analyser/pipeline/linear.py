"""
EFI Analyser - Linear pipeline implementation (Filter → Processor → Aggregator)
"""

from typing import Any, Dict, Optional, Callable, List
from pathlib import Path

# Try to import tqdm, fallback to None if not available
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    tqdm = None
    TQDM_AVAILABLE = False

from efi_corpus import CorpusHandle
from .base import AbstractPipeline
from ..types import PipelineResult, AnalysisResult


class LinearPipeline(AbstractPipeline):
    """
    A simple linear pipeline that applies: Filter → Processor → Aggregator
    
    Each stage can return None if no results, which will be handled gracefully.
    Processes documents one at a time to minimize memory usage.
    """
    
    def __init__(self, 
                 filter_func: Optional[Callable] = None,
                 processor_func: Optional[Callable] = None, 
                 aggregator_func: Optional[Callable] = None,
                 processor_name: str = "processor"):
        """
        Initialize the pipeline
        
        Args:
            filter_func: Function to filter documents (takes doc, returns bool or None)
            processor_func: Function to process documents (takes doc, returns processed data or None)
            aggregator_func: Function to aggregate results (takes list of AnalysisResult, returns aggregated data)
            processor_name: Name to use as key in processing_results (default: "processor")
        """
        self.filter_func = filter_func
        self.processor_func = processor_func
        self.aggregator_func = aggregator_func
        self.processor_name = processor_name
    
    def get_name(self) -> str:
        return "LinearPipeline"
    
    def run(self, corpus_handle: CorpusHandle) -> PipelineResult:
        """
        Run the pipeline on the given corpus
        
        Args:
            corpus_handle: Corpus to analyze
            
        Returns:
            PipelineResult containing the analysis results
        """
        total_docs = len(corpus_handle.list_ids())
        
        # Stage 1 & 2: Filter and process documents one at a time
        analysis_results = []
        
        # Create iterator with progress bar if tqdm is available
        if TQDM_AVAILABLE:
            doc_iterator = tqdm(corpus_handle.read_documents(), total=total_docs, desc="Processing documents")
        else:
            doc_iterator = corpus_handle.read_documents()
        
        for doc in doc_iterator:
            # Filter document
            if self.filter_func is not None:
                filter_result = self.filter_func(doc)
                if filter_result is None or not filter_result:
                    continue
            
            # Process document
            if self.processor_func is None:
                processing_results = {"raw_document": doc}
            else:
                processing_results = self.processor_func(doc)
                if processing_results is None:
                    continue
            
            # Structure processing results with processor name as key
            structured_results = {self.processor_name: processing_results}
            
            # Create AnalysisResult object compatible with existing aggregators
            analysis_result = AnalysisResult(
                doc_id=getattr(doc, 'stable_id', str(id(doc))),
                url=getattr(doc, 'url', ''),
                passed_filters=True,  # Document passed filter to get here
                filter_results={},  # No specific filter results for now
                processing_results=structured_results,
                meta=getattr(doc, 'meta', {})
            )
            analysis_results.append(analysis_result)
        
        if not analysis_results:
            return PipelineResult(data=None, metadata={"message": "No documents produced results"})
        
        # Stage 3: Aggregate results
        if self.aggregator_func is None:
            final_result = analysis_results
        else:
            final_result = self.aggregator_func(analysis_results)
        
        metadata = {
            "total_documents": total_docs,
            "filtered_documents": len(analysis_results),  # All processed docs passed filter
            "processed_results": len(analysis_results),
            "pipeline_name": self.get_name()
        }
        
        return PipelineResult(data=final_result, metadata=metadata)
