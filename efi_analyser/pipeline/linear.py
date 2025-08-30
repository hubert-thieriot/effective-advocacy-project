"""
EFI Analyser - Linear pipeline implementation (Filter → Processor → Aggregator)
"""

from typing import Any, Dict, Optional, Callable, List
from pathlib import Path

from tqdm import tqdm

from .base import AbstractPipeline
from ..types import PipelineResult, AnalysisResult, AnalysisPipelineResult, AggregatedResult
from efi_corpus import CorpusHandle


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
                 processor_name: Optional[str] = None):
        self.filter_func = filter_func
        self.processor_func = processor_func
        self.aggregator_func = aggregator_func
        self.processor_name = processor_name or "processor"
        self._stats: Dict[str, Any] = {
            "total_documents": 0,
            "passed_filters": 0,
            "failed_filters": 0,
            "processing_errors": 0,
            "start_time": None,
            "end_time": None,
        }
        self._processed_results: List[AnalysisResult] = []
        self._failed_results: List[AnalysisResult] = []
    
    def get_name(self) -> str:
        return "LinearPipeline"
    
    def run(self, corpus_handle: CorpusHandle) -> PipelineResult:
        import time
        self._stats["start_time"] = time.time()

        total_docs = corpus_handle.get_document_count()
        self._stats["total_documents"] = total_docs

        results: List[AnalysisResult] = []
        iterator = tqdm(corpus_handle.iter_documents(), total=total_docs, desc="Processing documents")

        for doc in iterator:
            passed = True
            filter_results: Dict[str, bool] = {}
            
            # Apply filter if provided
            if self.filter_func:
                try:
                    passed = bool(self.filter_func(doc))
                    filter_results["filter"] = passed
                except Exception:
                    passed = False
                    filter_results["filter"] = False
            
            if not passed:
                self._stats["failed_filters"] += 1
                ar = AnalysisResult(
                    doc_id=getattr(doc, "doc_id", ""),
                    url=getattr(doc, "url", ""),
                    passed_filters=False,
                    filter_results=filter_results,
                    processing_results={},
                    meta=getattr(doc, "meta", {}),
                )
                self._failed_results.append(ar)
                continue
                
            self._stats["passed_filters"] += 1

            processing_results: Dict[str, Any] = {}
            
            # Apply processor if provided
            if self.processor_func:
                try:
                    res = self.processor_func(doc)
                    if res is not None:
                        processing_results[self.processor_name] = res
                except Exception:
                    self._stats["processing_errors"] += 1
                    processing_results[self.processor_name] = {"error": "Processing failed"}
            
            ar = AnalysisResult(
                doc_id=getattr(doc, "doc_id", ""),
                url=getattr(doc, "url", ""),
                passed_filters=True,
                filter_results=filter_results,
                processing_results=processing_results,
                meta=getattr(doc, "meta", {}),
            )
            results.append(ar)

        self._processed_results = results
        self._stats["end_time"] = time.time()
        
        # Apply aggregator if provided
        aggregated_result = None
        if self.aggregator_func and results:
            try:
                aggregated_result = self.aggregator_func(results)
            except Exception:
                # Create a default aggregated result with error
                aggregated_result = AggregatedResult(
                    aggregator_name="error",
                    aggregated_data={"error": "Aggregation failed"},
                    metadata={"error": "Aggregation failed"}
                )
        
        # Create pipeline result
        pipeline_result = PipelineResult(
            pipeline_name=self.get_name(),
            results=results,
            aggregated_result=aggregated_result or AggregatedResult(
                aggregator_name="none",
                aggregated_data={},
                metadata={}
            ),
            metadata={
                "stats": self._stats,
                "filter_func": self.filter_func.__name__ if self.filter_func else None,
                "processor_func": self.processor_func.__name__ if self.processor_func else None,
                "aggregator_func": self.aggregator_func.__name__ if self.aggregator_func else None,
            }
        )
        
        return pipeline_result

    @property
    def stats(self) -> Dict[str, Any]:
        return self._stats

    def get_filtered_results(self) -> List[AnalysisResult]:
        return self._processed_results

    def get_failed_results(self) -> List[AnalysisResult]:
        return self._failed_results
