"""
EFI Analyser - Word Occurrence Pipeline

This pipeline analyzes keyword occurrences in a corpus using the existing
KeywordExtractorProcessor and KeywordPresenceAggregator.
"""

import time
from typing import Dict, List, Any
from pathlib import Path

from tqdm import tqdm

from .base import AbstractPipeline
from ..types import PipelineResult, AnalysisResult, AnalysisPipelineResult, WordOccurrenceResults, WordOccurrenceResult
from ..processors import KeywordExtractorProcessor
from ..aggregators import KeywordPresenceAggregator
from efi_corpus import CorpusHandle


class WordOccurrencePipeline(AbstractPipeline):
    """
    Pipeline for analyzing keyword occurrences in a corpus.
    
    Uses KeywordExtractorProcessor to extract keywords from documents and
    KeywordPresenceAggregator to aggregate results across the corpus.
    """
    
    def __init__(self, 
                 keywords: List[str],
                 case_sensitive: bool = False,
                 whole_word_only: bool = True,
                 allow_hyphenation: bool = True,
                 doc_limit: int | None = None,
                 date_from: str | None = None,
                 date_to: str | None = None,
                 patterns: List[str] | None = None):
        """
        Initialize the word occurrence pipeline.
        
        Args:
            keywords: List of keywords to search for
            case_sensitive: Whether search should be case sensitive
            whole_word_only: Whether to match only whole words
            allow_hyphenation: Whether to allow keywords split across hyphens
        """
        self.keywords = keywords
        self.case_sensitive = case_sensitive
        self.whole_word_only = whole_word_only
        self.allow_hyphenation = allow_hyphenation
        self.doc_limit = doc_limit
        self.date_from = date_from
        self.date_to = date_to
        self.patterns = patterns or []
        
        # Initialize processor and aggregator
        self.processor = KeywordExtractorProcessor(
            keywords=keywords,
            case_sensitive=case_sensitive,
            whole_word_only=whole_word_only,
            allow_hyphenation=allow_hyphenation,
            patterns=self.patterns,
        )
        
        self.aggregator = KeywordPresenceAggregator()
        
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
        return "WordOccurrencePipeline"
    
    def run(self, corpus_handle: CorpusHandle) -> PipelineResult:
        """Run the word occurrence analysis pipeline."""
        start_time = time.time()
        self._stats["start_time"] = start_time

        total_docs = corpus_handle.get_document_count()
        self._stats["total_documents"] = total_docs

        results: List[AnalysisResult] = []
        iterator = tqdm(corpus_handle.iter_documents(), total=total_docs, desc="Processing documents")

        processed = 0
        for doc in iterator:
            # Optional date filtering
            if self.date_from or self.date_to:
                try:
                    published_at = getattr(doc, "published_at", None)
                    if published_at:
                        # Support both datetime-like and string dates
                        from datetime import datetime
                        if hasattr(published_at, "isoformat"):
                            pub_str = published_at.date().isoformat()
                        else:
                            pub_str = str(published_at)[:10]
                        if self.date_from and pub_str < self.date_from:
                            continue
                        if self.date_to and pub_str > self.date_to:
                            continue
                except Exception:
                    # If date parsing fails, do not filter out
                    pass
            # All documents pass filters in word occurrence analysis
            passed = True
            filter_results: Dict[str, bool] = {"word_occurrence": True}
            
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
            
            # Apply processor
            try:
                res = self.processor.process(doc)
                if res is not None:
                    processing_results["keyword_extractor"] = res
            except Exception as e:
                self._stats["processing_errors"] += 1
                processing_results["keyword_extractor"] = {"error": f"Processing failed: {str(e)}"}
            
            ar = AnalysisResult(
                doc_id=getattr(doc, "doc_id", ""),
                url=getattr(doc, "url", ""),
                passed_filters=True,
                filter_results=filter_results,
                processing_results=processing_results,
                meta=getattr(doc, "meta", {}),
            )
            
            results.append(ar)
            self._processed_results.append(ar)

            processed += 1
            if self.doc_limit is not None and processed >= self.doc_limit:
                break

        # Aggregate results
        aggregated_result = self.aggregator.aggregate(results)
        
        end_time = time.time()
        self._stats["end_time"] = end_time
        self._stats["total_time"] = end_time - start_time

        # Create pipeline result
        pipeline_result = PipelineResult(
            pipeline_name=self.get_name(),
            results=results,
            aggregated_result=aggregated_result,
            metadata=self._stats
        )

        return pipeline_result
    
    def get_word_occurrence_results(self, pipeline_result: PipelineResult) -> WordOccurrenceResults:
        """Convert pipeline result to WordOccurrenceResults format."""
        # Extract keyword data from aggregated result
        aggregated_data = pipeline_result.aggregated_result.aggregated_data
        
        # Create individual results
        results = []
        for result in pipeline_result.results:
            if result.passed_filters and "keyword_extractor" in result.processing_results:
                keyword_data = result.processing_results["keyword_extractor"]
                
                # Skip if there was an error
                if "error" in keyword_data:
                    continue
                
                # Extract metadata
                meta = result.meta
                title = meta.get("title", "")
                # Prefer explicit 'date', fall back to 'published_at'
                date = meta.get("date", "") or meta.get("published_at", "")
                
                word_result = WordOccurrenceResult(
                    document_id=result.doc_id,
                    url=result.url,
                    title=title,
                    date=date,
                    keyword_counts=keyword_data.get("keyword_counts", {}),
                    keyword_positions=keyword_data.get("keyword_positions", {}),
                    total_keywords=keyword_data.get("total_keywords", 0),
                    metadata=meta
                )
                results.append(word_result)
        
        # Create final results container
        return WordOccurrenceResults(
            keywords=self.keywords,
            total_documents=self._stats["total_documents"],
            results=results,
            keyword_counts=aggregated_data.get("keyword_counts", {}),
            keyword_percentages=aggregated_data.get("keyword_percentages", {}),
            timing_stats=self._stats,
            metadata={
                "case_sensitive": self.case_sensitive,
                "whole_word_only": self.whole_word_only,
                "allow_hyphenation": self.allow_hyphenation,
                "pipeline_metadata": pipeline_result.metadata
            }
        )
