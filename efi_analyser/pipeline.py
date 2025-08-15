"""
Main analysis pipeline for processing corpora
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from .types import Filter, Processor, AnalysisResult, Aggregator, AggregatedResult, AnalysisPipelineResult
from efi_corpus.types import Document
from efi_corpus.corpus_reader import CorpusReader


class AnalysisPipeline:
    """Main analysis pipeline that applies filters and processors to documents"""
    
    def __init__(self, 
                 corpus_path: Path,
                 filters: Optional[List[Filter]] = None,
                 processors: Optional[List[Processor]] = None,
                 aggregators: Optional[List[Aggregator]] = None,
                 output_path: Optional[Path] = None,
                 keep_processed_results: bool = True):
        """
        Initialize analysis pipeline
        
        Args:
            corpus_path: Path to corpus directory
            filters: List of filters to apply (AND logic)
            processors: List of processors to run on filtered documents
            aggregators: List of aggregators to run on final results
            output_path: Optional path to save results
            keep_processed_results: Whether to keep individual processed results in memory
        """
        self.corpus_path = Path(corpus_path)
        self.filters = filters or []
        self.processors = processors or []
        self.aggregators = aggregators or []
        self.output_path = Path(output_path) if output_path else None
        self.keep_processed_results = keep_processed_results
        
        # Initialize corpus reader
        self.corpus_reader = CorpusReader(self.corpus_path)
        
        # Results storage
        self.results: List[AnalysisResult] = []
        self.aggregated_results: List[AggregatedResult] = []
        self.stats = {
            "total_documents": 0,
            "passed_filters": 0,
            "failed_filters": 0,
            "processing_errors": 0,
            "start_time": None,
            "end_time": None
        }
    
    def run(self) -> AnalysisPipelineResult:
        """Run the analysis pipeline and return results"""
        print(f"Starting analysis of corpus: {self.corpus_path}")
        print(f"Filters: {len(self.filters)}")
        print(f"Processors: {len(self.processors)}")
        print(f"Aggregators: {len(self.aggregators)}")
        
        self.stats["start_time"] = time.time()
        self.stats["total_documents"] = self.corpus_reader.get_document_count()
        
        # Process each document
        for i, document in enumerate(self.corpus_reader.read_documents(), 1):
            if i % 100 == 0:
                print(f"Processed {i}/{self.stats['total_documents']} documents...")
            
            try:
                result = self._process_document(document)
                self.results.append(result)
                
                if result.passed_filters:
                    self.stats["passed_filters"] += 1
                else:
                    self.stats["failed_filters"] += 1
                    
            except Exception as e:
                print(f"Error processing document {document.doc_id}: {e}")
                self.stats["processing_errors"] += 1
                continue
        
        # Run aggregators if any are specified
        if self.aggregators:
            print("\nRunning aggregators...")
            for aggregator in self.aggregators:
                try:
                    aggregated_result = aggregator.aggregate(self.results)
                    self.aggregated_results.append(aggregated_result)
                    print(f"  {aggregator.name}: Completed")
                except Exception as e:
                    print(f"  Error in aggregator {aggregator.name}: {e}")
                    continue
        
        self.stats["end_time"] = time.time()
        self._print_summary()
        
        # Save results if output path specified
        if self.output_path:
            self._save_results()
        
        # Create result object
        pipeline_result = AnalysisPipelineResult(
            processed_results=self.results if self.keep_processed_results else None,
            aggregated_results=self.aggregated_results,
            stats=self.stats
        )
        
        # Clear processed results from memory if not keeping them
        if not self.keep_processed_results:
            self.results = []
        
        return pipeline_result
    
    def _process_document(self, document: Document) -> AnalysisResult:
        """Process a single document through filters and processors"""
        # Apply filters
        filter_results = {}
        passed_filters = True
        
        for filter_obj in self.filters:
            try:
                filter_result = filter_obj.apply(document)
                filter_results[filter_obj.name] = filter_result
                if not filter_result:
                    passed_filters = False
            except Exception as e:
                print(f"Error applying filter {filter_obj.name}: {e}")
                filter_results[filter_obj.name] = False
                passed_filters = False
        
        # Apply processors only if document passed all filters
        processing_results = {}
        if passed_filters and self.processors:
            for processor in self.processors:
                try:
                    result = processor.process(document)
                    processing_results[processor.name] = result
                except Exception as e:
                    print(f"Error in processor {processor.name}: {e}")
                    processing_results[processor.name] = {"error": str(e)}
        
        return AnalysisResult(
            doc_id=document.doc_id,
            url=document.url,
            passed_filters=passed_filters,
            filter_results=filter_results,
            processing_results=processing_results,
            meta=document.meta
        )
    
    def _print_summary(self):
        """Print analysis summary"""
        duration = self.stats["end_time"] - self.stats["start_time"]
        
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        print(f"Corpus: {self.corpus_path}")
        print(f"Total documents: {self.stats['total_documents']}")
        print(f"Passed filters: {self.stats['passed_filters']}")
        print(f"Failed filters: {self.stats['failed_filters']}")
        print(f"Processing errors: {self.stats['processing_errors']}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Documents per second: {self.stats['total_documents']/duration:.2f}")
        
        if self.filters:
            print(f"\nFilter breakdown:")
            for filter_obj in self.filters:
                passed = sum(1 for r in self.results if r.filter_results.get(filter_obj.name, False))
                print(f"  {filter_obj.name}: {passed}/{self.stats['total_documents']} passed")
        
        if self.aggregated_results:
            print(f"\nAggregated Results:")
            for agg_result in self.aggregated_results:
                print(f"  {agg_result.aggregator_name}: {agg_result.metadata.get('description', 'No description')}")
                if 'keyword_counts' in agg_result.aggregated_data:
                    print(f"    Keywords found in documents:")
                    for keyword, count in agg_result.aggregated_data['keyword_counts'].items():
                        print(f"      {keyword}: {count} documents")
                elif 'total_documents' in agg_result.aggregated_data:
                    print(f"    Total documents: {agg_result.aggregated_data['total_documents']}")
                    if 'success_rate' in agg_result.aggregated_data:
                        print(f"    Success rate: {agg_result.aggregated_data['success_rate']:.1f}%")
    
    def _save_results(self):
        """Save results to output file"""
        if not self.output_path:
            return
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to serializable format (only if kept in memory)
        serializable_results = []
        if self.keep_processed_results:
            for result in self.results:
                serializable_results.append({
                    "doc_id": result.doc_id,
                    "url": result.url,
                    "passed_filters": result.passed_filters,
                    "filter_results": result.filter_results,
                    "processing_results": result.processing_results,
                    "meta": result.meta
                })
        
        # Convert aggregated results to serializable format
        serializable_aggregated = []
        for agg_result in self.aggregated_results:
            serializable_aggregated.append({
                "aggregator_name": agg_result.aggregator_name,
                "aggregated_data": agg_result.aggregated_data,
                "metadata": agg_result.metadata
            })
        
        # Save results
        output_data = {
            "corpus_path": str(self.corpus_path),
            "analysis_stats": self.stats,
            "results": serializable_results if self.keep_processed_results else None,
            "aggregated_results": serializable_aggregated
        }
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nResults saved to: {self.output_path}")
    
    def get_filtered_results(self) -> List[AnalysisResult]:
        """Get only results that passed all filters"""
        if not self.keep_processed_results:
            raise ValueError("Processed results not kept in memory. Set keep_processed_results=True to access individual results.")
        return [r for r in self.results if r.passed_filters]
    
    def get_failed_results(self) -> List[AnalysisResult]:
        """Get results that failed filters"""
        if not self.keep_processed_results:
            raise ValueError("Processed results not kept in memory. Set keep_processed_results=True to access individual results.")
        return [r for r in self.results if not r.passed_filters]
    
    def get_processing_errors(self) -> List[AnalysisResult]:
        """Get results with processing errors"""
        if not self.keep_processed_results:
            raise ValueError("Processed results not kept in memory. Set keep_processed_results=True to access individual results.")
        return [r for r in self.results if any("error" in str(v) for v in r.processing_results.values())]
    
    def get_aggregated_results(self) -> List[AggregatedResult]:
        """Get all aggregated results"""
        return self.aggregated_results
    
    def get_aggregated_result(self, aggregator_name: str) -> Optional[AggregatedResult]:
        """Get a specific aggregated result by name"""
        for result in self.aggregated_results:
            if result.aggregator_name == aggregator_name:
                return result
        return None
