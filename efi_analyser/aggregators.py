"""
Aggregators for combining analysis results
"""

from typing import List, Dict, Any
from .types import AnalysisResult, AggregatedResult


class KeywordPresenceAggregator:
    """Aggregator that counts keyword presence across documents"""
    
    def __init__(self, name: str = "keyword_presence"):
        self.name = name
    
    def aggregate(self, results: List[AnalysisResult]) -> AggregatedResult:
        """Count how many documents mention each keyword"""
        # Get only results that passed filters and have keyword extraction
        valid_results = [
            r for r in results 
            if r.passed_filters and 'keyword_extractor' in r.processing_results
        ]
        
        if not valid_results:
            return AggregatedResult(
                aggregator_name=self.name,
                aggregated_data={"keyword_counts": {}, "total_documents": 0},
                metadata={"message": "No valid results to aggregate"}
            )
        
        # Initialize keyword counts
        keyword_counts = {}
        total_documents = len(valid_results)
        
        # Count keyword presence across all documents
        for result in valid_results:
            keyword_data = result.processing_results['keyword_extractor']
            if 'keyword_counts' in keyword_data:
                for keyword, count in keyword_data['keyword_counts'].items():
                    if keyword not in keyword_counts:
                        keyword_counts[keyword] = 0
                    # Count as present if count > 0
                    if count > 0:
                        keyword_counts[keyword] += 1
        
        # Calculate percentages
        keyword_percentages = {}
        for keyword, count in keyword_counts.items():
            keyword_percentages[keyword] = (count / total_documents) * 100
        
        return AggregatedResult(
            aggregator_name=self.name,
            aggregated_data={
                "keyword_counts": keyword_counts,
                "keyword_percentages": keyword_percentages,
                "total_documents": total_documents
            },
            metadata={
                "description": "Counts how many documents mention each keyword",
                "valid_results": len(valid_results)
            }
        )


class DocumentCountAggregator:
    """Simple aggregator that counts documents by various criteria"""
    
    def __init__(self, name: str = "document_counts"):
        self.name = name
    
    def aggregate(self, results: List[AnalysisResult]) -> AggregatedResult:
        """Count documents by various criteria"""
        total_docs = len(results)
        passed_filters = sum(1 for r in results if r.passed_filters)
        failed_filters = total_docs - passed_filters
        
        # Count processing errors
        processing_errors = 0
        for result in results:
            if result.passed_filters:
                for processor_result in result.processing_results.values():
                    if isinstance(processor_result, dict) and "error" in processor_result:
                        processing_errors += 1
                        break
        
        return AggregatedResult(
            aggregator_name=self.name,
            aggregated_data={
                "total_documents": total_docs,
                "passed_filters": passed_filters,
                "failed_filters": failed_filters,
                "processing_errors": processing_errors,
                "success_rate": (passed_filters / total_docs) * 100 if total_docs > 0 else 0
            },
            metadata={
                "description": "Basic document counting and success rates"
            }
        )
