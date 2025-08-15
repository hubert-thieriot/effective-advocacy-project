"""
Tests for efi_analyser aggregators
"""

import pytest
from efi_analyser.aggregators import KeywordPresenceAggregator, DocumentCountAggregator
from efi_analyser.types import AnalysisResult


class TestKeywordPresenceAggregator:
    """Test KeywordPresenceAggregator"""
    
    def test_aggregator_init(self):
        """Test aggregator initialization"""
        aggregator = KeywordPresenceAggregator("test_aggregator")
        assert aggregator.name == "test_aggregator"
    
    def test_aggregator_with_keyword_results(self, test_keywords):
        """Test aggregator with documents containing keywords"""
        aggregator = KeywordPresenceAggregator("keyword_presence")
        
        # Create mock analysis results
        results = [
            AnalysisResult(
                doc_id="doc1",
                url="https://example.com/doc1",
                passed_filters=True,
                filter_results={},
                processing_results={
                    'keyword_extractor': {
                        'keyword_counts': {'CREA': 2, 'coal': 1, 'transport': 0},
                        'keyword_positions': {},
                        'total_keywords': 3
                    }
                },
                meta={}
            ),
            AnalysisResult(
                doc_id="doc2",
                url="https://example.com/doc2",
                passed_filters=True,
                filter_results={},
                processing_results={
                    'keyword_extractor': {
                        'keyword_counts': {'CREA': 0, 'coal': 1, 'transport': 1},
                        'keyword_positions': {},
                        'total_keywords': 2
                    }
                },
                meta={}
            ),
            AnalysisResult(
                doc_id="doc3",
                url="https://example.com/doc3",
                passed_filters=True,
                filter_results={},
                processing_results={
                    'keyword_extractor': {
                        'keyword_counts': {'CREA': 0, 'coal': 0, 'transport': 0},
                        'keyword_positions': {},
                        'total_keywords': 0
                    }
                },
                meta={}
            )
        ]
        
        result = aggregator.aggregate(results)
        
        assert result.aggregator_name == "keyword_presence"
        assert result.aggregated_data["total_documents"] == 3
        assert result.aggregated_data["keyword_counts"]["CREA"] == 1  # Present in 1 doc
        assert result.aggregated_data["keyword_counts"]["coal"] == 2  # Present in 2 docs
        assert result.aggregated_data["keyword_counts"]["transport"] == 1  # Present in 1 doc
        assert result.metadata["description"] == "Counts how many documents mention each keyword"
    
    def test_aggregator_with_no_valid_results(self):
        """Test aggregator with no valid results"""
        aggregator = KeywordPresenceAggregator("keyword_presence")
        
        # Results that didn't pass filters
        results = [
            AnalysisResult(
                doc_id="doc1",
                url="https://example.com/doc1",
                passed_filters=False,
                filter_results={},
                processing_results={},
                meta={}
            )
        ]
        
        result = aggregator.aggregate(results)
        
        assert result.aggregated_data["total_documents"] == 0
        assert result.aggregated_data["keyword_counts"] == {}
        assert "message" in result.metadata
    
    def test_aggregator_with_missing_keyword_extractor(self):
        """Test aggregator with results missing keyword extraction"""
        aggregator = KeywordPresenceAggregator("keyword_presence")
        
        results = [
            AnalysisResult(
                doc_id="doc1",
                url="https://example.com/doc1",
                passed_filters=True,
                filter_results={},
                processing_results={},  # No keyword_extractor
                meta={}
            )
        ]
        
        result = aggregator.aggregate(results)
        
        assert result.aggregated_data["total_documents"] == 0
        assert result.aggregated_data["keyword_counts"] == {}


class TestDocumentCountAggregator:
    """Test DocumentCountAggregator"""
    
    def test_aggregator_init(self):
        """Test aggregator initialization"""
        aggregator = DocumentCountAggregator("test_aggregator")
        assert aggregator.name == "test_aggregator"
    
    def test_aggregator_basic_counts(self):
        """Test basic document counting"""
        aggregator = DocumentCountAggregator("document_counts")
        
        results = [
            AnalysisResult(
                doc_id="doc1",
                url="https://example.com/doc1",
                passed_filters=True,
                filter_results={},
                processing_results={},
                meta={}
            ),
            AnalysisResult(
                doc_id="doc2",
                url="https://example.com/doc2",
                passed_filters=False,
                filter_results={},
                processing_results={},
                meta={}
            ),
            AnalysisResult(
                doc_id="doc3",
                url="https://example.com/doc3",
                passed_filters=True,
                processing_results={},
                filter_results={},
                meta={}
            )
        ]
        
        result = aggregator.aggregate(results)
        
        assert result.aggregated_data["total_documents"] == 3
        assert result.aggregated_data["passed_filters"] == 2
        assert result.aggregated_data["failed_filters"] == 1
        assert result.aggregated_data["processing_errors"] == 0
        assert result.aggregated_data["success_rate"] == pytest.approx(66.67, rel=0.01)
    
    def test_aggregator_with_processing_errors(self):
        """Test aggregator with processing errors"""
        aggregator = DocumentCountAggregator("document_counts")
        
        results = [
            AnalysisResult(
                doc_id="doc1",
                url="https://example.com/doc1",
                passed_filters=True,
                filter_results={},
                processing_results={"processor1": {"error": "Something went wrong"}},
                meta={}
            ),
            AnalysisResult(
                doc_id="doc2",
                url="https://example.com/doc2",
                passed_filters=True,
                filter_results={},
                processing_results={"processor1": {"result": "success"}},
                meta={}
            )
        ]
        
        result = aggregator.aggregate(results)
        
        assert result.aggregated_data["total_documents"] == 2
        assert result.aggregated_data["passed_filters"] == 2
        assert result.aggregated_data["processing_errors"] == 1
        assert result.aggregated_data["success_rate"] == 100.0
    
    def test_aggregator_empty_results(self):
        """Test aggregator with empty results"""
        aggregator = DocumentCountAggregator("document_counts")
        
        result = aggregator.aggregate([])
        
        assert result.aggregated_data["total_documents"] == 0
        assert result.aggregated_data["passed_filters"] == 0
        assert result.aggregated_data["failed_filters"] == 0
        assert result.aggregated_data["processing_errors"] == 0
        assert result.aggregated_data["success_rate"] == 0.0
