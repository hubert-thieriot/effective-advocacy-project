"""Tests for domain aggregation."""

import pytest

from apps.narrative_framing.aggregation_document import DocumentFrameAggregate
from apps.narrative_framing.aggregation_domain import DomainAggregate, DomainAggregator


@pytest.fixture
def sample_aggregates():
    """Create sample document aggregates with different domains."""
    return [
        DocumentFrameAggregate(
            doc_id="doc1",
            frame_scores={"frame_a": 0.8, "frame_b": 0.2},
            total_weight=100.0,
            url="https://example.com/article1"
        ),
        DocumentFrameAggregate(
            doc_id="doc2",
            frame_scores={"frame_a": 0.5, "frame_b": 0.5},
            total_weight=150.0,
            url="https://example.com/article2"
        ),
        DocumentFrameAggregate(
            doc_id="doc3",
            frame_scores={"frame_a": 0.2, "frame_b": 0.8},
            total_weight=200.0,
            url="https://newsite.com/article1"
        ),
        DocumentFrameAggregate(
            doc_id="doc4",
            frame_scores={"frame_a": 0.0, "frame_b": 1.0},
            total_weight=50.0,
            url="https://newsite.com/article2"
        ),
    ]


@pytest.fixture
def aggregates_with_no_frames():
    """Create aggregates with some documents having no frames."""
    return [
        DocumentFrameAggregate(
            doc_id="doc1",
            frame_scores={"frame_a": 0.8, "frame_b": 0.2},
            total_weight=100.0,
            url="https://example.com/article1"
        ),
        DocumentFrameAggregate(
            doc_id="doc2",
            frame_scores={"frame_a": 0.0, "frame_b": 0.0},  # No frames
            total_weight=150.0,
            url="https://example.com/article2"
        ),
        DocumentFrameAggregate(
            doc_id="doc3",
            frame_scores={"frame_a": 0.2, "frame_b": 0.8},
            total_weight=200.0,
            url="https://newsite.com/article1"
        ),
    ]


class TestDomainAggregator:
    """Test DomainAggregator."""
    
    def test_basic_aggregation(self, sample_aggregates):
        agg = DomainAggregator()
        results = agg.aggregate(sample_aggregates)
        
        assert len(results) == 2  # Two unique domains
        assert all(isinstance(r, DomainAggregate) for r in results)
        
        domains = {r.domain for r in results}
        assert "example.com" in domains
        assert "newsite.com" in domains
    
    def test_keep_documents_with_no_frames_false(self, aggregates_with_no_frames):
        agg = DomainAggregator(keep_documents_with_no_frames=False)
        results = agg.aggregate(aggregates_with_no_frames)
        
        # Should exclude doc2 which has no frames
        example_domain = next((r for r in results if r.domain == "example.com"), None)
        assert example_domain is not None
        assert example_domain.document_count == 1  # Only doc1
    
    def test_keep_documents_with_no_frames_true(self, aggregates_with_no_frames):
        agg = DomainAggregator(keep_documents_with_no_frames=True)
        results = agg.aggregate(aggregates_with_no_frames)
        
        # Should include doc2 even though it has no frames
        example_domain = next((r for r in results if r.domain == "example.com"), None)
        assert example_domain is not None
        assert example_domain.document_count == 2  # doc1 and doc2
    
    def test_weight_by_document_weight_true(self, sample_aggregates):
        agg = DomainAggregator(weight_by_document_weight=True)
        results = agg.aggregate(sample_aggregates)
        
        assert len(results) > 0
        for result in results:
            assert isinstance(result.frame_scores, dict)
            assert result.document_count > 0
    
    def test_weight_by_document_weight_false(self, sample_aggregates):
        agg = DomainAggregator(weight_by_document_weight=False)
        results = agg.aggregate(sample_aggregates)
        
        # Should compute averages without weighting
        assert len(results) > 0
        for result in results:
            assert isinstance(result.frame_scores, dict)
    
    def test_avg_or_sum_avg(self, sample_aggregates):
        agg = DomainAggregator(avg_or_sum="avg")
        results = agg.aggregate(sample_aggregates)
        
        assert len(results) > 0
        for result in results:
            assert isinstance(result.frame_scores, dict)
    
    def test_avg_or_sum_sum(self, sample_aggregates):
        agg = DomainAggregator(avg_or_sum="sum")
        results = agg.aggregate(sample_aggregates)
        
        assert len(results) > 0
        for result in results:
            assert isinstance(result.frame_scores, dict)
    
    def test_document_count(self, sample_aggregates):
        agg = DomainAggregator()
        results = agg.aggregate(sample_aggregates)
        
        # Counts should match document counts per domain
        domain_to_result = {r.domain: r for r in results}
        assert domain_to_result["example.com"].document_count == 2
        assert domain_to_result["newsite.com"].document_count == 2
    
    def test_empty_input(self):
        agg = DomainAggregator()
        results = agg.aggregate([])
        assert results == []
    
    def test_no_domain_skipped(self):
        # Document with no domain should be skipped
        no_domain_agg = DocumentFrameAggregate(
            doc_id="nodomain",
            frame_scores={"frame_a": 0.5},
            total_weight=100.0,
            url=None
        )
        agg = DomainAggregator()
        results = agg.aggregate([no_domain_agg])
        assert len(results) == 0
    
    def test_domain_aggregate_structure(self, sample_aggregates):
        agg = DomainAggregator()
        results = agg.aggregate(sample_aggregates)
        
        for result in results:
            assert isinstance(result, DomainAggregate)
            assert isinstance(result.domain, str)
            assert isinstance(result.frame_scores, dict)
            assert isinstance(result.document_count, int)
            assert result.document_count > 0
