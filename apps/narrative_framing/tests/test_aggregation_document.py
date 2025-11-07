"""Tests for document-level aggregation."""

import pytest
from datetime import date

from apps.narrative_framing.aggregation_document import (
    DocumentFrameAggregate,
    WeightedFrameAggregator,
    OccurrenceFrameAggregator,
    _extract_domain,
)


class TestExtractDomain:
    """Test domain extraction utility."""
    
    def test_extract_simple_domain(self):
        assert _extract_domain("https://example.com/article") == "example.com"
    
    def test_extract_with_www(self):
        assert _extract_domain("https://www.example.com/article") == "example.com"
    
    def test_extract_with_subdomain(self):
        assert _extract_domain("https://news.example.com/article") == "example.com"
    
    def test_extract_indonesian_domain(self):
        assert _extract_domain("https://kota.tribunnews.com/article") == "tribunnews.com"
    
    def test_extract_co_uk(self):
        assert _extract_domain("https://example.co.uk/article") == "example.co.uk"
    
    def test_extract_none(self):
        assert _extract_domain(None) is None
        assert _extract_domain("") is None


class TestDocumentFrameAggregate:
    """Test DocumentFrameAggregate dataclass."""
    
    def test_basic_creation(self):
        agg = DocumentFrameAggregate(
            doc_id="doc1",
            frame_scores={"frame_a": 0.5, "frame_b": 0.3},
            total_weight=100.0
        )
        assert agg.doc_id == "doc1"
        assert agg.frame_scores == {"frame_a": 0.5, "frame_b": 0.3}
        assert agg.total_weight == 100.0
        assert agg.domain is None
    
    def test_domain_extraction(self):
        agg = DocumentFrameAggregate(
            doc_id="doc1",
            frame_scores={"frame_a": 0.5},
            total_weight=100.0,
            url="https://example.com/article"
        )
        assert agg.domain == "example.com"
    
    def test_domain_not_overridden(self):
        agg = DocumentFrameAggregate(
            doc_id="doc1",
            frame_scores={"frame_a": 0.5},
            total_weight=100.0,
            url="https://example.com/article"
        )
        # Manually set domain should not be overridden
        object.__setattr__(agg, 'domain', 'custom.com')
        assert agg.domain == 'custom.com'
    
    def test_to_dict(self):
        agg = DocumentFrameAggregate(
            doc_id="doc1",
            frame_scores={"frame_a": 0.5, "frame_b": 0.3},
            total_weight=100.0,
            url="https://example.com/article"
        )
        d = agg.to_dict()
        assert d["doc_id"] == "doc1"
        assert d["frame_scores"] == {"frame_a": 0.5, "frame_b": 0.3}
        assert d["domain"] == "example.com"


class TestWeightedFrameAggregator:
    """Test WeightedFrameAggregator."""
    
    def test_basic_aggregation(self):
        agg = WeightedFrameAggregator(frame_ids=["frame_a", "frame_b"])
        agg.accumulate("doc1", "passage one text", {"frame_a": 0.8, "frame_b": 0.2})
        agg.accumulate("doc1", "passage two text", {"frame_a": 0.6, "frame_b": 0.4})
        
        results = agg.finalize()
        assert len(results) == 1
        doc = results[0]
        assert doc.doc_id == "doc1"
        assert doc.frame_scores["frame_a"] > 0.6
        assert doc.frame_scores["frame_b"] > 0.2
        assert doc.total_weight > 0
    
    def test_multiple_documents(self):
        agg = WeightedFrameAggregator(frame_ids=["frame_a", "frame_b"])
        agg.accumulate("doc1", "text one", {"frame_a": 0.9, "frame_b": 0.1})
        agg.accumulate("doc2", "text two", {"frame_a": 0.1, "frame_b": 0.9})
        
        results = agg.finalize()
        assert len(results) == 2
        assert results[0].doc_id == "doc1"
        assert results[1].doc_id == "doc2"
    
    def test_normalization(self):
        agg = WeightedFrameAggregator(frame_ids=["frame_a", "frame_b"], normalize=True)
        agg.accumulate("doc1", "text", {"frame_a": 0.8, "frame_b": 0.6})
        
        results = agg.finalize()
        doc = results[0]
        # Should sum to 1.0 with normalization
        assert abs(sum(doc.frame_scores.values()) - 1.0) < 0.01
    
    def test_no_normalization(self):
        agg = WeightedFrameAggregator(frame_ids=["frame_a", "frame_b"], normalize=False)
        agg.accumulate("doc1", "text", {"frame_a": 0.8, "frame_b": 0.6})
        
        results = agg.finalize()
        doc = results[0]
        # Without normalization, scores are weighted averages, may not sum to 1.0
        assert sum(doc.frame_scores.values()) > 0
    
    def test_threshold(self):
        agg = WeightedFrameAggregator(frame_ids=["frame_a", "frame_b"], min_threshold=0.5)
        agg.accumulate("doc1", "text", {"frame_a": 0.8, "frame_b": 0.3})
        
        results = agg.finalize()
        doc = results[0]
        assert doc.frame_scores["frame_a"] > 0
        assert doc.frame_scores["frame_b"] == 0.0
    
    def test_weighting_by_length(self):
        agg = WeightedFrameAggregator(frame_ids=["frame_a", "frame_b"])
        agg.accumulate("doc1", "short", {"frame_a": 1.0, "frame_b": 0.0})
        agg.accumulate("doc1", "much longer passage with many words", {"frame_a": 0.0, "frame_b": 1.0})
        
        results = agg.finalize()
        doc = results[0]
        # Second passage has more weight due to length
        assert doc.frame_scores["frame_b"] > doc.frame_scores["frame_a"]
    
    def test_empty_passage_ignored(self):
        agg = WeightedFrameAggregator(frame_ids=["frame_a", "frame_b"])
        agg.accumulate("doc1", "   ", {"frame_a": 1.0, "frame_b": 0.0})
        agg.accumulate("doc1", "valid text", {"frame_a": 0.0, "frame_b": 1.0})
        
        results = agg.finalize()
        assert len(results) == 1
        # Should only count valid passage
        assert results[0].frame_scores["frame_b"] > 0
    
    def test_metadata_preservation(self):
        agg = WeightedFrameAggregator(frame_ids=["frame_a", "frame_b"])
        agg.accumulate(
            "doc1", 
            "text", 
            {"frame_a": 0.5, "frame_b": 0.5},
            published_at="2023-01-01",
            title="Test Title",
            url="https://example.com/article"
        )
        
        results = agg.finalize()
        doc = results[0]
        assert "2023-01-01" in doc.published_at
        assert doc.title == "Test Title"
        assert doc.url == "https://example.com/article"
        assert doc.domain == "example.com"


class TestOccurrenceFrameAggregator:
    """Test OccurrenceFrameAggregator."""
    
    def test_binary_presence(self):
        agg = OccurrenceFrameAggregator(frame_ids=["frame_a", "frame_b"], min_threshold=0.5)
        agg.accumulate("doc1", "passage", {"frame_a": 0.8, "frame_b": 0.3})
        
        results = agg.finalize()
        doc = results[0]
        assert doc.frame_scores["frame_a"] == 1.0
        assert doc.frame_scores["frame_b"] == 0.0
    
    def test_multiple_thresholds(self):
        agg = OccurrenceFrameAggregator(frame_ids=["frame_a", "frame_b"], min_threshold=0.0)
        agg.accumulate("doc1", "passage", {"frame_a": 0.8, "frame_b": 0.2})
        
        results = agg.finalize()
        doc = results[0]
        assert doc.frame_scores["frame_a"] == 1.0
        assert doc.frame_scores["frame_b"] == 1.0
    
    def test_equal_weight_for_all_docs(self):
        agg = OccurrenceFrameAggregator(frame_ids=["frame_a"])
        agg.accumulate("doc1", "short", {"frame_a": 1.0})
        agg.accumulate("doc2", "much longer text with many words", {"frame_a": 1.0})
        
        results = agg.finalize()
        assert results[0].total_weight == 1.0
        assert results[1].total_weight == 1.0
    
    def test_top_frames_filtered(self):
        agg = OccurrenceFrameAggregator(frame_ids=["frame_a", "frame_b"], min_threshold=0.0, top_k=2)
        agg.accumulate("doc1", "passage", {"frame_a": 0.8, "frame_b": 0.5})
        
        results = agg.finalize()
        doc = results[0]
        # Should only have frames above threshold in top_frames
        assert "frame_a" in doc.top_frames
        assert "frame_b" in doc.top_frames

