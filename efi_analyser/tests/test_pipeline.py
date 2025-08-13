"""
Tests for efi_analyser pipeline
"""

import pytest
import tempfile
import json
from pathlib import Path
from efi_analyser.pipeline import AnalysisPipeline
from efi_analyser.filters import TextContainsFilter
from efi_analyser.processors import KeywordExtractorProcessor
from efi_corpus.types import Document


class TestAnalysisPipeline:
    """Test AnalysisPipeline"""
    
    @pytest.fixture
    def temp_corpus(self):
        """Create a temporary corpus for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_path = Path(temp_dir) / "test_corpus"
            corpus_path.mkdir()
            
            # Create minimal corpus structure with current format
            docs_dir = corpus_path / "documents"
            docs_dir.mkdir()
            
            # Create index.jsonl
            index_path = corpus_path / "index.jsonl"
            with open(index_path, 'w') as f:
                f.write(json.dumps({
                    "id": "doc1",
                    "url": "https://example.com/doc1",
                    "title": "Document 1",
                    "published_at": "2025-01-01",
                    "language": "en"
                }) + "\n")
            
            # Create document content with current structure
            # Create two-character prefix directory
            prefix = "do"
            prefix_dir = docs_dir / prefix
            prefix_dir.mkdir()
            
            doc_dir = prefix_dir / "doc1"
            doc_dir.mkdir()
            
            with open(doc_dir / "text.txt", 'w') as f:
                f.write("This document mentions CREA and coal.")
            
            with open(doc_dir / "meta.json", 'w') as f:
                json.dump({"source": "test"}, f)
            
            yield corpus_path
    
    def test_pipeline_init(self, temp_corpus):
        """Test pipeline initialization"""
        pipeline = AnalysisPipeline(
            corpus_path=temp_corpus,
            filters=[],
            processors=[],
            output_path=None
        )
        
        assert pipeline.corpus_path == temp_corpus
        assert len(pipeline.filters) == 0
        assert len(pipeline.processors) == 0
        assert pipeline.output_path is None
    
    def test_pipeline_no_filters_no_processors(self, temp_corpus):
        """Test pipeline with no filters and no processors"""
        pipeline = AnalysisPipeline(
            corpus_path=temp_corpus,
            filters=[],
            processors=[]
        )
        
        results = pipeline.run()
        
        assert len(results) == 1
        assert results[0].passed_filters is True
        assert results[0].processing_results == {}
    
    def test_pipeline_with_filter(self, temp_corpus):
        """Test pipeline with a filter"""
        crea_filter = TextContainsFilter(terms=["CREA"])
        
        pipeline = AnalysisPipeline(
            corpus_path=temp_corpus,
            filters=[crea_filter],
            processors=[]
        )
        
        results = pipeline.run()
        
        assert len(results) == 1
        assert results[0].passed_filters is True
        assert results[0].filter_results["text_contains_1_terms"] is True
    
    def test_pipeline_with_processor(self, temp_corpus):
        """Test pipeline with a processor"""
        processor = KeywordExtractorProcessor(keywords=["CREA", "coal"])
        
        pipeline = AnalysisPipeline(
            corpus_path=temp_corpus,
            filters=[],
            processors=[processor]
        )
        
        results = pipeline.run()
        
        assert len(results) == 1
        assert results[0].passed_filters is True
        assert "keyword_extractor" in results[0].processing_results
        
        proc_result = results[0].processing_results["keyword_extractor"]
        assert proc_result["keyword_counts"]["CREA"] == 1
        assert proc_result["keyword_counts"]["coal"] == 1
    
    def test_pipeline_filter_then_process(self, temp_corpus):
        """Test pipeline with filter then processor"""
        crea_filter = TextContainsFilter(terms=["CREA"])
        processor = KeywordExtractorProcessor(keywords=["coal"])
        
        pipeline = AnalysisPipeline(
            corpus_path=temp_corpus,
            filters=[crea_filter],
            processors=[processor]
        )
        
        results = pipeline.run()
        
        assert len(results) == 1
        assert results[0].passed_filters is True
        assert "keyword_extractor" in results[0].processing_results
        
        # Should have processed the document since it passed the filter
        proc_result = results[0].processing_results["keyword_extractor"]
        assert proc_result["keyword_counts"]["coal"] == 1
    
    def test_pipeline_with_output_path(self, temp_corpus):
        """Test pipeline with output path"""
        output_path = Path(temp_corpus) / "results.json"
        
        pipeline = AnalysisPipeline(
            corpus_path=temp_corpus,
            filters=[],
            processors=[],
            output_path=output_path
        )
        
        results = pipeline.run()
        
        # Check that results were saved
        assert output_path.exists()
        
        with open(output_path, 'r') as f:
            saved_data = json.load(f)
        
        assert "results" in saved_data
        assert "analysis_stats" in saved_data
        assert len(saved_data["results"]) == 1
    
    def test_pipeline_stats(self, temp_corpus):
        """Test pipeline statistics"""
        pipeline = AnalysisPipeline(
            corpus_path=temp_corpus,
            filters=[],
            processors=[]
        )
        
        results = pipeline.run()
        
        stats = pipeline.stats
        assert stats["total_documents"] == 1
        assert stats["passed_filters"] == 1
        assert stats["failed_filters"] == 0
        assert stats["processing_errors"] == 0
        assert stats["start_time"] is not None
        assert stats["end_time"] is not None
    
    def test_pipeline_get_filtered_results(self, temp_corpus):
        """Test getting filtered results"""
        crea_filter = TextContainsFilter(terms=["CREA"])
        
        pipeline = AnalysisPipeline(
            corpus_path=temp_corpus,
            filters=[crea_filter],
            processors=[]
        )
        
        pipeline.run()
        
        filtered = pipeline.get_filtered_results()
        failed = pipeline.get_failed_results()
        
        assert len(filtered) == 1
        assert len(failed) == 0
