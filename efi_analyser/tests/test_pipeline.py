"""
Tests for efi_analyser pipeline
"""

import pytest
import tempfile
import json
from pathlib import Path
from efi_analyser.pipeline import LinearPipeline
from efi_analyser.filters import TextContainsFilter
from efi_analyser.processors import KeywordExtractorProcessor
from efi_core.types import Document


class TestLinearPipeline:
    """Test LinearPipeline"""
    
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
            
            # Flat layout document directory
            doc_dir = docs_dir / "doc1"
            doc_dir.mkdir()
            
            with open(doc_dir / "text.txt", 'w') as f:
                f.write("This document mentions CREA and coal.")
            
            with open(doc_dir / "meta.json", 'w') as f:
                json.dump({"source": "test"}, f)
            
            yield corpus_path
    
    def test_pipeline_init(self, temp_corpus):
        """Test pipeline initialization"""
        pipeline = LinearPipeline(
            filter_func=None,
            processor_func=None,
            aggregator_func=None
        )
        
        assert pipeline.filter_func is None
        assert pipeline.processor_func is None
        assert pipeline.aggregator_func is None
    
    def test_pipeline_no_filters_no_processors(self, temp_corpus):
        """Test pipeline with no filters and no processors"""
        from efi_corpus import CorpusHandle
        
        pipeline = LinearPipeline(
            filter_func=None,
            processor_func=None,
            aggregator_func=None
        )
        
        corpus_handle = CorpusHandle(temp_corpus)
        pipeline_result = pipeline.run(corpus_handle)
        
        assert pipeline_result.results is not None
        assert len(pipeline_result.results) == 1
        assert pipeline_result.results[0].passed_filters is True
        assert pipeline_result.results[0].processing_results == {}
    
    def test_pipeline_with_filter(self, temp_corpus):
        """Test pipeline with a filter"""
        from efi_corpus import CorpusHandle
        
        def crea_filter(doc):
            return "CREA" in doc.text
        
        pipeline = LinearPipeline(
            filter_func=crea_filter,
            processor_func=None,
            aggregator_func=None
        )
        
        corpus_handle = CorpusHandle(temp_corpus)
        pipeline_result = pipeline.run(corpus_handle)
        
        assert pipeline_result.results is not None
        assert len(pipeline_result.results) == 1
        assert pipeline_result.results[0].passed_filters is True
        assert pipeline_result.results[0].filter_results["filter"] is True
    
    def test_pipeline_with_processor(self, temp_corpus):
        """Test pipeline with a processor"""
        from efi_corpus import CorpusHandle
        
        processor = KeywordExtractorProcessor(keywords=["CREA", "coal"])
        
        pipeline = LinearPipeline(
            filter_func=None,
            processor_func=processor.process,
            aggregator_func=None,
            processor_name="keyword_extractor"
        )
        
        corpus_handle = CorpusHandle(temp_corpus)
        pipeline_result = pipeline.run(corpus_handle)
        
        assert pipeline_result.results is not None
        assert len(pipeline_result.results) == 1
        assert pipeline_result.results[0].passed_filters is True
        assert "keyword_extractor" in pipeline_result.results[0].processing_results

        proc_result = pipeline_result.results[0].processing_results["keyword_extractor"]
        assert proc_result["keyword_counts"]["CREA"] == 1
        assert proc_result["keyword_counts"]["coal"] == 1
    
    def test_pipeline_filter_then_process(self, temp_corpus):
        """Test pipeline with filter then processor"""
        from efi_corpus import CorpusHandle
        
        def crea_filter(doc):
            return "CREA" in doc.text
        
        processor = KeywordExtractorProcessor(keywords=["coal"])
        
        pipeline = LinearPipeline(
            filter_func=crea_filter,
            processor_func=processor.process,
            aggregator_func=None,
            processor_name="keyword_extractor"
        )
        
        corpus_handle = CorpusHandle(temp_corpus)
        pipeline_result = pipeline.run(corpus_handle)
        
        assert pipeline_result.results is not None
        assert len(pipeline_result.results) == 1
        assert pipeline_result.results[0].passed_filters is True
        assert "keyword_extractor" in pipeline_result.results[0].processing_results

        # Should have processed the document since it passed the filter
        proc_result = pipeline_result.results[0].processing_results["keyword_extractor"]
        assert proc_result["keyword_counts"]["coal"] == 1
    
    def test_pipeline_stats(self, temp_corpus):
        """Test pipeline statistics"""
        from efi_corpus import CorpusHandle
        
        pipeline = LinearPipeline(
            filter_func=None,
            processor_func=None,
            aggregator_func=None
        )
        
        corpus_handle = CorpusHandle(temp_corpus)
        pipeline_result = pipeline.run(corpus_handle)
        
        stats = pipeline.stats
        assert stats["total_documents"] == 1
        assert stats["passed_filters"] == 1
        assert stats["failed_filters"] == 0
        assert stats["processing_errors"] == 0
        assert stats["start_time"] is not None
        assert stats["end_time"] is not None
    
    def test_pipeline_get_filtered_results(self, temp_corpus):
        """Test getting filtered results"""
        from efi_corpus import CorpusHandle
        
        def crea_filter(doc):
            return "CREA" in doc.text
        
        pipeline = LinearPipeline(
            filter_func=crea_filter,
            processor_func=None,
            aggregator_func=None
        )
        
        corpus_handle = CorpusHandle(temp_corpus)
        pipeline.run(corpus_handle)
        
        filtered = pipeline.get_filtered_results()
        failed = pipeline.get_failed_results()
        
        assert len(filtered) == 1
        assert len(failed) == 0
