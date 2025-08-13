"""
Tests for efi_analyser processors
"""

import pytest
from efi_analyser.processors import KeywordExtractorProcessor, TextStatisticsProcessor
from efi_corpus.types import Document


class TestKeywordExtractorProcessor:
    """Test KeywordExtractorProcessor"""
    
    def test_processor_init(self):
        """Test processor initialization"""
        processor = KeywordExtractorProcessor(
            keywords=["CREA", "coal"],
            case_sensitive=False,
            name="test_processor"
        )
        
        assert processor.keywords == ["CREA", "coal"]
        assert processor.case_sensitive is False
        assert processor.name == "test_processor"
    
    def test_processor_case_insensitive(self):
        """Test case-insensitive keyword extraction"""
        processor = KeywordExtractorProcessor(
            keywords=["CREA", "coal"],
            case_sensitive=False
        )
        
        doc = Document(
            doc_id="test",
            url="https://example.com",
            title="Test",
            text="This mentions crea and COAL multiple times. CREA is important.",
            published_at=None,
            language=None,
            meta={}
        )
        
        result = processor.process(doc)
        
        assert result["keyword_counts"]["CREA"] == 2
        assert result["keyword_counts"]["coal"] == 1
        assert result["total_keywords"] == 3
        assert "keyword_positions" in result
    
    def test_processor_case_sensitive(self):
        """Test case-sensitive keyword extraction"""
        processor = KeywordExtractorProcessor(
            keywords=["CREA", "coal"],
            case_sensitive=True
        )
        
        doc = Document(
            doc_id="test",
            url="https://example.com",
            title="Test",
            text="This mentions crea and COAL",
            published_at=None,
            language=None,
            meta={}
        )
        
        result = processor.process(doc)
        
        assert result["keyword_counts"]["CREA"] == 0  # No exact match
        assert result["keyword_counts"]["coal"] == 0  # No exact match
        assert result["total_keywords"] == 0


class TestTextStatisticsProcessor:
    """Test TextStatisticsProcessor"""
    
    def test_processor_init(self):
        """Test processor initialization"""
        processor = TextStatisticsProcessor(name="test_processor")
        assert processor.name == "test_processor"
    
    def test_processor_basic_stats(self):
        """Test basic text statistics"""
        processor = TextStatisticsProcessor()
        
        doc = Document(
            doc_id="test",
            url="https://example.com",
            title="Test",
            text="This is a test document. It has multiple sentences. And more content.",
            published_at=None,
            language=None,
            meta={}
        )
        
        result = processor.process(doc)
        
        # Count words and characters manually to verify
        expected_words = len(doc.text.split())
        expected_chars = len(doc.text)
        assert result["word_count"] == expected_words
        assert result["char_count"] == expected_chars
        assert result["sentence_count"] == 3
        assert result["avg_word_length"] > 0
    
    def test_processor_empty_text(self):
        """Test processor with empty text"""
        processor = TextStatisticsProcessor()
        
        doc = Document(
            doc_id="test",
            url="https://example.com",
            title="Test",
            text="",
            published_at=None,
            language=None,
            meta={}
        )
        
        result = processor.process(doc)
        
        assert result["word_count"] == 0
        assert result["char_count"] == 0
        assert result["sentence_count"] == 0
        assert result["avg_word_length"] == 0
