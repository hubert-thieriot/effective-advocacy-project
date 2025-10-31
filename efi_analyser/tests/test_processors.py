"""
Tests for efi_analyser processors
"""

import pytest
from efi_analyser.processors import KeywordExtractorProcessor, TextStatisticsProcessor
from efi_core.types import Document


class TestKeywordExtractorProcessor:
    """Test KeywordExtractorProcessor"""
    
    def test_processor_init(self):
        """Test processor initialization"""
        processor = KeywordExtractorProcessor(
            keywords=["CREA", "coal"],
            case_sensitive=False,
            whole_word_only=True,
            allow_hyphenation=True,
            name="test_processor"
        )
        
        assert processor.keywords == ["CREA", "coal"]
        assert processor.case_sensitive is False
        assert processor.whole_word_only is True
        assert processor.allow_hyphenation is True
        assert processor.name == "test_processor"
    
    def test_processor_case_insensitive(self):
        """Test case-insensitive keyword extraction"""
        processor = KeywordExtractorProcessor(
            keywords=["CREA", "coal"],
            case_sensitive=False,
            whole_word_only=True,
            allow_hyphenation=True
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
        assert result["whole_word_only"] is True
        assert result["case_sensitive"] is False
        assert "keyword_positions" in result
    
    def test_processor_case_sensitive(self):
        """Test case-sensitive keyword extraction"""
        processor = KeywordExtractorProcessor(
            keywords=["CREA", "coal"],
            case_sensitive=True,
            whole_word_only=True,
            allow_hyphenation=True
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
        assert result["whole_word_only"] is True
        assert result["case_sensitive"] is True
    
    def test_whole_word_matching_crea(self):
        """Test that CREA only matches whole words, not partial matches"""
        processor = KeywordExtractorProcessor(
            keywords=["CREA"],
            case_sensitive=False,
            whole_word_only=True
        )
        
        doc = Document(
            doc_id="test",
            url="https://example.com",
            title="Test",
            text="CREA is an organization. The creation of documents. We created a report. CREA's mission.",
            published_at=None,
            language=None,
            meta={}
        )
        
        result = processor.process(doc)
        
        # Should only match "CREA" and "CREA's" (with apostrophe as word boundary)
        assert result["keyword_counts"]["CREA"] == 2
        assert result["total_keywords"] == 2
        assert result["whole_word_only"] is True
    
    def test_whole_word_matching_various_patterns(self):
        """Test whole word matching with various word boundary patterns"""
        processor = KeywordExtractorProcessor(
            keywords=["CREA", "coal", "transport"],
            case_sensitive=False,
            whole_word_only=True
        )
        
        doc = Document(
            doc_id="test",
            url="https://example.com",
            title="Test",
            text="CREA, coal, transport. CREA's coal-transport system. transportation of coal.",
            published_at=None,
            language=None,
            meta={}
        )
        
        result = processor.process(doc)
        
        # Should match whole words only
        assert result["keyword_counts"]["CREA"] == 2  # "CREA" and "CREA's"
        assert result["keyword_counts"]["coal"] == 3  # "coal", "coal-transport" (hyphen creates word boundary)
        assert result["keyword_counts"]["transport"] == 2  # "transport" and "coal-transport" (hyphen creates word boundary)
        assert result["total_keywords"] == 7
    
    def test_substring_matching_disabled(self):
        """Test that substring matching is disabled when whole_word_only=True"""
        processor = KeywordExtractorProcessor(
            keywords=["CREA"],
            case_sensitive=False,
            whole_word_only=True
        )
        
        doc = Document(
            doc_id="test",
            url="https://example.com",
            title="Test",
            text="created creation creative recreate",
            published_at=None,
            language=None,
            meta={}
        )
        
        result = processor.process(doc)
        
        # Should not match any partial words
        assert result["keyword_counts"]["CREA"] == 0
        assert result["total_keywords"] == 0
    
    def test_substring_matching_enabled(self):
        """Test substring matching when whole_word_only=False"""
        processor = KeywordExtractorProcessor(
            keywords=["CREA"],
            case_sensitive=False,
            whole_word_only=False
        )
        
        doc = Document(
            doc_id="test",
            url="https://example.com",
            title="Test",
            text="created creation creative recreate",
            published_at=None,
            language=None,
            meta={}
        )
        
        result = processor.process(doc)
        
        # Should match partial words
        assert result["keyword_counts"]["CREA"] == 4
        assert result["total_keywords"] == 4
        assert result["whole_word_only"] is False
    
    def test_word_boundaries_with_punctuation(self):
        """Test word boundaries with various punctuation marks"""
        processor = KeywordExtractorProcessor(
            keywords=["CREA"],
            case_sensitive=False,
            whole_word_only=True
        )
        
        doc = Document(
            doc_id="test",
            url="https://example.com",
            title="Test",
            text="CREA. CREA! CREA? CREA: CREA; CREA, CREA.",
            published_at=None,
            language=None,
            meta={}
        )
        
        result = processor.process(doc)
        
        # Should match all instances as they're all whole words
        assert result["keyword_counts"]["CREA"] == 7
        assert result["total_keywords"] == 7
    
    def test_word_boundaries_with_numbers(self):
        """Test word boundaries with numbers and common false positives"""
        processor = KeywordExtractorProcessor(
            keywords=["CREA"],
            case_sensitive=False,
            whole_word_only=True
        )
        
        doc = Document(
            doc_id="test",
            url="https://example.com",
            title="Test",
            text="CREA2023 CREA_2023 CREA-2023 CREA2023report created creation creative",
            published_at=None,
            language=None,
            meta={}
        )
        
        result = processor.process(doc)
        
        # Regex word boundaries \b only match at transitions between word and non-word characters
        # "CREA2023" - no match (letter-number transition)
        # "CREA_2023" - no match (letter-underscore transition)  
        # "CREA-2023" - MATCHES (letter-hyphen transition)
        # "CREA2023report" - no match (letter-number transition)
        # "created", "creation", "creative" - no matches (whole word only)
        assert result["keyword_counts"]["CREA"] == 1  # Only "CREA-2023"
        assert result["total_keywords"] == 1
    
    def test_empty_text(self):
        """Test processor with empty text"""
        processor = KeywordExtractorProcessor(
            keywords=["CREA", "coal"],
            case_sensitive=False,
            whole_word_only=True
        )
        
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
        
        assert result["keyword_counts"]["CREA"] == 0
        assert result["keyword_counts"]["coal"] == 0
        assert result["total_keywords"] == 0
        assert result["whole_word_only"] is True
    
    def test_no_keywords(self):
        """Test processor with no keywords"""
        processor = KeywordExtractorProcessor(
            keywords=[],
            case_sensitive=False,
            whole_word_only=True
        )
        
        doc = Document(
            doc_id="test",
            url="https://example.com",
            title="Test",
            text="This is some text with CREA and coal.",
            published_at=None,
            language=None,
            meta={}
        )
        
        result = processor.process(doc)
        
        assert result["keyword_counts"] == {}
        assert result["total_keywords"] == 0
        assert result["whole_word_only"] is True
    
    def test_hyphenated_words(self):
        """Test that keywords can match across hyphens (common in newspaper formatting)"""
        processor = KeywordExtractorProcessor(
            keywords=["CREA", "coal", "transport"],
            case_sensitive=False,
            whole_word_only=True,
            allow_hyphenation=True
        )
        
        # Test various hyphenation patterns
        doc = Document(
            doc_id="test",
            url="https://example.com",
            title="Test",
            text="CREA-related research. Coal-mining industry. Transport-ation sector. CREA-",
            published_at=None,
            language=None,
            meta={}
        )
        
        result = processor.process(doc)
        
        # Should match hyphenated words
        assert result["keyword_counts"]["CREA"] == 2  # "CREA-related" and "CREA-"
        assert result["keyword_counts"]["coal"] == 1  # "Coal-mining"
        assert result["keyword_counts"]["transport"] == 1  # "Transport-ation"
        assert result["total_keywords"] == 4
        assert result["allow_hyphenation"] is True
    
    def test_line_break_hyphenation(self):
        """Test that keywords can match across line breaks with hyphens"""
        processor = KeywordExtractorProcessor(
            keywords=["CREA", "coal"],
            case_sensitive=False,
            whole_word_only=True,
            allow_hyphenation=True
        )
        
        # Simulate text with line breaks and hyphens
        doc = Document(
            doc_id="test",
            url="https://example.com",
            title="Test",
            text="The CREA-\nrelated findings show that coal-\nmining affects air quality.",
            published_at=None,
            language=None,
            meta={}
        )
        
        result = processor.process(doc)
        
        # Should match across line breaks
        assert result["keyword_counts"]["CREA"] == 1  # "CREA-\nrelated"
        assert result["keyword_counts"]["coal"] == 1  # "coal-\nmining"
        assert result["total_keywords"] == 2
        assert result["allow_hyphenation"] is True
    
    def test_mixed_hyphenation_patterns(self):
        """Test various hyphenation patterns that might occur in newspapers"""
        processor = KeywordExtractorProcessor(
            keywords=["CREA", "coal"],
            case_sensitive=False,
            whole_word_only=True,
            allow_hyphenation=True
        )
        
        doc = Document(
            doc_id="test",
            url="https://example.com",
            title="Test",
            text="CREA-2023 report. Coal-fired plants. CREA-\nbased analysis. Coal-\nmining operations.",
            published_at=None,
            language=None,
            meta={}
        )
        
        result = processor.process(doc)
        
        # Should match various hyphenation patterns
        assert result["keyword_counts"]["CREA"] == 2  # "CREA-2023" and "CREA-\nbased"
        assert result["keyword_counts"]["coal"] == 2  # "Coal-fired" and "Coal-\nmining"
        assert result["total_keywords"] == 4
        assert result["allow_hyphenation"] is True
    
    def test_hyphenation_with_substring_matching(self):
        """Test hyphenation behavior when whole_word_only=False"""
        processor = KeywordExtractorProcessor(
            keywords=["CREA"],
            case_sensitive=False,
            whole_word_only=False,
            allow_hyphenation=True
        )
        
        doc = Document(
            doc_id="test",
            url="https://example.com",
            title="Test",
            text="CREA-related CREA-\nbased CREA-2023",
            published_at=None,
            language=None,
            meta={}
        )
        
        result = processor.process(doc)
        
        # Should match all instances including hyphenated ones
        assert result["keyword_counts"]["CREA"] == 3
        assert result["total_keywords"] == 3
        assert result["whole_word_only"] is False
        assert result["allow_hyphenation"] is True
    
    def test_keyword_spanning_hyphens(self):
        """Test that keywords like 'transport' can match 'tran-sport' or 'tran\nsport'"""
        processor = KeywordExtractorProcessor(
            keywords=["transport", "CREA"],
            case_sensitive=False,
            whole_word_only=True,
            allow_hyphenation=True
        )
        
        doc = Document(
            doc_id="test",
            url="https://example.com",
            title="Test",
            text="The tran-sport system. Tran\nsport infrastructure. CREA-\nrelated research.",
            published_at=None,
            language=None,
            meta={}
        )
        
        result = processor.process(doc)
        
        # Should match keywords that span across hyphens and line breaks
        assert result["keyword_counts"]["transport"] == 2  # "tran-sport" and "Tran\nsport"
        assert result["keyword_counts"]["CREA"] == 1  # "CREA-\nrelated"
        assert result["total_keywords"] == 3
        assert result["allow_hyphenation"] is True
    
    def test_hyphenation_disabled(self):
        """Test that hyphenation can be disabled"""
        processor = KeywordExtractorProcessor(
            keywords=["transport", "CREA"],
            case_sensitive=False,
            whole_word_only=True,
            allow_hyphenation=False
        )
        
        doc = Document(
            doc_id="test",
            url="https://example.com",
            title="Test",
            text="The tran-sport system. Tran\nsport infrastructure. CREA-\nrelated research.",
            published_at=None,
            language=None,
            meta={}
        )
        
        result = processor.process(doc)
        
        # When hyphenation is disabled:
        # - "CREA" should still match as a whole word (it's not split, just has trailing hyphen)
        # - "transport" should NOT match because it's split across hyphens/line breaks
        assert result["keyword_counts"]["transport"] == 0  # No matches for split words
        assert result["keyword_counts"]["CREA"] == 1  # "CREA" is a whole word, not split
        assert result["total_keywords"] == 1
        assert result["allow_hyphenation"] is False
    
    def test_various_split_points(self):
        """Test different ways keywords can be split"""
        processor = KeywordExtractorProcessor(
            keywords=["transport", "analysis"],
            case_sensitive=False,
            whole_word_only=True,
            allow_hyphenation=True
        )
        
        doc = Document(
            doc_id="test",
            url="https://example.com",
            title="Test",
            text="tran-sport trans-port transp-ort anal-ysis analy-sis",
            published_at=None,
            language=None,
            meta={}
        )
        
        result = processor.process(doc)
        
        # Should match various split points
        assert result["keyword_counts"]["transport"] == 3  # "tran-sport", "trans-port", "transp-ort"
        assert result["keyword_counts"]["analysis"] == 2  # "anal-ysis", "analy-sis"
        assert result["total_keywords"] == 5
        assert result["allow_hyphenation"] is True


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


class TestKeywordExtractorPatterns:
    """Tests for raw regex pattern support in KeywordExtractorProcessor."""

    def test_raw_patterns_combined_matching(self):
        processor = KeywordExtractorProcessor(
            keywords=["transport"],
            case_sensitive=False,
            whole_word_only=True,
            allow_hyphenation=True,
            patterns=[
                r"(?i)(konstruksi|pembangunan|pekerjaan\s+jalan).{0,80}(debu|polusi|partikel)",
                r"(?i)resuspensi\s+debu",
            ],
        )

        doc = Document(
            doc_id="doc1",
            url="https://example.com",
            title="Test",
            text=(
                "Pemerintah melakukan pembangunan jalan utama di Jakarta yang menyebabkan partikel dan debu "
                "berterbangan selama musim kemarau. Selain itu, ada resuspensi debu dari lalu lintas."
            ),
            published_at=None,
            language=None,
            meta={},
        )

        result = processor.process(doc)
        # Keyword 'transport' should not match here
        assert result["keyword_counts"].get("transport", 0) == 0
        # Raw patterns should match
        assert any(
            result["keyword_counts"].get(pat, 0) > 0
            for pat in processor._custom_patterns  # type: ignore[attr-defined]
        )

    def test_patterns_do_not_require_whole_word_flags(self):
        """Patterns are compiled as-is; whole_word_only/allow_hyphenation do not alter them."""
        processor = KeywordExtractorProcessor(
            keywords=[],
            case_sensitive=False,
            whole_word_only=True,
            allow_hyphenation=False,
            patterns=[r"(?i)semen.{0,40}debu|debu.{0,40}semen"],
        )

        doc = Document(
            doc_id="doc2",
            url="https://example.com",
            title="Test",
            text="Pabrik semen di pinggiran kota menimbulkan banyak debu pada siang hari.",
            published_at=None,
            language=None,
            meta={},
        )

        result = processor.process(doc)
        # Should match via raw pattern regardless of whole_word_only/hyphenation
        pat = processor._custom_patterns[0]  # type: ignore[attr-defined]
        assert result["keyword_counts"].get(pat, 0) >= 1
