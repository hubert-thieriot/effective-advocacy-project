"""Test the sentence chunker with various text patterns and edge cases."""

import pytest
import re
from efi_analyser.chunkers.sentence_chunker import SentenceChunker


class TestSentenceChunker:
    """Test sentence chunker functionality."""

    @pytest.fixture
    def chunker(self):
        """Create a sentence chunker instance."""
        return SentenceChunker(max_chunk_size=200, overlap=50)

    def test_basic_sentence_splitting(self, chunker):
        """Test basic sentence splitting functionality."""
        text = "This is sentence one. This is sentence two. This is sentence three."
        
        chunks = chunker.chunk(text)
        
        assert len(chunks) == 1  # All sentences fit in one chunk
        assert chunks[0] == text

    def test_original_case_period_no_space(self, chunker):
        """Test the original case: period followed by capital letter without space."""
        text = """Prolonged exposure to PM2.5 is linked to heart disease, respiratory illnesses, and premature death.The CREA analysis shows that ammonium sulfate, a secondary pollutant formed in the atmosphere from SO₂ and NH₃, makes up 34% of the PM2.5 mass on an average across Indian cities."""
        
        chunks = chunker.chunk(text)
        
        # Should split into 2 chunks since text is 277 characters > 200 limit
        assert len(chunks) == 2
        assert len(chunks[0]) <= 200
        assert len(chunks[1]) <= 200
        
        # Check that first chunk ends at sentence boundary
        assert chunks[0].endswith("death.")
        assert chunks[1].startswith("The CREA analysis")

    def test_quote_followed_by_capital(self, chunker):
        """Test case: quote followed by capital letter."""
        text = """The study serves as a wake-up call for policymakers to recalibrate India's clean air strategy, shifting focus from just visible dust to invisible but deadly chemical reactions that are silently poisoning the air."In addition to ammonium sulfate, other secondary pollutants like ammonium nitrate also contribute significantly to PM2.5 mass."""
        
        chunks = chunker.chunk(text)
        
        # Should split into 2 chunks since text is 339 characters > 200 limit
        assert len(chunks) == 2
        
        # Check that chunks respect size limits (allowing for some flexibility)
        for chunk in chunks:
            assert len(chunk) <= 250  # Allow some flexibility around the 200 limit
        
        # Check that first chunk ends at quote boundary
        assert chunks[0].endswith('air."')
        assert chunks[1].startswith("In addition to")

    def test_quote_followed_by_capital_case2(self, chunker):
        """Test another case: quote followed by capital letter."""
        text = """The main driver of ammonium sulphate formation and more than 60% of SO2 emissions in India originate from coal-fired thermal power plants."This makes them a critical target for reducing secondary PM2.5 pollution through the implementation of flue gas desulphurisation (FGD) systems," the report notes."""
        
        chunks = chunker.chunk(text)
        
        # Should split into 2 chunks since text is 301 characters > 200 limit
        assert len(chunks) == 2
        
        # Check that chunks respect size limits (allowing for some flexibility)
        for chunk in chunks:
            assert len(chunk) <= 250  # Allow some flexibility around the 200 limit
        
        # Check that first chunk ends at quote boundary
        assert chunks[0].endswith('plants."')
        assert chunks[1].startswith("This makes them")

    def test_decimal_numbers_not_split(self, chunker):
        """Test that decimal numbers like PM2.5 are not split."""
        text = "The PM2.5 concentration was 34.5 μg/m³. This is high."
        
        chunks = chunker.chunk(text)
        
        # Should split only at the period between sentences, not at decimal points
        assert len(chunks) == 1  # Both sentences should fit in one chunk
        assert "PM2.5" in chunks[0]
        assert "34.5" in chunks[0]

    def test_multiple_sentence_endings(self, chunker):
        """Test text with multiple types of sentence endings."""
        text = "First sentence. Second sentence! Third sentence? Fourth sentence."
        
        chunks = chunker.chunk(text)
        
        # Should split at each sentence ending
        assert len(chunks) == 1  # All sentences should fit in one chunk
        assert chunks[0] == text

    def test_empty_text(self, chunker):
        """Test handling of empty text."""
        chunks = chunker.chunk("")
        assert chunks == []
        
        chunks = chunker.chunk("   ")
        assert chunks == []

    def test_single_sentence(self, chunker):
        """Test single sentence that fits in chunk size."""
        text = "This is a single sentence that should fit in the chunk."
        chunks = chunker.chunk(text)
        
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_sentence_end_pattern_matching(self, chunker):
        """Test that the sentence end pattern correctly identifies boundaries."""
        text = "Sentence one. Sentence two."
        
        # Test the pattern directly - it's a compiled regex object
        pattern = chunker.sentence_end_pattern
        matches = list(pattern.finditer(text))
        
        assert len(matches) == 1
        assert matches[0].group() == ". "  # The pattern matches period + space
        

    def test_chunker_spec_includes_pattern(self, chunker):
        """Test that the chunker spec includes the sentence end pattern."""
        spec = chunker.spec
        
        assert spec.name == "sentence"
        assert "sentence_end_pattern" in spec.params
        assert "max_size" in spec.params
        assert "overlap" in spec.params
        
        # The pattern should be included as a string
        pattern_str = spec.params["sentence_end_pattern"]
        assert isinstance(pattern_str, str)
        assert "[" in pattern_str  # Should contain regex pattern

    def test_different_patterns_produce_different_specs(self):
        """Test that different sentence end patterns produce different chunker specs."""
        chunker1 = SentenceChunker(max_chunk_size=200, overlap=50)
        chunker1.sentence_end_pattern = re.compile(r'[.!?]\s+')  # Old pattern
        
        chunker2 = SentenceChunker(max_chunk_size=200, overlap=50)
        chunker2.sentence_end_pattern = re.compile(r'[.!?]["\s]*(?=[A-Z])')  # New pattern
        
        spec1 = chunker1.spec
        spec2 = chunker2.spec
        
        # Different patterns should produce different specs
        assert spec1.params["sentence_end_pattern"] != spec2.params["sentence_end_pattern"]
        
        # Different specs should have different keys
        assert spec1.key() != spec2.key()

    def test_chunk_size_respect(self, chunker):
        """Test that chunks respect the maximum size limit."""
        # Create a very long text that should definitely be split
        long_text = "This is a very long sentence. " * 50  # Much longer than 200 chars
        
        chunks = chunker.chunk(long_text)
        
        assert len(chunks) > 1  # Should be split into multiple chunks
        
        # Each chunk should respect the size limit
        for chunk in chunks:
            assert len(chunk) <= chunker.max_chunk_size
