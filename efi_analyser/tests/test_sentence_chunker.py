"""Test the sentence chunker with various text patterns and edge cases."""

import pytest
import re
from efi_analyser.chunkers.sentence_chunker import SentenceChunker


class TestSentenceChunker:
    """Test sentence chunker functionality."""

    @pytest.fixture
    def chunker(self):
        """Create a sentence chunker instance."""
        return SentenceChunker(max_chunk_size=500, overlap=50)

    def test_basic_sentence_splitting(self, chunker):
        """Test basic sentence splitting functionality."""
        text = "This is sentence one. This is sentence two. This is sentence three."
        
        chunks = chunker.chunk(text)
        
        # Now each sentence becomes its own chunk
        assert len(chunks) == 3
        assert chunks[0] == "This is sentence one."
        assert chunks[1] == "This is sentence two."
        assert chunks[2] == "This is sentence three."

    def test_original_case_period_no_space(self, chunker):
        """Test the original case: period followed by capital letter without space."""
        text = """Prolonged exposure to PM2.5 is linked to heart disease, respiratory illnesses, and premature death.The CREA analysis shows that ammonium sulfate, a secondary pollutant formed in the atmosphere from SO₂ and NH₃, makes up 34% of the PM2.5 mass on an average across Indian cities."""
        
        chunks = chunker.chunk(text)
        
        # Should split into 2 chunks since there are 2 sentences
        assert len(chunks) == 2
        
        # Check that first chunk ends at sentence boundary
        assert chunks[0].endswith("death.")
        assert chunks[1].startswith("The CREA analysis")

    def test_quote_followed_by_capital(self, chunker):
        """Test case: quote followed by capital letter."""
        text = """The study serves as a wake-up call for policymakers to recalibrate India's clean air strategy, shifting focus from just visible dust to invisible but deadly chemical reactions that are silently poisoning the air."In addition to ammonium sulfate, other secondary pollutants like ammonium nitrate also contribute significantly to PM2.5 mass."""
        
        chunks = chunker.chunk(text)
        
        # Should split into 2 chunks since there are 2 sentences
        assert len(chunks) == 2
        
        # Check that first chunk ends at sentence boundary (period)
        assert chunks[0].endswith('air.')
        assert chunks[1].startswith('"In addition to')

    def test_quote_followed_by_capital_case2(self, chunker):
        """Test another case: quote followed by capital letter."""
        text = """The main driver of ammonium sulphate formation and more than 60% of SO2 emissions in India originate from coal-fired thermal power plants."This makes them a critical target for reducing secondary PM2.5 pollution through the implementation of flue gas desulphurisation (FGD) systems," the report notes."""
        
        chunks = chunker.chunk(text)
        
        # Should split into 2 chunks since there are 2 sentences
        assert len(chunks) == 2
        
        # Check that first chunk ends at sentence boundary (period)
        assert chunks[0].endswith('plants.')
        assert chunks[1].startswith('"This makes them')

    def test_newline_between_sentences(self, chunker):
        """Test newlines between sentences with quotes."""
        text = '''The main driver of ammonium sulphate formation and more than 60% of SO2 emissions in India originate from coal-fired thermal power plants.

"This makes them a critical target for reducing secondary PM2.5 pollution through the implementation of flue gas desulphurisation (FGD) systems," the report notes.'''
        
        chunks = chunker.chunk(text)
        
        # Should split into 2 chunks
        assert len(chunks) == 2
        assert "coal-fired thermal power plants" in chunks[0]
        assert "This makes them a critical target" in chunks[1]
        
        # Verify chunk lengths are reasonable
        assert len(chunks[0]) <= chunker.max_chunk_size
        assert len(chunks[1]) <= chunker.max_chunk_size

    def test_decimal_numbers_not_split(self, chunker):
        """Test that decimal numbers like PM2.5 are not split."""
        text = "The PM2.5 concentration was 34.5 μg/m³. This is high."
        
        chunks = chunker.chunk(text)
        
        # Should split only at the period between sentences, not at decimal points
        assert len(chunks) == 2  # Each sentence becomes its own chunk
        assert "PM2.5" in chunks[0]
        assert "34.5" in chunks[0]
        assert chunks[0] == "The PM2.5 concentration was 34.5 μg/m³."
        assert chunks[1] == "This is high."

    def test_multiple_sentence_endings(self, chunker):
        """Test text with multiple types of sentence endings."""
        text = "First sentence. Second sentence! Third sentence? Fourth sentence."
        
        chunks = chunker.chunk(text)
        
        # Should split at each sentence ending
        assert len(chunks) == 4  # Each sentence becomes its own chunk
        assert chunks[0] == "First sentence."
        assert chunks[1] == "Second sentence!"
        assert chunks[2] == "Third sentence?"
        assert chunks[3] == "Fourth sentence."

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
        """Test that the sentence end patterns correctly identify boundaries."""
        text = "Sentence one. Sentence two."
        
        # Test the patterns directly - they're compiled regex objects
        all_matches = []
        for pattern in chunker.sentence_end_patterns:
            matches = list(pattern.finditer(text))
            all_matches.extend(matches)
        
        assert len(all_matches) >= 1  # At least one pattern should match
        # Check that we found a period followed by space
        found_period_space = any(match.group() == "." and text[match.end():match.end()+1] == " " for match in all_matches)
        assert found_period_space
        

    def test_chunker_spec_includes_pattern(self, chunker):
        """Test that the chunker spec includes the sentence end patterns."""
        spec = chunker.spec
        
        assert spec.name == "sentence"
        assert "sentence_end_patterns" in spec.params
        assert "max_size" in spec.params
        assert "overlap" in spec.params
        
        # The patterns should be included as a list of strings
        patterns = spec.params["sentence_end_patterns"]
        assert isinstance(patterns, list)
        assert len(patterns) > 0
        for pattern_str in patterns:
            assert isinstance(pattern_str, str)
            assert "[" in pattern_str  # Should contain regex pattern

    def test_different_patterns_produce_different_specs(self):
        """Test that different sentence end patterns produce different chunker specs."""
        chunker1 = SentenceChunker(max_chunk_size=200, overlap=50)
        chunker1.sentence_end_patterns = [re.compile(r'[.!?]\s+')]  # Old pattern
        
        chunker2 = SentenceChunker(max_chunk_size=200, overlap=50)
        chunker2.sentence_end_patterns = [re.compile(r'[.!?]["\s]*(?=[A-Z])')]  # New pattern
        
        spec1 = chunker1.spec
        spec2 = chunker2.spec
        
        # Different patterns should produce different specs
        assert spec1.params["sentence_end_patterns"] != spec2.params["sentence_end_patterns"]
        
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

    def test_unicode_quotes_paragraph_break(self, chunker):
        """Handle curly quotes and paragraph break after a quoted sentence."""
        text = (
            "Sunderrajan, coordinator of the environmental advocacy non-profit, Poovulagin Nanbargal "
            "described NEERI’s study as “worrying.” Speaking to The Hindu, he said, “Then it could be "
            "because FGDs are either not working or are not running properly,” adding, “If it is not "
            "bringing down the air pollution, then there could be other sources that emit more sulphur "
            "dioxide in the vicinity of those plants.”\n\n"
            "The Environment Ministry’s note specifically on the potential for emissions states that Indian coal co"
        )
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2
        # First sentence should end at the period of “worrying.”
        assert chunks[0].rstrip().endswith("worrying.")
        # There is a paragraph break; ensure we split before the next paragraph
        assert "The Environment Ministry’s note" in " ".join(chunks[1:])

    def test_double_newline_after_closing_quote(self, chunker):
        """Split when a sentence ends with a closing quote followed by a blank line and a capital."""
        text = (
            "Adding more long-lived CO2 emissions while removing short-lived SO2 emissions by installing FGDs "
            "indiscriminately in all TPPs in India despite the low Sulphur content of Indian coal will enhance "
            "global warming.”\n\n"
            "On the other hand, given that burning coal is India’s primary source of electricity, India’s annual "
            "SO2 emissions has risen from 4,000 kilotonnes in 2010 to 6,000 kilotonnes in 2022."
        )
        chunks = chunker.chunk(text)
        assert len(chunks) == 2
        assert chunks[0].rstrip().endswith("warming.")
        # Next chunk may start with a closing quote depending on split rule; ensure the content starts correctly
        assert "On the other hand" in chunks[1]
