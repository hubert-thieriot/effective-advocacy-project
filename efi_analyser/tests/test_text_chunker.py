"""
Tests for TextChunker with linguistic analysis and content-aware heuristics.
"""

import pytest
from efi_analyser.chunkers.text_chunker import TextChunker, TextChunkerConfig


class TestTextChunker:
    """Test TextChunker functionality with various heuristics."""

    @pytest.fixture
    def chunker(self):
        """Create a TextChunker instance for testing."""
        config = TextChunkerConfig(max_words=50)
        return TextChunker(config=config)

    def test_chunker_initialization(self):
        """Test chunker initialization with different parameters."""
        config = TextChunkerConfig(max_words=100, spacy_model="en_core_web_sm")
        chunker = TextChunker(config=config)
        assert chunker.max_words == 100
        assert chunker.spacy_model == "en_core_web_sm"

        # Test spec property
        spec = chunker.spec
        assert spec.name == "text"
        assert spec.params["max_words"] == 100
        assert spec.params["spacy_model"] == "en_core_web_sm"

    def test_empty_text(self, chunker):
        """Test chunking empty or whitespace-only text."""
        assert chunker.chunk("") == []
        assert chunker.chunk("   ") == []
        assert chunker.chunk("\n\n") == []

    def test_basic_chunking(self, chunker):
        """Test basic chunking functionality."""
        text = "This is a simple sentence. This is another sentence. This is a third sentence."
        chunks = chunker.chunk(text)

        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        # Should combine sentences that fit within word limit
        combined = " ".join(chunks)
        assert "simple sentence" in combined

    def test_paragraph_boundaries(self, chunker):
        """Test paragraph boundary detection (\n\n)."""
        # Create paragraphs where the first one exceeds half max_words
        text = """This is a longer first paragraph that contains many words and should exceed half of the maximum word limit for testing purposes.

Second paragraph with different content.

Third paragraph here."""

        chunks = chunker.chunk(text)
        # First paragraph should be in its own chunk since it exceeds half max_words
        assert len(chunks) >= 2

    def test_quote_attribution_preservation(self, chunker):
        """Test quote attribution patterns are preserved."""
        text = '"This is a very important quote," said the official. This additional sentence should be separate.'

        chunks = chunker.chunk(text)
        # Quote and attribution should be together
        assert len(chunks) >= 1
        first_chunk = chunks[0]
        assert '"This is a very important quote," said the official.' in first_chunk

    def test_indirect_attribution(self, chunker):
        """Test indirect attribution patterns."""
        text = '"Environmental regulations are crucial," according to experts. The report continues with more details.'

        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
        first_chunk = chunks[0]
        assert '"Environmental regulations are crucial," according to experts.' in first_chunk

    def test_discourse_connectors(self, chunker):
        """Test discourse connector attachment."""
        text = "The project is moving forward. However, there are some challenges. Despite these issues, we continue."

        chunks = chunker.chunk(text)
        # "However" sentence should attach to previous
        combined = " ".join(chunks)
        assert "forward. However, there are some challenges." in combined

    def test_short_sentence_merging(self, chunker):
        """Test short sentence merging logic."""
        text = "This is a longer sentence with many words. Short. Another short one."

        chunks = chunker.chunk(text)
        # Short sentences should merge with neighbors when appropriate
        combined = " ".join(chunks)
        assert "many words. Short. Another short one." in combined

    def test_overflow_tolerance(self, chunker):
        """Test 10% overflow tolerance for attach heuristics."""
        # Create text that would slightly exceed limit with discourse connector
        text = "This is a sentence. " * 8  # ~40 words
        text += "However, this additional point is important."

        chunks = chunker.chunk(text)
        # Should allow slight overflow for discourse connector
        assert len(chunks) >= 1

    def test_word_count_accuracy(self, chunker):
        """Test word counting is accurate."""
        text = "This is a test with exactly ten words in it."
        word_count = chunker._count_words(text)
        assert word_count == 10

        text = "One two three."
        word_count = chunker._count_words(text)
        assert word_count == 3

    def test_discourse_connector_detection(self, chunker):
        """Test discourse connector detection."""
        assert chunker._is_discourse_connector("However, this is important.")
        assert chunker._is_discourse_connector("But there are exceptions.")
        assert chunker._is_discourse_connector("Therefore, we proceed.")
        assert not chunker._is_discourse_connector("This is just a regular sentence.")

    def test_attribution_detection(self, chunker):
        """Test quote attribution pattern detection."""
        assert chunker._is_quote_attribution('"Quote here," said John.')
        assert chunker._is_quote_attribution('"Another quote," according to experts.')
        assert chunker._is_quote_attribution('"Third quote," explained the spokesperson.')
        assert not chunker._is_quote_attribution('This is just a regular sentence.')

    def test_text_chunker_config(self):
        """Test TextChunkerConfig functionality."""
        # Test default config
        config = TextChunkerConfig()
        assert config.max_words == 80
        assert config.spacy_model == "en_core_web_sm"

        # Test custom config
        custom_config = TextChunkerConfig(max_words=100, spacy_model="en_core_web_sm")
        assert custom_config.max_words == 100
        assert custom_config.spacy_model == "en_core_web_sm"

        # Test that chunker uses config correctly
        chunker = TextChunker(config=custom_config)
        assert chunker.config.max_words == 100
        assert chunker.config.spacy_model == "en_core_web_sm"

    def test_max_words_constraint(self, chunker):
        """Test that chunks respect max_words constraint."""
        # Create a very long sentence
        long_sentence = "word " * 60 + "end."
        text = long_sentence + " Short sentence."

        chunks = chunker.chunk(text)

        # First chunk should be truncated to respect max_words
        first_chunk_words = chunker._count_words(chunks[0])
        assert first_chunk_words <= chunker.max_words

    def test_multiple_paragraphs(self, chunker):
        """Test handling multiple paragraphs with different content."""
        text = """First paragraph with environmental content and many additional words to make it longer and exceed the half max words limit for proper testing.

Second paragraph about climate change.

Third paragraph discussing policy."""

        chunks = chunker.chunk(text)
        assert len(chunks) >= 2

        # Each chunk should contain content from its respective paragraph
        combined = " ".join(chunks)
        assert "environmental content" in combined
        assert "climate change" in combined

    def test_sentence_splitting(self, chunker):
        """Test that spaCy sentence splitting works correctly."""
        text = "First sentence! Second sentence? Third sentence."
        sentences = chunker._split_into_sentences(text)

        assert len(sentences) == 3
        assert "First sentence!" in sentences
        assert "Second sentence?" in sentences
        assert "Third sentence." in sentences

    def test_complex_quote_patterns(self, chunker):
        """Test complex quote and attribution patterns."""
        text = '"We must act now," stated the president. "Climate change is real," added the scientist. "The data shows this," noted the researcher.'

        chunks = chunker.chunk(text)

        # Each quote-attribution pair should be preserved
        combined = " ".join(chunks)
        assert '"We must act now," stated the president.' in combined
        assert '"Climate change is real," added the scientist.' in combined

    def test_edge_case_punctuation(self, chunker):
        """Test edge cases with punctuation."""
        text = "Sentence with quotes: 'Hello world,' he said. Another sentence!"

        chunks = chunker.chunk(text)
        assert len(chunks) >= 1

        combined = " ".join(chunks)
        assert "'Hello world,' he said." in combined

    def test_long_quote_attribution_preservation(self, chunker):
        """Test that long quotes with attribution are kept together."""
        text = 'IANS "Instead, the evidence points to a temporary adjustment phase followed by stronger inflows. While an immediate reduction in rates can cause a short-term dip of around 3–4 per cent month-on-month (roughly Rs 5,000 crore, or an annualised Rs 60,000 crore), revenues typically rebound with sustained growth of 5–6 per cent per month," the report mentioned.'

        chunks = chunker.chunk(text)

        # The quote and its attribution should be in the same chunk
        combined = " ".join(chunks)
        assert '"Instead, the evidence points to a temporary adjustment phase followed by stronger inflows. While an immediate reduction in rates can cause a short-term dip of around 3–4 per cent month-on-month (roughly Rs 5,000 crore, or an annualised Rs 60,000 crore), revenues typically rebound with sustained growth of 5–6 per cent per month," the report mentioned.' in combined

        # Check that no chunk contains just the attribution part
        for chunk in chunks:
            assert not chunk.strip().startswith('er month,"')
            assert not chunk.strip().endswith('cent p')
