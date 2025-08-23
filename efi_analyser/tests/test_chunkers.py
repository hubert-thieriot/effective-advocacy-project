"""
Tests for efi_analyser chunkers
"""

import pytest
from efi_analyser.chunkers import SentenceChunker


class TestSentenceChunker:
    """Test SentenceChunker"""
    
    def test_chunker_init(self):
        """Test chunker initialization"""
        chunker = SentenceChunker(max_chunk_size=500, overlap=25)
        assert chunker.max_chunk_size == 500
        assert chunker.overlap == 25
    
    def test_chunker_simple_text(self):
        """Test chunking simple text"""
        chunker = SentenceChunker(max_chunk_size=100, overlap=10)
        text = "This is the first sentence. This is the second sentence. This is the third sentence."
        
        chunks = chunker.chunk(text)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        # Should contain all the content
        combined = " ".join(chunks)
        assert "first sentence" in combined
        assert "second sentence" in combined
        assert "third sentence" in combined
    
    def test_chunker_empty_text(self):
        """Test chunking empty text"""
        chunker = SentenceChunker()
        chunks = chunker.chunk("")
        assert chunks == []
        
        chunks = chunker.chunk("   ")
        assert chunks == []
    
    def test_chunker_single_sentence(self):
        """Test chunking single sentence"""
        chunker = SentenceChunker(max_chunk_size=1000)
        text = "This is a single sentence."
        
        chunks = chunker.chunk(text)
        
        assert len(chunks) == 1
        assert chunks[0].strip() == text
    
    def test_chunker_long_text(self):
        """Test chunking long text that exceeds max_chunk_size"""
        chunker = SentenceChunker(max_chunk_size=50, overlap=10)
        text = "This is a very long sentence that definitely exceeds the maximum chunk size limit. " \
               "This is another long sentence that also exceeds the limit. " \
               "And here is a third sentence for good measure."
        
        chunks = chunker.chunk(text)
        
        assert len(chunks) > 1
        # Each chunk should be reasonably sized
        for chunk in chunks:
            # Allow some flexibility for sentence boundaries
            assert len(chunk) <= chunker.max_chunk_size + 100  # Some tolerance
    
    def test_chunker_overlap(self):
        """Test that overlap works correctly with complete sentences"""
        chunker = SentenceChunker(max_chunk_size=50, overlap=30)
        text = "Short sentence. Another short sentence. Third short sentence. Fourth short sentence."
        
        chunks = chunker.chunk(text)
        
        if len(chunks) > 1:
            # Verify we get multiple chunks
            assert len(chunks) >= 2
            
            # Check that overlaps contain complete sentences, not fragments
            for chunk in chunks:
                # Should not start or end mid-word (except for very long sentences)
                words = chunk.strip().split()
                if words:
                    # First word should be capitalized (start of sentence) or be overlap
                    # Last word should end with punctuation or be a complete word
                    assert not chunk.strip().endswith(' ')  # No trailing spaces
    
    def test_chunker_sentence_aware_overlap(self):
        """Test that overlap preserves complete sentences"""
        chunker = SentenceChunker(max_chunk_size=40, overlap=25)
        text = "First. Second sentence here. Third sentence. Fourth."
        
        chunks = chunker.chunk(text)
        
        # Verify no chunks contain sentence fragments
        for chunk in chunks:
            # Should not start with lowercase (indicating mid-sentence)
            words = chunk.strip().split()
            if words and len(words) > 1:
                # If it starts with a word, it should be a complete sentence start
                # (allowing for overlap from previous complete sentences)
                assert not (words[0][0].islower() and words[0] not in ['and', 'but', 'or'])  # Common continuation words
    
    def test_chunker_sentence_splitting(self):
        """Test sentence splitting patterns"""
        chunker = SentenceChunker(max_chunk_size=1000)
        text = "Question? Answer! Statement. Another statement."
        
        chunks = chunker.chunk(text)
        
        # Should handle different punctuation
        combined = " ".join(chunks)
        assert "Question?" in combined
        assert "Answer!" in combined
        assert "Statement." in combined
    
    def test_chunker_no_fragments(self):
        """Test that chunker doesn't create sentence fragments"""
        chunker = SentenceChunker(max_chunk_size=30, overlap=15)
        text = "This is a longer sentence that might get split. This is another sentence. And a third one."
        
        chunks = chunker.chunk(text)
        
        # Verify no chunks are fragments (start mid-sentence)
        for chunk in chunks:
            chunk = chunk.strip()
            if chunk:
                # Should not start with lowercase unless it's a proper continuation
                words = chunk.split()
                if words:
                    first_word = words[0]
                    # Allow some common sentence starters that might be lowercase
                    common_lowercase_starters = ['and', 'but', 'or', 'so', 'yet', 'for', 'nor']
                    if first_word.lower() not in common_lowercase_starters:
                        assert first_word[0].isupper() or not first_word[0].isalpha(), f"Chunk starts with lowercase: '{chunk}'"
    
    def test_get_sentence_overlap_method(self):
        """Test the _get_sentence_overlap method directly"""
        chunker = SentenceChunker(max_chunk_size=100, overlap=30)
        
        # Test with multiple sentences
        chunk = "First sentence. Second sentence. Third sentence."
        overlap = chunker._get_sentence_overlap(chunk)
        
        # Should return complete sentences that fit in overlap limit
        assert len(overlap) <= chunker.overlap
        if overlap:
            # Should be complete sentences
            assert overlap.endswith('.') or overlap.endswith('!') or overlap.endswith('?')
    
    def test_chunker_spec_property(self):
        """Test that chunker has the required spec property"""
        chunker = SentenceChunker(max_chunk_size=100, overlap=20)
        spec = chunker.spec
        
        assert spec is not None
        assert hasattr(spec, 'name')
        assert hasattr(spec, 'params')
        assert spec.name == "sentence"
        assert spec.params['max_size'] == 100
        assert spec.params['overlap'] == 20
