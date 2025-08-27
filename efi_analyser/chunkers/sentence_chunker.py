"""
Sentence-based text chunker for EFI Analyser

This chunker splits text into chunks based on sentence boundaries while respecting
maximum chunk size and overlap constraints.
"""

import re
from typing import List
from efi_core.types import ChunkerSpec


class SentenceChunker:
    """
    Text chunker that splits text into sentence-based chunks.
    
    This chunker ensures that chunks are complete sentences and respects
    maximum chunk size and overlap constraints.
    """
    
    def __init__(self, max_chunk_size: int = 500, overlap: int = 50):
        """
        Initialize the sentence chunker.
        
        Args:
            max_chunk_size: Maximum size of each chunk in characters
            overlap: Number of characters to overlap between chunks
        """
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        
        # Sentence ending patterns - using lookahead to identify boundaries
        # Support straight and curly quotes
        quote_chars = '"”’\'”'
        # Pattern 1: Period followed by newline (most reliable sentence boundary)
        # Pattern 2: Period followed by space and capital letter (e.g., "dark.The")
        # Pattern 3: Period followed by capital letter without space (e.g., "death.The")
        # Pattern 4: Period followed by quote (straight or curly)
        # Pattern 5: Period followed by quote and newline
        # Pattern 6: Period followed by newline and capital
        # Pattern 7: Period followed by quote, double newline and capital
        # But avoid splitting on decimal numbers (e.g., PM2.5, 34.5%)
        self.sentence_end_patterns = [
            re.compile(r'[.!?](?=\s*\n)'),                              # Period + whitespace + newline
            re.compile(r'[.!?](?=\s+[A-Z])'),                           # Period + space + capital letter
            re.compile(r'[.!?](?=[A-Z])'),                               # Period + capital letter (no space)
            re.compile(r'[.!?](?=\s*[\"”])'),                           # Period + whitespace + straight/curly quote
            re.compile(r'[.!?](?=[\"”]\s*\n)'),                        # Period + straight/curly quote + whitespace + newline
            re.compile(r'[.!?](?=\s*\n\s*[A-Z])'),                    # Period + whitespace + newline + whitespace + capital
            re.compile(r'[.!?](?=[\"”]\s*\n\s*\n\s*[A-Z])'),        # Period + straight/curly quote + blank line + capital
        ]
    
    @property
    def spec(self) -> ChunkerSpec:
        """Get the chunker specification."""
        return ChunkerSpec(
            name="sentence",
            params={
                "max_size": self.max_chunk_size,
                "overlap": self.overlap,
                "sentence_end_patterns": [str(pattern) for pattern in self.sentence_end_patterns]
            }
        )
    
    def chunk(self, text: str) -> List[str]:
        """
        Split text into sentence-based chunks.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks, one per sentence
        """
        if not text or not text.strip():
            return []
        
        # Split text into sentences
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return []
        
        # Always create one chunk per sentence
        chunks = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # If a single sentence exceeds max_chunk_size, truncate it
                if len(sentence) > self.max_chunk_size:
                    sentence = sentence[:self.max_chunk_size].strip()
                chunks.append(sentence)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using multiple regex patterns.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Find all sentence endings using multiple patterns
        sentence_endings = []
        
        # Collect all matches from all patterns
        for pattern in self.sentence_end_patterns:
            for match in pattern.finditer(text):
                sentence_endings.append((match.end(), match.group()))
        
        # Sort by position to process in order
        sentence_endings.sort(key=lambda x: x[0])
        
        sentences = []
        current_pos = 0
        
        # Split at each sentence ending
        for end_pos, match_text in sentence_endings:
            sentence = text[current_pos:end_pos].strip()
            if sentence:
                sentences.append(sentence)
            current_pos = end_pos
        
        # Add the last sentence if there's remaining text
        if current_pos < len(text):
            remaining = text[current_pos:].strip()
            if remaining:
                sentences.append(remaining)
        
        return sentences
    
    def _get_sentence_overlap(self, chunk: str) -> str:
        """
        Get overlap text from the end of a chunk.
        
        Args:
            chunk: Chunk text to get overlap from
            
        Returns:
            Overlap text (complete sentences that fit within overlap limit)
        """
        if not chunk or len(chunk) <= self.overlap:
            return chunk
        
        # Find the last sentence that fits within overlap limit
        sentences = self._split_into_sentences(chunk)
        
        overlap_text = ""
        for sentence in reversed(sentences):
            if len(overlap_text + " " + sentence) <= self.overlap:
                overlap_text = sentence + " " + overlap_text if overlap_text else sentence
            else:
                break
        
        return overlap_text.strip()
