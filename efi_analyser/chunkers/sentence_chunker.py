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
    
    def __init__(self, max_chunk_size: int = 200, overlap: int = 50):
        """
        Initialize the sentence chunker.
        
        Args:
            max_chunk_size: Maximum size of each chunk in characters
            overlap: Number of characters to overlap between chunks
        """
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        
        # Sentence ending patterns
        self.sentence_end_pattern = re.compile(r'[.!?]\s+')
    
    @property
    def spec(self) -> ChunkerSpec:
        """Get the chunker specification."""
        return ChunkerSpec(
            name="sentence",
            params={
                "max_size": self.max_chunk_size,
                "overlap": self.overlap
            }
        )
    
    def chunk(self, text: str) -> List[str]:
        """
        Split text into sentence-based chunks.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        # Split text into sentences
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return []
        
        # If text is short enough, return as single chunk
        if len(text) <= self.max_chunk_size:
            return [text.strip()]
        
        # Create chunks with overlap
        chunks = []
        current_chunk = ""
        overlap_text = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence would exceed max size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= self.max_chunk_size:
                # Add sentence to current chunk
                current_chunk = potential_chunk
            else:
                # Current chunk is full, save it and start new one
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap from previous
                if overlap_text:
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex pattern.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Use a better approach: find all sentence endings and split properly
        sentences = []
        current_pos = 0
        
        # Find all sentence endings
        for match in self.sentence_end_pattern.finditer(text):
            end_pos = match.end()
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
