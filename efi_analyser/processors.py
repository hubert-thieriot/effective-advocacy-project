"""
Document processors for the analysis system
"""

import json
import re
from typing import Dict, Any, List, Optional
from pathlib import Path
from .types import Processor
from efi_core.types import Document


class CosineSimilarityProcessor(Processor):
    """Process documents using cosine similarity with reference vectors"""
    
    def __init__(self, reference_vectors: List[Dict[str, Any]], threshold: float = 0.8, name: str = None):
        """
        Initialize cosine similarity processor
        
        Args:
            reference_vectors: List of reference vectors with 'text' and 'vector' keys
            threshold: Similarity threshold (0.0 to 1.0)
            name: Optional name for this processor
        """
        self.reference_vectors = reference_vectors
        self.threshold = threshold
        self.name = name or "cosine_similarity"
        
        # Import here to avoid dependency issues
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self._encode_references()
        except ImportError:
            print("Warning: sentence-transformers not available. Cosine similarity will not work.")
            self.model = None
    
    def _encode_references(self):
        """Encode reference texts to vectors"""
        if self.model:
            for ref in self.reference_vectors:
                if 'text' in ref and 'vector' not in ref:
                    ref['vector'] = self.model.encode(ref['text'])
    
    def process(self, document: Document) -> Dict[str, Any]:
        """Process document and return cosine similarity results"""
        if not self.model:
            return {"error": "sentence-transformers not available"}
        
        # Encode document text
        doc_vector = self.model.encode(document.text)
        
        # Calculate similarities
        similarities = []
        above_threshold = []
        
        for i, ref in enumerate(self.reference_vectors):
            if 'vector' in ref:
                from sklearn.metrics.pairwise import cosine_similarity
                similarity = cosine_similarity([doc_vector], [ref['vector']])[0][0]
                
                similarities.append({
                    "reference_index": i,
                    "reference_text": ref.get('text', f'ref_{i}'),
                    "similarity": float(similarity)
                })
                
                if similarity >= self.threshold:
                    above_threshold.append({
                        "reference_index": i,
                        "reference_text": ref.get('text', f'ref_{i}'),
                        "similarity": float(similarity)
                    })
        
        return {
            "similarities": similarities,
            "above_threshold": above_threshold,
            "max_similarity": max([s["similarity"] for s in similarities]) if similarities else 0.0,
            "threshold": self.threshold
        }


class TextStatisticsProcessor(Processor):
    """Process documents to extract basic text statistics"""
    
    def __init__(self, name: str = None):
        """Initialize text statistics processor"""
        self.name = name or "text_statistics"
    
    def process(self, document: Document) -> Dict[str, Any]:
        """Extract basic text statistics from document"""
        text = document.text
        
        # Basic statistics
        word_count = len(text.split())
        char_count = len(text)
        sentence_count = len([s for s in text.split('.') if s.strip()])
        
        # Average word length
        avg_word_length = char_count / word_count if word_count > 0 else 0
        
        return {
            "word_count": word_count,
            "char_count": char_count,
            "sentence_count": sentence_count,
            "avg_word_length": round(avg_word_length, 2)
        }


class KeywordExtractorProcessor(Processor):
    """Extract keywords from documents using regex for whole word matching.

    Also supports raw regex patterns via the optional ``patterns`` argument. These
    are compiled as-is (with case sensitivity per ``case_sensitive``) and are not
    escaped nor subject to whole-word or hyphenation handling. This enables
    combined/boolean-like matching such as:

    - (konstruksi|pembangunan).{0,80}(debu|polusi|partikel)
    - resuspensi\s+debu
    """
    
    def __init__(self, keywords: List[str], case_sensitive: bool = False, whole_word_only: bool = True, 
                 allow_hyphenation: bool = True, name: str = None, patterns: Optional[List[str]] = None):
        """
        Initialize keyword extractor
        
        Args:
            keywords: List of keywords to search for
            case_sensitive: Whether search should be case sensitive
            whole_word_only: Whether to match only whole words (using word boundaries)
            allow_hyphenation: Whether to allow keywords split across hyphens and line breaks
            name: Optional name for this processor
        """
        self.keywords = keywords
        self.case_sensitive = case_sensitive
        self.whole_word_only = whole_word_only
        self.allow_hyphenation = allow_hyphenation
        self.name = name or "keyword_extractor"
        self._custom_patterns = list(patterns or [])
        
        # Compile regex patterns for whole word matching
        self.regex_patterns = {}
        for keyword in keywords:
            # Escape special regex characters
            escaped_keyword = re.escape(keyword)
            
            # Create pattern based on options
            if self.whole_word_only:
                if self.allow_hyphenation:
                    # Allow keywords split across hyphens and line breaks
                    # For "transport", create pattern like: \b(tran\s*[-\\n]\s*sport)\b
                    # This matches "tran-sport", "tran\nsport", "tran- sport", etc.
                    hyphenated_pattern = self._create_hyphenated_pattern(keyword)
                    pattern = r'\b(' + hyphenated_pattern + r')\b'
                else:
                    # Use word boundaries for whole word matching
                    pattern = r'\b' + escaped_keyword + r'\b'
            else:
                if self.allow_hyphenation:
                    # Allow keywords split across hyphens and line breaks without word boundaries
                    hyphenated_pattern = self._create_hyphenated_pattern(keyword)
                    pattern = r'(' + hyphenated_pattern + r')'
                else:
                    # Simple substring matching
                    pattern = escaped_keyword
            
            # Compile with appropriate flags
            flags = 0
            if not self.case_sensitive:
                flags |= re.IGNORECASE
            
            self.regex_patterns[keyword] = re.compile(pattern, flags)

        # Compile raw regex patterns (as-is)
        if self._custom_patterns:
            flags = 0 if self.case_sensitive else re.IGNORECASE
            for pat in self._custom_patterns:
                try:
                    self.regex_patterns[pat] = re.compile(pat, flags)
                except re.error:
                    # Skip invalid patterns silently; alternatively could record in result metadata
                    continue
    
    def _create_hyphenated_pattern(self, keyword: str) -> str:
        """Create a regex pattern that allows keywords to be split across hyphens and line breaks"""
        if len(keyword) < 3:
            # For very short keywords, just return the original
            return re.escape(keyword)
        
        # Try to find good split points (avoid splitting single letters)
        # For "transport", try splits like: "tran-sport", "trans-port", "transp-ort"
        patterns = []
        
        # Add the original keyword
        patterns.append(re.escape(keyword))
        
        # Try different split points
        for i in range(2, len(keyword) - 1):  # Avoid splitting at first or last character
            part1 = keyword[:i]
            part2 = keyword[i:]
            
            # Skip if either part is too short (less than 2 characters)
            if len(part1) < 2 or len(part2) < 2:
                continue
            
            # Create pattern: part1\s*[-\\n]\s*part2
            # This matches: part1-part2, part1\npart2, part1- part2, part1\n part2
            pattern = re.escape(part1) + '\\s*[-\\n]\\s*' + re.escape(part2)
            patterns.append(pattern)
        
        # Join all patterns with | (OR)
        return '|'.join(patterns)
    
    def process(self, document: Document) -> Dict[str, Any]:
        """Extract keyword occurrences from document using regex whole word matching"""
        text = document.text
        
        keyword_counts = {}
        keyword_positions = {}
        
        for keyword in self.keywords:
            pattern = self.regex_patterns[keyword]
            matches = pattern.findall(text)
            
            # Handle the case where we have multiple capture groups
            if isinstance(matches, list) and matches and isinstance(matches[0], tuple):
                # Flatten the list of tuples
                flat_matches = []
                for match_tuple in matches:
                    flat_matches.extend([m for m in match_tuple if m])
                matches = flat_matches
            
            count = len(matches)
            keyword_counts[keyword] = count
            
            if count > 0:
                # Find first occurrence position
                match = pattern.search(text)
                keyword_positions[keyword] = match.start() if match else 0
        
        return {
            "keyword_counts": keyword_counts,
            "keyword_positions": keyword_positions,
            "total_keywords": sum(keyword_counts.values()),
            "whole_word_only": self.whole_word_only,
            "case_sensitive": self.case_sensitive,
            "allow_hyphenation": self.allow_hyphenation
        }
