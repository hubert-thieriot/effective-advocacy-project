"""
Document processors for the analysis system
"""

import json
import re
from typing import Dict, Any, List, Optional
from pathlib import Path
from .types import Processor
from efi_corpus.types import Document


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
    """Extract keywords from documents using regex for whole word matching"""
    
    def __init__(self, keywords: List[str], case_sensitive: bool = False, name: str = None):
        """
        Initialize keyword extractor
        
        Args:
            keywords: List of keywords to search for
            case_sensitive: Whether search should be case sensitive
            name: Optional name for this processor
        """
        self.keywords = keywords
        self.case_sensitive = case_sensitive
        self.name = name or "keyword_extractor"
        
        # Compile regex patterns for whole word matching
        self.regex_patterns = {}
        for keyword in keywords:
            # Escape special regex characters and create word boundary pattern
            escaped_keyword = re.escape(keyword)
            if self.case_sensitive:
                pattern = r'\b' + escaped_keyword + r'\b'
            else:
                pattern = r'\b' + escaped_keyword + r'\b'
            self.regex_patterns[keyword] = re.compile(pattern, re.IGNORECASE if not self.case_sensitive else 0)
    
    def process(self, document: Document) -> Dict[str, Any]:
        """Extract keyword occurrences from document using regex whole word matching"""
        text = document.text
        
        keyword_counts = {}
        keyword_positions = {}
        
        for keyword in self.keywords:
            pattern = self.regex_patterns[keyword]
            matches = pattern.findall(text)
            count = len(matches)
            keyword_counts[keyword] = count
            
            if count > 0:
                # Find first occurrence position
                match = pattern.search(text)
                keyword_positions[keyword] = match.start() if match else 0
        
        return {
            "keyword_counts": keyword_counts,
            "keyword_positions": keyword_positions,
            "total_keywords": sum(keyword_counts.values())
        }
