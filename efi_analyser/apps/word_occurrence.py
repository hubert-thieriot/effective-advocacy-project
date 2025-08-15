"""
EFI Analyser - Keyword occurrence analysis application

This app uses the existing KeywordExtractorProcessor and KeywordPresenceAggregator
to provide a simple interface for keyword analysis.
"""

from typing import Dict, List
from pathlib import Path

from efi_corpus import CorpusHandle
from ..pipeline import LinearPipeline
from ..processors import KeywordExtractorProcessor
from ..aggregators import KeywordPresenceAggregator
from ..types import AppResult


class WordOccurrenceApp:
    """
    Application for counting keyword occurrences in a corpus
    
    Uses the existing KeywordExtractorProcessor and KeywordPresenceAggregator
    to provide keyword presence analysis (how many documents mention each keyword)
    """
    
    def __init__(self, corpus_path: Path, keywords: List[str]):
        """
        Initialize the app with a corpus path and keywords
        
        Args:
            corpus_path: Path to the corpus directory
            keywords: List of keywords to search for
        """
        self.corpus_path = Path(corpus_path)
        self.keywords = keywords
    
    def run(self) -> AppResult:
        """
        Run the keyword occurrence analysis
        
        Returns:
            AppResult containing keyword presence data and metadata
        """
        # Create corpus handle
        corpus_handle = CorpusHandle(self.corpus_path)
        
        # Create the processor and aggregator
        processor = KeywordExtractorProcessor(
            keywords=self.keywords,
            case_sensitive=False,
            whole_word_only=True,
            allow_hyphenation=True
        )
        
        aggregator = KeywordPresenceAggregator()
        
        # Create the pipeline with the correct processor name
        pipeline = LinearPipeline(
            filter_func=None,  # No filtering - process all documents
            processor_func=processor.process,
            aggregator_func=aggregator.aggregate,
            processor_name="keyword_extractor"  # This matches what the aggregator expects
        )
        
        # Run the pipeline
        pipeline_result = pipeline.run(corpus_handle)
        
        # Return app result
        return AppResult(
            data=pipeline_result.data,
            metadata={
                "app_name": "WordOccurrenceApp",
                "corpus_path": str(self.corpus_path),
                "keywords": self.keywords,
                "pipeline_metadata": pipeline_result.metadata
            }
        )
