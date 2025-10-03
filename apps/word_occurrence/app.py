"""
EFI Analyser - Word Occurrence Analysis Application

This app provides a high-level interface for analyzing keyword occurrences in a corpus
using the new clean architecture with dedicated pipeline and report generator.
"""

from typing import Dict, Optional
from pathlib import Path

from efi_corpus import CorpusHandle
from efi_analyser.pipeline.word_occurrence import WordOccurrencePipeline
from efi_analyser.report_generator.word_occurrence import WordOccurrenceReportGenerator
from efi_analyser.types import WordOccurrenceConfig, WordOccurrenceResults, ReportConfig


class WordOccurrenceApp:
    """
    Application for analyzing keyword occurrences in a corpus.
    
    Uses the new clean architecture with dedicated pipeline and report generator
    to provide comprehensive keyword analysis and reporting.
    """
    
    def __init__(self, config: WordOccurrenceConfig):
        """
        Initialize the app with configuration.
        
        Args:
            config: Configuration for the word occurrence analysis
        """
        self.config = config
        
        # Initialize pipeline
        self.pipeline = WordOccurrencePipeline(
            keywords=config.keywords,
            case_sensitive=config.case_sensitive,
            whole_word_only=config.whole_word_only,
            allow_hyphenation=config.allow_hyphenation
        )
        
        # Initialize report generator
        report_config = ReportConfig(
            output_dir=config.workspace_path / "word_occurrence_reports",
            formats=config.output_formats,
            include_metadata=True,
            include_timing=True
        )
        self.report_generator = WordOccurrenceReportGenerator(report_config)
    
    def analyze_keywords(self, corpus_path: Optional[Path] = None) -> WordOccurrenceResults:
        """
        Analyze keyword occurrences in the specified corpus.
        
        Args:
            corpus_path: Optional path to override the configured corpus path
            
        Returns:
            WordOccurrenceResults containing the analysis results
        """
        # Use provided corpus path or fall back to configured one
        target_corpus_path = corpus_path or self.config.corpus_path
        
        # Create corpus handle
        corpus_handle = CorpusHandle(target_corpus_path)
        
        # Run the pipeline
        pipeline_result = self.pipeline.run(corpus_handle)
        
        # Convert to WordOccurrenceResults format
        results = self.pipeline.get_word_occurrence_results(pipeline_result)
        
        return results
    
    def generate_reports(self, results: WordOccurrenceResults) -> Dict[str, Path]:
        """
        Generate reports for the analysis results.
        
        Args:
            results: The word occurrence analysis results
            
        Returns:
            Dictionary mapping report format to file path
        """
        return self.report_generator.generate_all(results)
    
    def run_analysis(self, corpus_path: Optional[Path] = None) -> Dict[str, Path]:
        """
        Run the complete analysis workflow and generate reports.
        
        Args:
            corpus_path: Optional path to override the configured corpus path
            
        Returns:
            Dictionary mapping report format to file path
        """
        # Run analysis
        results = self.analyze_keywords(corpus_path)
        
        # Generate reports
        report_paths = self.generate_reports(results)
        
        return report_paths
