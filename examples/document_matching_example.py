#!/usr/bin/env python3
"""
Document Matching Example

Clean example of using the DocumentMatchingApp to find matches between
findings and documents in a corpus.
"""

from pathlib import Path
from efi_analyser.pipeline.finding_document_matching import DocumentMatchingPipeline
from efi_analyser.report_generator.document_matching import DocumentMatchingReportGenerator
from efi_analyser.types import ReportConfig
from efi_analyser.chunkers import SentenceChunker
from efi_analyser.embedders import SentenceTransformerEmbedder
from efi_analyser.rescorers import NLIReScorer
from efi_corpus.embedded.embedded_corpus import EmbeddedCorpus
from efi_library.embedded.embedded_library import EmbeddedLibrary
from efi_core.types import FindingFilters


def main():
    """Run document matching analysis."""
    
    # Paths
    corpus_path = Path("test_corpus")
    library_path = Path("test_library")
    workspace_path = Path("workspace")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Initialize components
    chunker = SentenceChunker()
    embedder = SentenceTransformerEmbedder(lazy_load=True)
    
    # Load embedded data
    embedded_corpus = EmbeddedCorpus(corpus_path, workspace_path, chunker, embedder)
    embedded_library = EmbeddedLibrary(library_path, workspace_path, chunker, embedder)
    
    # Initialize rescorers
    nli_rescorer = NLIReScorer()
    
    # Create pipeline
    pipeline = DocumentMatchingPipeline(
        embedded_corpus=embedded_corpus,
        embedded_library=embedded_library,
        rescorers_stage1=[nli_rescorer],
        rescorers_stage2=[],
        workspace_path=workspace_path
    )
    
    # Define filters
    filters = FindingFilters(
        include_keywords=["india"],
        exclude_keywords=["russian", "russia"]
    )
    
    # Run matching
    results = pipeline.run_matching(
        finding_filters=filters,
        n_findings=10,
        top_n_retrieval=50,
        top_n_rescoring_stage1=10
    )
    
    # Generate reports
    report_config = ReportConfig(
        output_dir=results_dir,
        formats=["csv", "json", "html"],
        include_metadata=True,
        include_timing=True
    )
    report_generator = DocumentMatchingReportGenerator(report_config)
    report_files = report_generator.generate_all(results)
    
    # Output results
    print(f"Processed {results.findings_processed} findings")
    print(f"Generated {results.total_matches} total matches")
    print(f"Reports saved to: {results_dir}")


if __name__ == "__main__":
    main()
