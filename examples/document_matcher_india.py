#!/usr/bin/env python3
"""
Simple Document Matcher for India using the DocumentMatchingPipeline

This script demonstrates the new pipeline class with minimal output.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from efi_analyser.chunkers import SentenceChunker
from efi_analyser.embedders import SentenceTransformerEmbedder
from efi_analyser.rescorers import NLIReScorer, OllamaReScorer, LLMReScorerConfig
from efi_core.types import FindingFilters
from efi_analyser.document_matching_pipeline import DocumentMatchingPipeline
from efi_analyser.report_generator import ReportGenerator
from efi_corpus.embedded.embedded_corpus import EmbeddedCorpus
from efi_library.embedded.embedded_library import EmbeddedLibrary


def main():
    """Run document matching using the pipeline."""
    
    # Paths
    corpus_path = project_root / "corpora" / "air_quality" / "india"
    library_path = project_root / "libraries" / "crea_publications"
    workspace = project_root / "workspace"
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Initialize components
    chunker = SentenceChunker()
    embedder = SentenceTransformerEmbedder(lazy_load=True)
    
    # Load embedded data
    embedded_corpus = EmbeddedCorpus(corpus_path, workspace, chunker, embedder)
    embedded_library = EmbeddedLibrary(library_path, workspace, chunker, embedder)
    
    # Make sure the corpus is embedded
    # embedded_corpus.build_all()
    
    
    # Initialize rescorers
    nli_rescorer = NLIReScorer()
    ollama_rescorer_gemma = OllamaReScorer(LLMReScorerConfig(model="gemma3:4b"))
    ollama_rescorer_phi = OllamaReScorer(LLMReScorerConfig(model="phi3:3.8b"))
    
    # Create pipeline with two-stage rescoring
    pipeline = DocumentMatchingPipeline(
        embedded_corpus=embedded_corpus,
        embedded_library=embedded_library,
        rescorers_stage1=[nli_rescorer],           # Fast rescorers for filtering
        rescorers_stage2=[ollama_rescorer_gemma, ollama_rescorer_phi],  # Slow rescorers for final scoring
        workspace_path=workspace
    )
    
    # Define filters (India findings, exclude Russia)
    filters = FindingFilters(
        include_keywords=["india"],
        exclude_keywords=["russian", "russia"]
    )
    
    # Run matching
    print("Running document matching...")
    results = pipeline.run_matching(
        finding_filters=filters,
        n_findings=100,
        top_n_retrieval=100,      # Retrieve 100 candidates initially
        top_n_rescoring_stage1=10 # Use stage1 rescorers to filter down to top 10
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"india_matching_results_{timestamp}.pkl"
    pipeline.save_results(results, output_path)
    
    print(f"Processed {results.findings_processed} findings")
    print(f"Generated {results.total_matches} total matches")
    print(f"Results saved to: {output_path}")
    
    # Generate reports
    print("Generating reports...")
    report_generator = ReportGenerator(results)
    report_files = report_generator.generate_all_reports(
        results_dir, 
        f"india_matching_report_{timestamp}"
    )
    
    print("Reports generated:")
    for format_name, file_path in report_files.items():
        print(f"  - {format_name.upper()}: {file_path}")


if __name__ == "__main__":
    main()
