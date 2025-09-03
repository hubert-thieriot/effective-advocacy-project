#!/usr/bin/env python3
"""
Document Matching Example

Clean example of using the DocumentMatchingApp to find matches between
findings and documents in a corpus.
"""

from pathlib import Path
from efi_analyser.pipeline.finding_document_matching import FindingDocumentMatchingPipeline
from efi_analyser.report_generator import FindingDocumentMatchingReportGenerator
from efi_analyser.types import ReportConfig
from efi_analyser.chunkers import SentenceChunker
from efi_analyser.embedders import SentenceTransformerEmbedder
from efi_analyser.scorers import NLIHFScorer, NLILLMScorer, LLMScorerConfig
from efi_corpus.embedded.embedded_corpus import EmbeddedCorpus
from efi_library.embedded.embedded_library import EmbeddedLibrary
from efi_core.types import FindingFilters, Task


def main():
    """Run document matching analysis."""
    
    # Paths - using India corpus as requested
    corpus_path = Path("corpora/mediacloud_india_airquality")
    library_path = Path("libraries/crea_publications")
    workspace_path = Path("workspace")
    results_dir = Path("results/finding_document_matching")
    results_dir.mkdir(exist_ok=True)

    # Initialize components
    chunker = SentenceChunker()
    embedder = SentenceTransformerEmbedder(lazy_load=True)

    # Load embedded data
    embedded_corpus = EmbeddedCorpus(corpus_path, workspace_path, chunker, embedder)
    embedded_library = EmbeddedLibrary(library_path, workspace_path, chunker, embedder)

    # Initialize scorers: NLI in stage 1, two LLMs in stage 2
    nli_scorer = NLIHFScorer(name="nli_roberta")

    # Two NLI LLM scorers as requested - using NLI LLM scorer with different models
    llm_scorer_phi3 = NLILLMScorer(name="phi3_3.8b", config=LLMScorerConfig(model="phi3:3.8b"))
    llm_scorer_gemma = NLILLMScorer(name="gemma3_4b", config=LLMScorerConfig(model="gemma3:4b"))

    # Create pipeline
    pipeline = FindingDocumentMatchingPipeline(
        embedded_corpus=embedded_corpus,
        embedded_library=embedded_library,
        scorers_stage1=[nli_scorer],
        scorers_stage2=[llm_scorer_phi3, llm_scorer_gemma],
        workspace_path=workspace_path
    )
    
    # Define filters focused on India
    filters = FindingFilters(
        include_keywords=["india", "indian", "delhi", "mumbai", "bengaluru"],
        exclude_keywords=["russian", "russia", "china", "chinese", "pakistan"]
    )

    # Run matching with more findings
    results = pipeline.run_matching(
        finding_filters=filters,
        n_findings=20,
        top_n_retrieval=100,
        top_n_rescoring_stage1=15
    )
    
    # Generate reports
    report_config = ReportConfig(
        output_dir=results_dir,
        formats=["html"],
        include_metadata=True,
        include_timing=True
    )
    report_generator = FindingDocumentMatchingReportGenerator(report_config)
    report_files = report_generator.generate_all(results)
    
    # Output results
    print(f"Processed {results.findings_processed} India-focused findings")
    print(f"Generated {results.total_matches} total matches")
    print(f"Used NLI scorer in stage 1 and two LLMs (phi3:3.8b, gemma3:4b) in stage 2")
    print(f"Reports saved to: {results_dir}")


if __name__ == "__main__":
    main()
