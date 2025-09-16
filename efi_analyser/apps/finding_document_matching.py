#!/usr/bin/env python3
"""
Finding Document Matching App

This app finds the best matches between findings from a library and document chunks from a corpus.
It uses existing embeddings and can optionally build FAISS indexes for faster search.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from efi_core.retrieval import RetrieverIndex
from efi_core.retrieval.index_builder import IndexBuilder
from efi_analyser.chunkers import TextChunker
from efi_analyser.embedders import SentenceTransformerEmbedder
from efi_corpus.embedded.embedded_corpus import EmbeddedCorpus
from efi_library.embedded.embedded_library import EmbeddedLibrary
from efi_analyser.scorers import NLIHFScorer, NLIHFScorerConfig
from efi_core.types import Task, Candidate, RescoreEngine, RerankPolicy, RerankingEngine



def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Find best matches between findings and document chunks"
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("corpora/mediacloud_india_coal"),
        help="Path to corpus directory (default: corpora/mediacloud_india_coal)"
    )
    parser.add_argument(
        "--library", 
        type=Path,
        default=Path("libraries/crea_publications"),
        help="Path to library directory (default: libraries/crea_publications)"
    )
    parser.add_argument(
        "--workspace", 
        type=Path,
        default=Path("workspace"),
        help="Path to workspace root directory (default: workspace)"
    )
    parser.add_argument(
        "--num-findings",
        type=int,
        default=100,
        help="Number of findings to process (default: 100)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top matches to return per finding (default: 5)"
    )
    parser.add_argument(
        "--show-stats",
        action="store_true",
        help="Show detailed statistics and exit"
    )
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="Apply NLI-based re-scoring after retrieval"
    )
    parser.add_argument(
        "--rescorer-model",
        type=str,
        default="facebook/bart-large-mnli",
        help="HuggingFace model to use for re-scoring (default: facebook/bart-large-mnli)"
    )

    args = parser.parse_args()
    
    # Validate paths
    if not args.corpus.exists():
        print(f"Error: Corpus does not exist: {args.corpus}")
        return 1
    
    if not args.library.exists():
        print(f"Error: Library does not exist: {args.library}")
        return 1
    
    if not args.workspace.exists():
        print(f"Creating workspace directory: {args.workspace}")
        args.workspace.mkdir(parents=True, exist_ok=True)
    
    # Initialize components (use lazy loading for faster startup)
    chunker = TextChunker()
    embedder = SentenceTransformerEmbedder(lazy_load=True)
    
    print("🔍 Finding-Document Matcher")
    print("=" * 50)
    
    # Load embedded corpus
    print(f"\n📚 Loading embedded corpus from: {args.corpus}")
    embedded_corpus = EmbeddedCorpus(
        corpus_path=args.corpus,
        workspace_path=args.workspace,
        chunker=chunker,
        embedder=embedder
    )
    
    # Check available embeddings
    print(embedded_corpus.show_stats())

    
    # Load embedded library
    embedded_library = EmbeddedLibrary(
        library_path=args.library,
        workspace_path=args.workspace,
        chunker=chunker,
        embedder=embedder
    )
    # Check available embeddings
    print(embedded_library.show_stats())
    
    # Build corpus index if requested
    print(f"\n🔨 Building FAISS index for corpus...")
    success = embedded_corpus.build_index()

    # Get findings to process
    print(f"\n🎯 Processing {args.num_findings} findings...")
    findings_to_process = []
    for doc_findings in embedded_library.library.iter_findings():
        for finding in doc_findings.findings:
            if embedded_library.embedding_store.has_embeddings(finding.finding_id, embedded_library.chunker_spec, embedded_library.embedder.spec):
                findings_to_process.append(finding.finding_id)
                if len(findings_to_process) >= args.num_findings:
                    break
        if len(findings_to_process) >= args.num_findings:
            break
    
    if not findings_to_process:
        print("No findings with embeddings available")
        return 1
    
    print(f"  ✓ Found {len(findings_to_process)} findings with embeddings")

    # Optional re-scoring step
    rescore_engine = None
    if args.rescore:
        nli_scorer = NLIHFScorer(
            name="nli_roberta",
            task=Task.NLI,
            config=NLIHFScorerConfig(model_name=args.rescorer_model)
        )
        rescore_engine = RescoreEngine(scorers=[nli_scorer])
        rerank_policy = RerankPolicy(positive_labels={"entails": 1.0})
        reranking_engine = RerankingEngine(policies={"nli_roberta": rerank_policy})

    # Create retriever for finding-to-document search
    print(f"\n🔍 Creating retriever for finding → document search...")
    retriever = RetrieverIndex(
        embedded_data_source=embedded_corpus,
        workspace_path=args.workspace,
        chunker_spec=chunker.spec,
        embedder_spec=embedder.spec,
        auto_rebuild=True
    )
    
    # Process each finding
    print(f"\n📊 Finding best matches (top-{args.top_k})...")
    print("=" * 120)
    print(f"{'Finding ID':<20} {'Score':<8} {'Document Chunk':<80}")
    print("=" * 120)
    
    total_matches = 0
    for i, finding_id in enumerate(findings_to_process, 1):
        try:
            # Get finding text from library
            finding = embedded_library.library.get_finding(finding_id)
            if not finding:
                continue
                
            # Get finding embedding
            finding_embedding = embedded_library.get_embeddings(finding_id, materialize_if_necessary=False)
            if finding_embedding is None:
                continue
            
            # Search for similar document chunks
            candidates = retriever.query(finding.text, top_k=args.top_k)

            if candidates:
                # Enrich candidates with text
                enriched_candidates = []
                for candidate in candidates:
                    try:
                        chunk = embedded_corpus.get_chunk(
                            chunk_id=candidate.item_id,
                            materialize_if_necessary=False
                        )
                        candidate.text = chunk.text if chunk else "Chunk not found"
                        enriched_candidates.append(candidate)
                    except Exception:
                        candidate.text = "Chunk not found"
                        enriched_candidates.append(candidate)

                # Apply optional re-scoring
                # For finding document matching: finding (premise) entails document (hypothesis)
                if rescore_engine:
                    enriched_candidates = rescore_engine.rescore(enriched_candidates, finding.text, premise_is_target=True)
                    enriched_candidates = reranking_engine.rerank(enriched_candidates)

                results = enriched_candidates

                # Print finding header
                finding_preview = finding.text[:60] + "..." if len(finding.text) > 60 else finding.text
                print(f"\n🔍 Finding {i}: {finding_id}")
                print(f"   Text: {finding_preview}")
                print("-" * 120)

                for j, result in enumerate(results, 1):
                    chunk_preview = result.text[:75] + "..." if len(result.text) > 75 else result.text
                    print(f"{result.item_id:<20} {result.ann_score:<8.4f} {chunk_preview}")

                total_matches += len(results)
            else:
                print(f"\n🔍 Finding {i}: {finding_id} - No matches found")
                
        except Exception as e:
            print(f"\n⚠️ Error processing finding {finding_id}: {e}")
            continue
    
    print(f"\n" + "=" * 80)
    print(f"🎯 Summary:")
    print(f"   Findings processed: {len(findings_to_process)}")
    print(f"   Total matches found: {total_matches}")
    print(f"   Average matches per finding: {total_matches/len(findings_to_process):.1f}")
    
    return 0



















if __name__ == "__main__":
    exit(main())
