#!/usr/bin/env python3
"""
Finding-Document Matcher App

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

from efi_core.retrieval import RetrieverIndex, SearchResult
from efi_core.retrieval.index_builder import IndexBuilder
from efi_analyser.chunkers import SentenceChunker
from efi_analyser.embedders import SentenceTransformerEmbedder
from efi_corpus.embedded.embedded_corpus import EmbeddedCorpus
from efi_library.embedded.embedded_library import EmbeddedLibrary



def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Find best matches between findings and document chunks"
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("corpora/air_quality/india"),
        help="Path to corpus directory (default: corpora/air_quality/india)"
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
    chunker = SentenceChunker()
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
            results = retriever.query(finding_embedding, top_k=args.top_k)
            
            if results:
                # Print finding header
                finding_preview = finding.text[:60] + "..." if len(finding.text) > 60 else finding.text
                print(f"\n🔍 Finding {i}: {finding_id}")
                print(f"   Text: {finding_preview}")
                print("-" * 120)
                
                for j, result in enumerate(results, 1):
                    # Get chunk text from corpus using stored chunks instead of re-chunking
                    doc_id, chunk_idx = result.item_id.split('_chunk_')
                    chunk_idx_int = int(chunk_idx)
                    
                    # Get stored chunks directly from the chunk store
                    chunks = embedded_corpus.get_chunks(doc_id, materialize_if_necessary=False)
                    if chunks and chunk_idx_int < len(chunks):
                        chunk_text = chunks[chunk_idx_int]
                    else:
                        chunk_text = "Chunk not found"
                    
                    # Format chunk text for table
                    chunk_preview = chunk_text[:75] + "..." if len(chunk_text) > 75 else chunk_text
                    print(f"{result.item_id:<20} {result.score:<8.4f} {chunk_preview}")
                
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
