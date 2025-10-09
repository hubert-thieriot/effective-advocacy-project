#!/usr/bin/env python3
"""
Build embedded corpora script.

This script pre-builds all embeddings for documents in corpora, providing
progress bars with ETA for long-running operations.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List
import time
from datetime import datetime, timedelta

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from efi_analyser.chunkers import TextChunker
from efi_analyser.embedders import SentenceTransformerEmbedder
from efi_corpus.embedded.embedded_corpus import EmbeddedCorpus


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Build embedded corpora with progress tracking"
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        # default=Path("corpora/mediacloud_india_coal"),
        help="Path to corpus directory (default: corpora/mediacloud_india_coal)"
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("workspace"),
        help="Path to workspace directory (default: workspace)"
    )
    parser.add_argument(
        "--list-corpora",
        action="store_true",
        help="List available corpora and exit"
    )
    parser.add_argument(
        "--build-all",
        action="store_true",
        default=True,
        help="Build all available corpora instead of just one"
    )
    
    parser.add_argument(
        "--max-documents",
        type=int,
        help="Maximum number of documents to process (useful for testing with large corpora)"
    )
    
    args = parser.parse_args()
    
    # List available corpora if requested
    if args.list_corpora:
        corpora_dir = Path("corpora")
        if corpora_dir.exists():
            print("Available corpora:")
            for top_level_dir in sorted(corpora_dir.iterdir()):
                if top_level_dir.is_dir():
                    # Check if it's a direct corpus or contains sub-corpora
                    index_file = top_level_dir / "index.jsonl"
                    documents_dir = top_level_dir / "documents"
                    
                    if index_file.exists() or documents_dir.exists():
                        # Direct corpus
                        print(f"  {top_level_dir.name}/")
                    else:
                        # Check for sub-corpora
                        for sub_dir in sorted(top_level_dir.iterdir()):
                            if sub_dir.is_dir():
                                sub_index = sub_dir / "index.jsonl"
                                sub_docs = sub_dir / "documents"
                                if sub_index.exists() or sub_docs.exists():
                                    print(f"  {top_level_dir.name}/{sub_dir.name}/")
        else:
            print("No corpora directory found")
        return 0
    
    # Validate paths
    if args.corpus is not None and not args.corpus.exists():
        print(f"Error: Corpus path does not exist: {args.corpus}")
        print("Use --list-corpora to see available corpora")
        return 1
    
    if not args.workspace.exists():
        print(f"Creating workspace directory: {args.workspace}")
        args.workspace.mkdir(parents=True, exist_ok=True)
    
    if args.build_all:
        # Build all available corpora
        print("Building all available corpora...")
        corpora_dir = Path("corpora")
        total_corpora = 0
        successful_corpora = 0
        
        for top_level_dir in sorted(corpora_dir.iterdir()):
            if top_level_dir.is_dir():
                # Check if it's a direct corpus or contains sub-corpora
                index_file = top_level_dir / "index.jsonl"
                documents_dir = top_level_dir / "documents"
                
                if index_file.exists() or documents_dir.exists():
                    # Direct corpus
                    total_corpora += 1
                    if build_single_corpus(top_level_dir, args.workspace, args.max_documents):
                        successful_corpora += 1
                else:
                    # Check for sub-corpora
                    for sub_dir in sorted(top_level_dir.iterdir()):
                        if sub_dir.is_dir():
                            sub_index = sub_dir / "index.jsonl"
                            sub_docs = sub_dir / "documents"
                            if sub_index.exists() or sub_docs.exists():
                                total_corpora += 1
                                if build_single_corpus(sub_dir, args.workspace, args.max_documents):
                                    successful_corpora += 1
        
        print(f"\nBuild complete: {successful_corpora}/{total_corpora} corpora built successfully")
        return 0 if successful_corpora == total_corpora else 1
    else:
        # Build single corpus
        if not corpus_has_documents(args.corpus):
            print(f"Error: Corpus {args.corpus.name} does not contain document data")
            print("Expected either index.jsonl or documents/ directory")
            return 1
        
        return build_single_corpus(args.corpus, args.workspace, args.max_documents)


def corpus_has_documents(corpus_path: Path) -> bool:
    """Check if a corpus path contains document data"""
    index_file = corpus_path / "index.jsonl"
    documents_dir = corpus_path / "documents"
    return index_file.exists() or documents_dir.exists()


def build_single_corpus(corpus_path: Path, workspace_path: Path, max_documents: Optional[int] = None) -> bool:
    """Build a single corpus and return success status"""
    try:
        print(f"\nBuilding corpus: {corpus_path.name}")
        
        embedded_corpus = EmbeddedCorpus(
            corpus_path=corpus_path,
            workspace_path=workspace_path,
            chunker=TextChunker(),
            embedder=SentenceTransformerEmbedder()
        )
        
        # Get corpus info
        info = embedded_corpus.get_corpus_info()
        print(f"  Contains {info['document_count']} documents")
        
        if max_documents:
            print(f"  Processing only the first {max_documents} documents")
            embedded_corpus.build_all(max_documents=max_documents)
        else:
            embedded_corpus.build_all()
            # embedded_corpus.build_embeddings()
            # embedded_corpus.build_index()
            
        print(f"  ✓ Successfully built corpus: {corpus_path.name}")
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to build corpus {corpus_path.name}: {e}")
        return False


if __name__ == "__main__":
    main()
