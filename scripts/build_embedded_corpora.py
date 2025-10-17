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
from multiprocessing import Pool, cpu_count
from functools import partial

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from efi_analyser.chunkers import TextChunker
from efi_analyser.chunkers.text_chunker import TextChunkerConfig
from efi_analyser.embedders import SentenceTransformerEmbedder
from efi_corpus.embedded.embedded_corpus import EmbeddedCorpus
from tqdm import tqdm


def process_document_worker(args_tuple):
    """
    Worker function to process a single document (chunking and embedding).
    This function is designed to be called by multiprocessing.Pool.
    
    Args:
        args_tuple: Tuple of (doc_id, corpus_path, workspace_path, max_words)
    
    Returns:
        Tuple of (doc_id, success, error_message)
    """
    doc_id, corpus_path, workspace_path, max_words = args_tuple
    
    try:
        # Create embedder and chunker for this worker
        chunker = TextChunker(TextChunkerConfig(max_words=max_words))
        embedder = SentenceTransformerEmbedder(lazy_load=True)
        
        # Create embedded corpus instance
        embedded_corpus = EmbeddedCorpus(
            corpus_path=corpus_path,
            workspace_path=workspace_path,
            chunker=chunker,
            embedder=embedder,
        )
        
        # Process the document: chunk it
        _ = embedded_corpus.get_chunks(doc_id, materialize_if_necessary=True)
        
        # Process the document: embed it
        _ = embedded_corpus.get_embeddings(doc_id, materialize_if_necessary=True)
        
        return (doc_id, True, None)
    except Exception as e:
        return (doc_id, False, str(e))


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
        default=False,
        help="Build all available corpora instead of just one"
    )
    
    parser.add_argument(
        "--max-documents",
        type=int,
        help="Maximum number of documents to process (useful for testing with large corpora)"
    )
    
    parser.add_argument(
        "--max-words",
        type=int,
        default=200,
        help="Maximum words per chunk (default: 200, matching narrative framing default)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers for processing documents (default: auto-detect based on CPU count)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation (default: 32)"
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
    
    # Decide whether to build a specific corpus or all corpora
    # If --corpus is specified, always build just that one (ignore --build-all)
    if args.corpus is not None:
        # Build single corpus
        if not corpus_has_documents(args.corpus):
            print(f"Error: Corpus {args.corpus.name} does not contain document data")
            print("Expected either index.jsonl or documents/ directory")
            return 1
        
        success = build_single_corpus(args.corpus, args.workspace, args.max_documents, args.max_words, args.workers, args.batch_size)
        return 0 if success else 1
    elif args.build_all:
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
                    if build_single_corpus(top_level_dir, args.workspace, args.max_documents, args.max_words, args.workers, args.batch_size):
                        successful_corpora += 1
                else:
                    # Check for sub-corpora
                    for sub_dir in sorted(top_level_dir.iterdir()):
                        if sub_dir.is_dir():
                            sub_index = sub_dir / "index.jsonl"
                            sub_docs = sub_dir / "documents"
                            if sub_index.exists() or sub_docs.exists():
                                total_corpora += 1
                                if build_single_corpus(sub_dir, args.workspace, args.max_documents, args.max_words, args.workers, args.batch_size):
                                    successful_corpora += 1
        
        print(f"\nBuild complete: {successful_corpora}/{total_corpora} corpora built successfully")
        return 0 if successful_corpora == total_corpora else 1
    else:
        # Neither --corpus nor --build-all specified
        print("Error: Please specify either --corpus <path> or --build-all")
        print("Use --list-corpora to see available corpora")
        return 1


def corpus_has_documents(corpus_path: Path) -> bool:
    """Check if a corpus path contains document data"""
    index_file = corpus_path / "index.jsonl"
    documents_dir = corpus_path / "documents"
    return index_file.exists() or documents_dir.exists()


def build_single_corpus(
    corpus_path: Path,
    workspace_path: Path,
    max_documents: Optional[int] = None,
    max_words: int = 200,
    workers: Optional[int] = None,
    batch_size: int = 32
) -> bool:
    """Build a single corpus and return success status"""
    try:
        print(f"\nBuilding corpus: {corpus_path.name}")
        print(f"  Using max_words={max_words} for chunking")
        
        # Determine number of workers
        if workers is None:
            # Auto-detect: use CPU count - 1, minimum 1
            num_workers = max(1, cpu_count() - 1)
        else:
            num_workers = max(1, workers)
        
        # Create embedded corpus instance to get document list
        embedded_corpus = EmbeddedCorpus(
            corpus_path=corpus_path,
            workspace_path=workspace_path,
            chunker=TextChunker(TextChunkerConfig(max_words=max_words)),
            embedder=SentenceTransformerEmbedder(lazy_load=True)
        )
        
        # Get corpus info
        info = embedded_corpus.get_corpus_info()
        total_docs = info['document_count']
        print(f"  Contains {total_docs} documents")
        
        # Get document IDs
        doc_ids = list(embedded_corpus.corpus.list_ids())
        if max_documents:
            doc_ids = doc_ids[:max_documents]
            print(f"  Processing only the first {max_documents} documents")
        
        if not doc_ids:
            print(f"  No documents to process")
            return True
        
        # Decide whether to use parallel processing
        if num_workers > 1 and len(doc_ids) > 1:
            print(f"  Using {num_workers} parallel workers")
            
            # Prepare arguments for worker processes
            worker_args = [
                (doc_id, corpus_path, workspace_path, max_words)
                for doc_id in doc_ids
            ]
            
            # Process documents in parallel
            successful = 0
            failed = 0
            
            with Pool(processes=num_workers) as pool:
                # Use imap_unordered for better progress tracking
                results_iter = pool.imap_unordered(process_document_worker, worker_args)
                
                # Wrap with tqdm for progress bar
                with tqdm(total=len(doc_ids), desc="  Processing documents", unit="doc") as pbar:
                    for doc_id, success, error_msg in results_iter:
                        if success:
                            successful += 1
                        else:
                            failed += 1
                            print(f"\n  ✗ Failed to process {doc_id}: {error_msg}")
                        pbar.update(1)
            
            if failed > 0:
                print(f"  ⚠ Processed {successful} documents successfully, {failed} failed")
            
        else:
            # Sequential processing (original behavior)
            if num_workers == 1:
                print(f"  Using sequential processing (1 worker)")
            
            embedded_corpus.build_all(max_documents=max_documents)
        
        # Build index after all documents are processed
        print(f"  Building FAISS index...")
        embedded_corpus.build_index()
        
        print(f"  ✓ Successfully built corpus: {corpus_path.name}")
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to build corpus {corpus_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()
