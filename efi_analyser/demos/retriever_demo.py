#!/usr/bin/env python3
"""
Retriever Demo

Demonstrates the two retriever types and their capabilities:
- RetrieverBrute: Always uses brute-force cosine similarity
- RetrieverIndex: Uses FAISS indexes with auto-rebuild capability
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from efi_analyser.chunkers import SentenceChunker
from efi_analyser.embedders import SentenceTransformerEmbedder
from efi_core.retrieval import RetrieverBrute, RetrieverIndex
from efi_corpus.embedded.embedded_corpus import EmbeddedCorpus
from efi_library.embedded.embedded_library import EmbeddedLibrary


def demo_retrievers():
    """Demo different retriever types"""
    print("🔍 Retriever Architecture Demo")
    print("=" * 50)
    
    # Initialize components
    chunker = SentenceChunker()
    embedder = SentenceTransformerEmbedder(lazy_load=True)
    
    # Load embedded data
    print("\n📚 Loading embedded data...")
    try:
        embedded_corpus = EmbeddedCorpus(
            corpus_path=Path("corpora/air_quality/india"),
            workspace_path=Path("workspace"),
            chunker=chunker,
            embedder=embedder
        )
        
        embedded_library = EmbeddedLibrary(
            library_path=Path("libraries/crea_publications"),
            workspace_path=Path("workspace"),
            chunker=chunker,
            embedder=embedder
        )
        
        print("✓ Successfully loaded embedded data")
        
    except Exception as e:
        print(f"❌ Failed to load embedded data: {e}")
        return
    
    # Demo 1: Brute Force Retriever
    print("\n" + "=" * 50)
    print("1️⃣ RetrieverBrute Demo")
    print("=" * 50)
    
    brute_retriever = RetrieverBrute(
        embedded_data_source=embedded_corpus,
        chunker_spec=chunker.spec,
        embedder_spec=embedder.spec
    )
    
    print(f"Retriever Info: {brute_retriever.get_info()}")
    
    # Demo 2: Index Retriever with Auto-rebuild
    print("\n" + "=" * 50)
    print("2️⃣ RetrieverIndex Demo")
    print("=" * 50)
    
    indexed_retriever = RetrieverIndex(
        embedded_data_source=embedded_corpus,
        workspace_path=Path("workspace"),
        chunker_spec=chunker.spec,
        embedder_spec=embedder.spec,
        auto_rebuild=True,
        fallback_to_brute_force=True
    )
    
    print(f"Retriever Info: {indexed_retriever.get_info()}")
    print(f"Has index: {indexed_retriever.has_index()}")
    
    # Demo 4: Test a simple query
    print("\n" + "=" * 50)
    print("4️⃣ Query Demo")
    print("=" * 50)
    
    # Get a sample finding to use as query
    sample_finding = None
    for doc_findings in embedded_library.library.read_findings():
        if doc_findings.findings:
            sample_finding = doc_findings.findings[0]
            break
    
    if sample_finding:
        query_text = sample_finding.text[:100] + "..." if len(sample_finding.text) > 100 else sample_finding.text
        print(f"Query text: {query_text}")
        
        # Test with brute force retriever
        print(f"\n🔍 Testing with RetrieverBruteForce...")
        try:
            results = brute_retriever.query(query_text, top_k=3)
            print(f"Found {len(results)} results")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result.item_id} (score: {result.score:.4f})")
        except Exception as e:
            print(f"❌ Brute force query failed: {e}")
        
        # Test with indexed retriever
        print(f"\n🔍 Testing with RetrieverIndexed...")
        try:
            results = indexed_retriever.query(query_text, top_k=3)
            print(f"Found {len(results)} results")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result.item_id} (score: {result.score:.4f})")
        except Exception as e:
            print(f"❌ Indexed query failed: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Demo completed!")
    print("=" * 50)


if __name__ == "__main__":
    demo_retrievers()
