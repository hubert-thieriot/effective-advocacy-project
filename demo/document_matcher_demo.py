#!/usr/bin/env python3
"""Demo building and matching documents using example data."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from efi_corpus.builders import ExampleCorpusBuilder
from efi_library import ExampleLibraryBuilder
from efi_analyser.chunkers import SentenceChunker
from efi_analyser.embedders import SentenceTransformerEmbedder
from efi_corpus.embedded.embedded_corpus import EmbeddedCorpus
from efi_library.embedded.embedded_library import EmbeddedLibrary
from efi_core.retrieval import RetrieverBrute


def main() -> None:
    raw_root = project_root / "examples" / "raw" / "example_animal"
    built_root = project_root / "examples" / "example_animal"
    corpus_built = built_root / "corpus"
    library_built = built_root / "library"

    # Clean previous builds
    if corpus_built.exists():
        import shutil
        shutil.rmtree(corpus_built)
    if library_built.exists():
        import shutil
        shutil.rmtree(library_built)

    # Build corpus and library
    print("Building example corpus and library...")
    ExampleCorpusBuilder(raw_root / "corpus", corpus_built).build()
    ExampleLibraryBuilder(raw_root / "library", library_built).build()

    # Set up embedding components
    chunker = SentenceChunker()
    embedder = SentenceTransformerEmbedder(lazy_load=True)

    workspace = project_root / "workspace"
    embedded_corpus = EmbeddedCorpus(corpus_built, workspace, chunker, embedder)
    embedded_library = EmbeddedLibrary(library_built, workspace, chunker, embedder)

    print("Embedding corpus...")
    embedded_corpus.build_all(reindex=False)
    print("Embedding library...")
    embedded_library.build_all(reindex=False)

    retriever = RetrieverBrute(
        embedded_data_source=embedded_corpus,
        chunker_spec=chunker.spec,
        embedder_spec=embedder.spec,
    )

    print("\nRunning document matching demo:")
    for doc in embedded_library.library.iter_findings():
        for finding in doc.findings:
            print(f"\nFinding: {finding.text}")
            results = retriever.query(finding.text, top_k=2)
            for res in results:
                print(f"  Match {res.item_id} score={res.score:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
