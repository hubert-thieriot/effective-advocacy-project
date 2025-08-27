#!/usr/bin/env python3
"""Demo building and matching documents using example data."""

import sys
import os
from pathlib import Path
from datetime import datetime
import time

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from efi_corpus.builders import ExampleCorpusBuilder
from efi_library import ExampleLibraryBuilder
from efi_analyser.chunkers import SentenceChunker
from efi_analyser.embedders import SentenceTransformerEmbedder
from efi_corpus.embedded.embedded_corpus import EmbeddedCorpus
from efi_library.embedded.embedded_library import EmbeddedLibrary
from efi_core.retrieval import RetrieverIndex
from efi_analyser.rescorers import NLIReScorer, NLIReScorerConfig


def main() -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    wall_start = datetime.now()
    t0 = time.perf_counter()
    print(f"Start: {wall_start:%Y-%m-%d %H:%M:%S}")
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
    print(f"✓ Built example data in {time.perf_counter() - t0:.2f}s")

    # Set up embedding components
    chunker = SentenceChunker()
    embedder = SentenceTransformerEmbedder(lazy_load=True)

    workspace = project_root / "workspace"
    embedded_corpus = EmbeddedCorpus(corpus_built, workspace, chunker, embedder)
    embedded_library = EmbeddedLibrary(library_built, workspace, chunker, embedder)

    step = time.perf_counter()
    print("Embedding corpus...")
    embedded_corpus.build_all()
    print(f"✓ Embedded corpus in {time.perf_counter() - step:.2f}s")

    step = time.perf_counter()
    print("Embedding library...")
    embedded_library.build_all()
    print(f"✓ Embedded library in {time.perf_counter() - step:.2f}s")

    retriever = RetrieverIndex(
        embedded_data_source=embedded_corpus,
        workspace_path=workspace,
        chunker_spec=chunker.spec,
        embedder_spec=embedder.spec,
        auto_rebuild=True
    )

    print("\nRunning document matching demo:")
    step = time.perf_counter()
    rescorer = NLIReScorer(NLIReScorerConfig(
        model_name="textattack/distilbert-base-uncased-MNLI",
        batch_size=1,
        max_length=256,
        device=-1,
        local_files_only=False,
    ))
    print(f"✓ NLI rescorer ready in {time.perf_counter() - step:.2f}s")
    processed = 0
    loop_start = time.perf_counter()
    for doc in embedded_library.library.iter_findings():
        for finding in doc.findings:
            print(f"\nFinding: {finding.text}")
            # Retrieve top-K candidates from FAISS
            f_start = time.perf_counter()
            results = retriever.query(finding.text, top_k=5)

            # Enrich with chunk text for NLI rescoring
            enriched = []
            for res in results:
                try:
                    doc_id, chunk_idx = res.item_id.split('_chunk_')
                    chunk_idx_int = int(chunk_idx)
                    chunks = embedded_corpus.get_chunks(doc_id, materialize_if_necessary=False)
                    chunk_text = chunks[chunk_idx_int].text if chunks and chunk_idx_int < len(chunks) else ""
                    res.metadata["text"] = chunk_text
                    enriched.append(res)
                except Exception:
                    enriched.append(res)

            # Rescore with NLI and keep top-2
            rescored = rescorer.rescore(finding.text, enriched)[:2]
            for res in rescored:
                print(f"  Match {res.item_id} score={res.score:.3f}")
            processed += 1
            print(f"  (Finding processed in {time.perf_counter() - f_start:.2f}s; total {processed})")

    total_s = time.perf_counter() - t0
    wall_end = datetime.now()
    print("\nDone.")
    print(f"End:   {wall_end:%Y-%m-%d %H:%M:%S}")
    print(f"Total elapsed: {total_s:.2f}s (~{total_s/60:.1f} min)")


if __name__ == "__main__":
    main()
