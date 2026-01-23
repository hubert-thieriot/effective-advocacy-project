"""
Corpus loading stage for discourse analysis pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from efi_core.pipeline import PipelineStage
from efi_corpus.embedded import EmbeddedCorpus
from efi_analyser.chunkers.sentence_chunker import SentenceChunker
from efi_analyser.chunkers import TextChunker, TextChunkerConfig
from efi_analyser.frames.corpora import EmbeddedCorpora
from efi_analyser.frames.classifier import EmbeddedCorporaSampler

from .base import StageContext


class CorpusLoadingStage(PipelineStage[StageContext, EmbeddedCorpora]):
    """Load embedded corpora for analysis."""

    def should_run(self, input_data: Optional[StageContext]) -> bool:
        return input_data is not None

    def execute(self, input_data: StageContext) -> EmbeddedCorpora:
        config = input_data.config
        state = input_data.state

        self.logger.info(f"Loading {len(input_data.corpus_names)} corpora...")

        corpora_map = {}
        for corpus_name in input_data.corpus_names:
            corpus_path = config.corpora_root / corpus_name
            if not corpus_path.exists():
                raise FileNotFoundError(f"Corpus not found: {corpus_path}")

            if config.chunking.chunker_type == "text":
                chunker_config = TextChunkerConfig(
                    max_words=config.chunking.target_words,
                    spacy_model=config.chunking.chunker_model,
                )
                chunker = TextChunker(chunker_config)
                self.logger.info(
                    f"Using TextChunker (max_words={config.chunking.target_words}, model={config.chunking.chunker_model})"
                )
            else:
                chunker = SentenceChunker()
                self.logger.info("Using SentenceChunker")

            corpus = EmbeddedCorpus(
                data_path=corpus_path,
                corpus_path=corpus_path,
                workspace_path=config.workspace_root,
                chunker=chunker,
                embedder=None,
            )
            corpora_map[corpus_name] = corpus

        embedded_corpora = EmbeddedCorpora(corpora_map)
        state.sampler = EmbeddedCorporaSampler(embedded_corpora)
        return embedded_corpora

    def get_metadata(self) -> dict:
        return {"stage": self.name, "note": "Loads embedded corpora"}
