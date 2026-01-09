"""
Corpus loading stage for narrative framing pipeline.

Loads and initializes embedded corpora for analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict

from efi_core.pipeline import PipelineStage

from efi_corpus.embedded import EmbeddedCorpus
from efi_analyser.chunkers.sentence_chunker import SentenceChunker
from efi_analyser.frames.corpora import EmbeddedCorpora
from efi_analyser.frames.classifier import EmbeddedCorporaSampler

from .base import StageContext


class CorpusLoadingStage(PipelineStage[StageContext, EmbeddedCorpora]):
    """
    Pipeline stage for loading corpora.

    This stage:
    1. Loads embedded corpora from disk
    2. Initializes sampler for passage collection
    3. Stores in workflow state for use by other stages

    This is typically the first stage in the pipeline.
    """

    def should_run(self, input_data: Optional[StageContext]) -> bool:
        """Always run corpus loading."""
        return input_data is not None

    def execute(self, input_data: StageContext) -> EmbeddedCorpora:
        """Load embedded corpora.

        Args:
            input_data: Stage context

        Returns:
            EmbeddedCorpora wrapper containing all loaded corpora
        """
        config = input_data.config
        state = input_data.state

        self.logger.info(f"Loading {len(input_data.corpus_names)} corpora...")

        # Load each corpus
        corpora_map = {}
        for corpus_name in input_data.corpus_names:
            corpus_path = config.corpora_root / corpus_name

            if not corpus_path.exists():
                raise FileNotFoundError(f"Corpus not found: {corpus_path}")

            self.logger.info(f"  Loading {corpus_name}...")

            # Create chunker
            # Note: SentenceChunker doesn't take parameters, uses defaults
            chunker = SentenceChunker()

            # Load corpus
            # Note: dataclass inheritance requires both data_path (base) and corpus_path (subclass)
            corpus = EmbeddedCorpus(
                data_path=corpus_path,
                corpus_path=corpus_path,
                workspace_path=config.workspace_root,
                chunker=chunker,
                embedder=None,  # Not needed for frame analysis
            )

            corpora_map[corpus_name] = corpus

        self.logger.info(f"Loaded {len(corpora_map)} corpora")

        # Wrap in EmbeddedCorpora for classifier compatibility
        embedded_corpora = EmbeddedCorpora(corpora_map)

        # Create sampler
        sampler = EmbeddedCorporaSampler(embedded_corpora)
        state.sampler = sampler
        # Don't set state.corpora_map here - it will be set by pipeline from return value

        return embedded_corpora

    def get_metadata(self) -> dict:
        return {
            "stage": self.name,
            "note": "Loads embedded corpora for analysis"
        }
