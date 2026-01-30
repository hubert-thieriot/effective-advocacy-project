"""NER extraction using Stanza."""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

from .types import EntityMention


class StanzaNERExtractor:
    """Named Entity Recognition using Stanza (Stanford NLP).

    Stanza provides pre-trained NER models for 60+ languages with consistent API.
    """

    def __init__(
        self,
        language: str = "en",
        entity_types: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize the Stanza NER extractor.

        Args:
            language: Stanza language code (e.g., "en", "de", "fr", "zh").
            entity_types: Optional filter for entity types (e.g., ["PERSON", "ORG"]).
                         If None, all entity types are returned.
        """
        self.language = language
        self.entity_types = set(entity_types) if entity_types else None
        self.logger = logging.getLogger(self.__class__.__name__)

        self._pipeline = None

    def _ensure_pipeline(self) -> None:
        """Lazily initialize the Stanza pipeline."""
        if self._pipeline is not None:
            return

        try:
            import stanza
        except ImportError:
            raise ImportError(
                "Stanza is required for NER extraction. "
                "Install it with: pip install stanza"
            )

        # Download model if not already present
        self.logger.info(f"Loading Stanza NER model for language: {self.language}")
        try:
            stanza.download(self.language, processors="tokenize,ner", verbose=False)
        except Exception as e:
            self.logger.warning(f"Could not download Stanza model: {e}")

        # Initialize pipeline with tokenize and NER processors only (faster)
        self._pipeline = stanza.Pipeline(
            lang=self.language,
            processors="tokenize,ner",
            verbose=False,
        )

    def extract(self, texts: List[str]) -> List[List[EntityMention]]:
        """Extract named entities from a list of texts.

        Args:
            texts: List of text strings to process.

        Returns:
            List of entity lists, one per input text.
        """
        self._ensure_pipeline()

        results: List[List[EntityMention]] = []

        for text in texts:
            entities = self._extract_single(text)
            results.append(entities)

        return results

    def extract_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> List[List[EntityMention]]:
        """Extract entities from texts in batches.

        Args:
            texts: List of text strings to process.
            batch_size: Number of texts to process at once.
            show_progress: Whether to show progress bar.

        Returns:
            List of entity lists, one per input text.
        """
        self._ensure_pipeline()

        results: List[List[EntityMention]] = []

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(0, len(texts), batch_size), desc="NER extraction")
            except ImportError:
                iterator = range(0, len(texts), batch_size)
        else:
            iterator = range(0, len(texts), batch_size)

        for i in iterator:
            batch = texts[i : i + batch_size]
            batch_results = self.extract(batch)
            results.extend(batch_results)

        return results

    def _extract_single(self, text: str) -> List[EntityMention]:
        """Extract entities from a single text."""
        if not text or not text.strip():
            return []

        try:
            doc = self._pipeline(text)
        except Exception as e:
            self.logger.warning(f"NER extraction failed: {e}")
            return []

        entities: List[EntityMention] = []

        for sentence in doc.sentences:
            for ent in sentence.ents:
                # Filter by entity type if specified
                if self.entity_types and ent.type not in self.entity_types:
                    continue

                entities.append(
                    EntityMention(
                        text=ent.text,
                        type=ent.type,
                        start_char=ent.start_char,
                        end_char=ent.end_char,
                    )
                )

        return entities


__all__ = ["StanzaNERExtractor"]
