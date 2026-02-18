"""NER extraction using Stanza."""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Union

from .types import EntityMention


class StanzaNERExtractor:
    """Named Entity Recognition using Stanza (Stanford NLP).

    Stanza provides pre-trained NER models for 60+ languages with consistent API.
    """

    def __init__(
        self,
        language: Union[str, Sequence[str]] = "en",
        entity_types: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize the Stanza NER extractor.

        Args:
            language: Stanza language code or list of codes (e.g., "en" or ["en", "es"]).
            entity_types: Optional filter for entity types (e.g., ["PERSON", "ORG"]).
                         If None, all entity types are returned.
        """
        self.language, self._languages = self._normalize_languages(language)
        self.entity_types = set(entity_types) if entity_types else None
        self.logger = logging.getLogger(self.__class__.__name__)

        self._pipeline = None
        self._is_multilingual = False

    @staticmethod
    def _normalize_languages(
        language: Union[str, Sequence[str]],
    ) -> tuple[Union[str, List[str]], Optional[List[str]]]:
        if isinstance(language, (list, tuple, set)):
            raw = [str(item).strip().lower() for item in language if str(item).strip()]
        else:
            text = str(language).strip()
            if not text:
                raw = []
            elif "," in text:
                raw = [part.strip().lower() for part in text.split(",") if part.strip()]
            else:
                raw = [text.lower()]

        seen = set()
        langs = []
        for lang in raw:
            if lang not in seen:
                seen.add(lang)
                langs.append(lang)

        if not langs:
            return "en", None
        if len(langs) == 1:
            return langs[0], None
        return langs, langs

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

        if self._languages:
            language_list = ", ".join(self._languages)
            self.logger.info(
                f"Loading Stanza multilingual NER models for languages: {language_list}"
            )

            try:
                stanza.download("multilingual", verbose=False)
            except Exception as e:
                self.logger.warning(f"Could not download Stanza multilingual model: {e}")

            for lang in self._languages:
                try:
                    stanza.download(lang, processors="tokenize,ner", verbose=False)
                except Exception as e:
                    self.logger.warning(f"Could not download Stanza model for {lang}: {e}")

            from stanza.pipeline.multilingual import MultilingualPipeline

            lang_id_config = {"langid_lang_subset": self._languages}
            lang_configs = {
                lang: {"processors": "tokenize,ner", "verbose": False}
                for lang in self._languages
            }
            self._pipeline = MultilingualPipeline(
                lang_id_config=lang_id_config,
                lang_configs=lang_configs,
                max_cache_size=len(self._languages),
            )
            self._is_multilingual = True
            return

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
            if self._is_multilingual:
                docs = self._pipeline([text])
                doc = docs[0] if docs else None
            else:
                doc = self._pipeline(text)
        except Exception as e:
            self.logger.warning(f"NER extraction failed: {e}")
            return []

        if not doc:
            return []

        return self._extract_from_doc(doc)

    def _extract_from_doc(self, doc) -> List[EntityMention]:
        """Extract entities from a processed Stanza document."""
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
