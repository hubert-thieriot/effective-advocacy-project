"""
NER (Named Entity Recognition) stage for discourse analysis pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from efi_core.pipeline import PipelineStage
from efi_analyser.ner import (
    StanzaNERExtractor,
    EntityMention,
    ChunkEntities,
    DocumentEntities,
    NERResult,
    NERConsolidator,
    ConsolidatedNERResult,
)
from efi_analyser.scorers.openai_interface import OpenAIInterface, OpenAIConfig

from .base import StageContext


class NERStage(PipelineStage[StageContext, NERResult]):
    """Extract named entities from chunks with significant framing."""

    def should_run(self, input_data: Optional[StageContext]) -> bool:
        if input_data is None:
            return False
        return bool(input_data.config.ner.enabled)

    def execute(self, input_data: StageContext) -> NERResult:
        """Main execution: runs entity extraction then consolidation."""
        config = input_data.config
        state = input_data.state
        paths = input_data.paths

        # Step 1: Extract entities (with its own caching)
        ner_result = self._run_entity_extraction(input_data)

        # Step 2: Run consolidation if enabled (with its own caching)
        if config.ner.consolidation.enabled:
            consolidated = self._run_consolidation(
                ner_result,
                config.ner.consolidation,
                paths,
                config,
            )
            if consolidated:
                ner_result.consolidated = consolidated
                # Save updated result with consolidation
                ner_result.save(paths.ner_entities_path)

        return ner_result

    def _run_entity_extraction(self, input_data: StageContext) -> NERResult:
        """Run entity extraction substage with caching."""
        config = input_data.config
        state = input_data.state
        paths = input_data.paths

        ner_config = config.ner
        ner_path = paths.ner_entities_path

        frame_classifications = state.frame_classifications
        if not frame_classifications:
            self.logger.warning("No frame classifications available for NER extraction.")
            return NERResult(language=ner_config.language, frame_threshold=ner_config.frame_threshold)

        # Collect all chunks that meet the threshold
        all_chunks_to_process = self._collect_significant_chunks(
            frame_classifications,
            threshold=ner_config.frame_threshold,
        )

        if not all_chunks_to_process:
            self.logger.info("No chunks met the frame threshold for NER extraction.")
            return NERResult(language=ner_config.language, frame_threshold=ner_config.frame_threshold)

        # Try to load cached results for incremental processing
        cached: Optional[NERResult] = None
        cached_chunk_ids: set = set()
        if ner_path.exists():
            try:
                cached = NERResult.load(ner_path)
                cached_chunk_ids = self._get_chunk_ids(cached)
                self.logger.info(
                    f"Found cached NER results: {cached.n_chunks} chunks already processed"
                )
                if not self._languages_match(cached.language, ner_config.language):
                    self.logger.warning(
                        "Cached NER language (%s) does not match config (%s).",
                        self._format_language(cached.language),
                        self._format_language(ner_config.language),
                    )
                    if not config.regenerate_report_only:
                        cached = None
                        cached_chunk_ids = set()
            except Exception as e:
                self.logger.warning(f"Failed to load cached NER results: {e}")
                cached = None

        # If regenerate_report_only, just return cached (no new extraction, but consolidation may run)
        if config.regenerate_report_only and cached is not None:
            self.logger.info("Using cached entity extraction results (regenerate_report_only=True).")
            return cached

        # Filter to only new chunks (not already in cache)
        new_chunks = [c for c in all_chunks_to_process if c["chunk_id"] not in cached_chunk_ids]

        # If reload_ner is True and all chunks are cached, return cached
        if ner_config.reload_ner and not new_chunks and cached is not None:
            self.logger.info("All chunks already processed, using cached results.")
            return cached

        if not new_chunks:
            if cached is not None:
                self.logger.info("No new chunks to process, using cached entity extraction results.")
                return cached
            self.logger.info("No chunks to process for NER extraction.")
            return NERResult(language=ner_config.language, frame_threshold=ner_config.frame_threshold)

        self.logger.info(
            f"Running NER on {len(new_chunks)} new chunks "
            f"({len(all_chunks_to_process)} total, {len(cached_chunk_ids)} cached) "
            f"(threshold={ner_config.frame_threshold}, lang={self._format_language(ner_config.language)})"
        )

        # Initialize extractor
        extractor = StanzaNERExtractor(
            language=ner_config.language,
            entity_types=ner_config.entity_types,
        )

        # Extract entities from new chunks only
        texts = [c["text"] for c in new_chunks]
        entity_results = extractor.extract_batch(
            texts,
            batch_size=ner_config.batch_size,
            show_progress=True,
        )

        # Build result for new chunks
        new_result = self._build_result(
            new_chunks,
            entity_results,
            language=ner_config.language,
            threshold=ner_config.frame_threshold,
        )

        # Merge with cached results
        if cached is not None:
            result = self._merge_results(cached, new_result)
        else:
            result = new_result

        # Save combined results
        result.save(ner_path)
        self.logger.info(
            f"NER extraction complete: {result.n_documents} docs, "
            f"{result.n_chunks} chunks, {result.n_entities} entities "
            f"({len(new_chunks)} newly processed)"
        )

        return result

    def _get_chunk_ids(self, ner_result: NERResult) -> set:
        """Extract all chunk_ids from an NERResult."""
        chunk_ids = set()
        for doc in ner_result.documents:
            for chunk in doc.chunks:
                chunk_ids.add(chunk.chunk_id)
        return chunk_ids

    def _merge_results(self, cached: NERResult, new: NERResult) -> NERResult:
        """Merge cached and new NER results."""
        # Build a map of doc_id -> DocumentEntities for cached
        doc_map: Dict[str, DocumentEntities] = {}
        for doc in cached.documents:
            doc_map[doc.doc_id] = DocumentEntities(
                doc_id=doc.doc_id,
                chunks=list(doc.chunks),
            )

        # Add new chunks to existing docs or create new docs
        for new_doc in new.documents:
            if new_doc.doc_id in doc_map:
                doc_map[new_doc.doc_id].chunks.extend(new_doc.chunks)
            else:
                doc_map[new_doc.doc_id] = new_doc

        return NERResult(
            documents=list(doc_map.values()),
            language=new.language,
            frame_threshold=new.frame_threshold,
        )

    def _collect_significant_chunks(
        self,
        frame_classifications,
        threshold: float,
    ) -> List[Dict]:
        """Collect chunks where max frame probability >= threshold.

        Returns list of dicts with: doc_id, chunk_id, text, frame_id, frame_score
        """
        chunks = []

        for doc in frame_classifications:
            doc_id = doc.doc_id
            if not doc_id:
                continue

            doc_chunks = doc.payload.get("chunks", [])
            if not isinstance(doc_chunks, list):
                continue

            for chunk in doc_chunks:
                if not isinstance(chunk, dict):
                    continue

                text = str(chunk.get("text", "")).strip()
                if not text:
                    continue

                probabilities = chunk.get("probabilities", {})
                if not probabilities:
                    continue

                # Find max probability frame
                max_frame_id = None
                max_score = 0.0
                for frame_id, score in probabilities.items():
                    if score > max_score:
                        max_score = score
                        max_frame_id = frame_id

                if max_score >= threshold and max_frame_id:
                    chunk_id = str(chunk.get("chunk_id", "")).strip()
                    chunks.append({
                        "doc_id": doc_id,
                        "chunk_id": chunk_id or f"{doc_id}:unknown",
                        "text": text,
                        "frame_id": max_frame_id,
                        "frame_score": max_score,
                    })

        return chunks

    def _build_result(
        self,
        chunks_to_process: List[Dict],
        entity_results: List[List[EntityMention]],
        language: Union[str, List[str]],
        threshold: float,
    ) -> NERResult:
        """Build NERResult from extraction results."""
        # Group by document
        doc_chunks: Dict[str, List[Tuple[Dict, List[EntityMention]]]] = {}

        for chunk_info, entities in zip(chunks_to_process, entity_results):
            doc_id = chunk_info["doc_id"]
            if doc_id not in doc_chunks:
                doc_chunks[doc_id] = []
            doc_chunks[doc_id].append((chunk_info, entities))

        # Build document entities
        documents = []
        for doc_id, chunk_list in doc_chunks.items():
            chunk_entities = []
            for chunk_info, entities in chunk_list:
                chunk_entities.append(
                    ChunkEntities(
                        chunk_id=chunk_info["chunk_id"],
                        text=chunk_info["text"],
                        frame_id=chunk_info["frame_id"],
                        frame_score=chunk_info["frame_score"],
                        entities=entities,
                    )
                )
            documents.append(DocumentEntities(doc_id=doc_id, chunks=chunk_entities))

        return NERResult(
            documents=documents,
            language=language,
            frame_threshold=threshold,
        )

    def _run_consolidation(
        self,
        ner_result: NERResult,
        consolidation_config,
        paths,
        config,
    ) -> Optional[ConsolidatedNERResult]:
        """Run NER consolidation substage with caching.

        This runs independently of entity extraction, so even if entities
        are loaded from cache, consolidation can still be performed or
        loaded from its own cache.
        """
        consolidated_path = paths.ner_consolidated_path

        # Try to load cached consolidated results
        cached_consolidated: Optional[ConsolidatedNERResult] = None
        if consolidated_path.exists():
            try:
                cached_consolidated = ConsolidatedNERResult.load(consolidated_path)
                self.logger.info(
                    f"Found cached consolidated NER results: {cached_consolidated.n_entities} entities"
                )
            except Exception as e:
                self.logger.warning(f"Failed to load cached consolidated results: {e}")

        # If regenerate_report_only, return cached if available
        if config.regenerate_report_only and cached_consolidated is not None:
            self.logger.info("Using cached consolidation results (regenerate_report_only=True).")
            return cached_consolidated

        # Check if we need to re-run consolidation
        # Re-run if: no cache exists, or if the raw entity count has changed
        should_consolidate = (
            cached_consolidated is None or
            cached_consolidated.raw_entity_count != ner_result.n_entities
        )

        if not should_consolidate:
            self.logger.info("Consolidation already up-to-date, using cached results.")
            return cached_consolidated

        self.logger.info("Running NER consolidation...")

        try:
            # Initialize LLM client
            llm_config = OpenAIConfig(
                model=consolidation_config.model,
                temperature=consolidation_config.temperature,
                flex_processing=consolidation_config.flex_processing,
            )
            llm_client = OpenAIInterface(name="ner_consolidation", config=llm_config)

            # Initialize consolidator
            consolidator = NERConsolidator(
                llm_client=llm_client,
                batch_size=consolidation_config.batch_size,
                min_count=consolidation_config.min_count,
                entity_types=consolidation_config.entity_types,
                guidance=consolidation_config.guidance,
            )

            # Run consolidation
            domain = getattr(config.framing, "domain", None) or ""
            consolidated = consolidator.consolidate(ner_result, domain=domain)

            # Save results
            consolidated.save(consolidated_path)
            self.logger.info(
                f"Consolidation complete: {consolidated.raw_entity_count} raw → "
                f"{consolidated.n_entities} consolidated"
            )

            return consolidated

        except Exception as e:
            self.logger.error(f"Consolidation failed: {e}")
            return None

    def get_metadata(self) -> dict:
        return {"stage": self.name}

    @staticmethod
    def _normalize_languages(language: Union[str, List[str]]) -> List[str]:
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
        return langs or ["en"]

    @classmethod
    def _languages_match(cls, left: Union[str, List[str]], right: Union[str, List[str]]) -> bool:
        left_norm = set(cls._normalize_languages(left))
        right_norm = set(cls._normalize_languages(right))
        return left_norm == right_norm

    @classmethod
    def _format_language(cls, language: Union[str, List[str]]) -> str:
        langs = cls._normalize_languages(language)
        return ", ".join(langs)
