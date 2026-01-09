"""
Classification stage for narrative framing pipeline.

Applies trained classifier or LLM to classify all documents.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from efi_core.pipeline import PipelineStage

from efi_analyser.frames.classifier import (
    FrameClassifier,
    FrameClassifierModel,
    DocumentClassifications,
)

from .base import StageContext


class ClassificationStage(PipelineStage[StageContext, DocumentClassifications]):
    """
    Pipeline stage for classifying documents.

    This stage:
    1. Loads trained classifier (if enabled) or uses LLM annotations
    2. Applies classification to all documents in corpora
    3. Saves classifications to disk

    Classifications are used for aggregation and reporting.
    """

    def __init__(self, name: str, output_dir: Path):
        super().__init__(name, output_dir)
        self._classifications_path: Optional[Path] = None
        self._classifications_dir: Optional[Path] = None

    def should_run(self, input_data: Optional[StageContext]) -> bool:
        """Check if classification should run."""
        if input_data is None or not input_data.state.schema:
            return False

        config = input_data.config
        paths = input_data.paths

        # Store paths for later use (before any early returns)
        self._classifications_path = paths.classifications_path
        self._classifications_dir = paths.classifications_dir

        # Skip in regenerate mode
        if config.regenerate_report_only:
            self.logger.info("Regenerate mode - will load cached classifications")
            return False

        # If reload requested, skip execution (load cached)
        if config.reload_classifications:
            has_cached = False
            if self._classifications_path and self._classifications_path.exists():
                has_cached = True
            elif self._classifications_dir and self._classifications_dir.exists():
                has_cached = any(self._classifications_dir.glob("*.json"))

            if has_cached:
                self.logger.info("Reload from cache requested - will load cached classifications")
                return False

            self.logger.warning("Reload requested but cached classifications not found - will run")
            return input_data.allow_new_work

        # Otherwise run if new work allowed
        return input_data.allow_new_work

    def execute(self, input_data: StageContext) -> DocumentClassifications:
        """Classify documents.

        Args:
            input_data: Stage context

        Returns:
            DocumentClassifications with frame predictions for all documents
        """
        config = input_data.config
        state = input_data.state
        paths = input_data.paths

        if not state.schema:
            raise ValueError("Schema required for classification")

        # Load cached if requested
        should_reload = config.reload_classifications or config.regenerate_report_only
        if should_reload:
            # Try loading from single file first (new format)
            if paths.classifications_path and paths.classifications_path.exists():
                try:
                    classifications = DocumentClassifications.load(paths.classifications_path)
                    self.logger.info(f"Loaded {classifications.n_docs} cached classifications from file")
                    return classifications
                except Exception as e:
                    self.logger.warning(f"Failed to load classifications file: {e}")

            # Fall back to loading from folder (old format)
            if paths.classifications_dir and paths.classifications_dir.exists():
                try:
                    classifications = DocumentClassifications.from_folder(paths.classifications_dir)
                    if classifications.n_docs > 0:
                        self.logger.info(f"Loaded {classifications.n_docs} cached classifications from folder")
                        return classifications
                except Exception as e:
                    self.logger.warning(f"Failed to load classifications from folder: {e}")

        # Need to run classification
        if not input_data.allow_new_work:
            raise RuntimeError("Classifications not found and new work disabled")

        # Check if classifier enabled
        if config.classifier.enabled:
            # Use trained classifier
            if not paths.classifier_dir or not paths.classifier_dir.exists():
                self.logger.warning("Classifier enabled but model not found - using LLM annotations")
                if state.assignments:
                    return state.assignments.to_classifications()
                raise ValueError("No classifier model and no LLM annotations available")

            self.logger.info("Classifying with trained model...")
            # Load the trained model
            classifier_model = FrameClassifierModel.load(paths.classifier_dir)

            # Debug: Check type of corpora_map
            self.logger.info(f"corpora_map type: {type(state.corpora_map)}")
            if state.corpora_map:
                self.logger.info(f"corpora_map has {len(state.corpora_map)} corpora")

            # Create corpus classifier wrapper
            classifier = FrameClassifier(
                model=classifier_model,
                corpora=state.corpora_map,
                batch_size=max(1, config.classifier.batch_size),
                seed=config.seed,
                require_keywords=config.filter.document.keywords,
                exclude_regex=config.filter.chunk.exclude_regex,
                exclude_min_hits=config.filter.chunk.exclude_min_hits,
                trim_after_markers=config.filter.chunk.trim_after_markers,
            )

            # Collect documents to classify
            sample_size = config.classification.size
            doc_ids = state.sampler.collect_docs(
                sample_size=sample_size,
                seed=config.seed,
                date_from=config.filter.date_from,
                date_windows=config.filter.date_windows,
                require_keywords=config.filter.document.keywords,
                domain_whitelist=config.filter.document.domain_whitelist,
            )

            self.logger.info(f"Classifying {len(doc_ids)} documents...")

            # Run classification
            classifications_list = classifier.run(
                sample_size=None,  # Already sampled
                doc_ids=doc_ids,
                output_dir=paths.classifications_dir,
            )

            # Convert list to DocumentClassifications
            classifications = DocumentClassifications(classifications_list)

        else:
            # Use LLM annotations
            self.logger.info("Classifier disabled - using LLM annotations")
            if not state.assignments:
                raise ValueError("No LLM annotations available for classification")
            classifications = state.assignments.to_classifications()

        # Save classifications to folder
        if paths.classifications_dir:
            paths.classifications_dir.mkdir(parents=True, exist_ok=True)
            classifications.to_folder(paths.classifications_dir)
            self.logger.info(f"Saved {classifications.n_docs} classifications")

        return classifications

    def load_cached(self) -> DocumentClassifications:
        """Load cached classifications."""
        # Load from folder
        if self._classifications_dir and self._classifications_dir.exists():
            try:
                classifications = DocumentClassifications.from_folder(self._classifications_dir)
                if classifications.n_docs > 0:
                    self.logger.info(f"Loaded {classifications.n_docs} cached classifications from folder")
                    return classifications
            except Exception as e:
                self.logger.warning(f"Failed to load classifications from folder: {e}")

        raise FileNotFoundError(f"Classifications not found at {self._classifications_path} or {self._classifications_dir}")

    def get_metadata(self) -> dict:
        return {"stage": self.name}
