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

    def should_run(self, input_data: Optional[StageContext]) -> bool:
        """Check if classification should run."""
        if input_data is None or not input_data.state.schema:
            return False

        config = input_data.config

        # Skip in regenerate mode
        if config.regenerate_report_only:
            self.logger.info("Regenerate mode - will load cached classifications")
            return False

        # Run if reload requested or new work allowed
        return config.reload_classifications or input_data.allow_new_work

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
        if should_reload and paths.classifications_path and paths.classifications_path.exists():
            try:
                classifications = DocumentClassifications.load(paths.classifications_path)
                self.logger.info(f"Loaded {classifications.n_docs} cached classifications")
                return classifications
            except Exception as e:
                self.logger.warning(f"Failed to load cached classifications: {e}")

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
            classifier = FrameClassifier.from_pretrained(paths.classifier_dir)

            # Get corpus sample size
            sample_size = config.classification.size

            # Classify documents
            classifications = classifier.classify_corpora(
                schema=state.schema,
                corpora_map=state.corpora_map,
                corpus_names=input_data.corpus_names,
                output_dir=paths.classifications_dir,
                sample_size=sample_size,
                top_k=config.annotation.top_k,
                filter_spec=getattr(input_data, 'filter_spec', None),
            )

        else:
            # Use LLM annotations
            self.logger.info("Classifier disabled - using LLM annotations")
            if not state.assignments:
                raise ValueError("No LLM annotations available for classification")
            classifications = state.assignments.to_classifications()

        # Save
        if paths.classifications_path:
            classifications.save(paths.classifications_path)
            self.logger.info(f"Saved {classifications.n_docs} classifications")

        return classifications

    def load_cached(self) -> DocumentClassifications:
        """Load cached classifications."""
        raise NotImplementedError("Handled in should_run/execute logic")

    def get_metadata(self) -> dict:
        return {"stage": self.name}
