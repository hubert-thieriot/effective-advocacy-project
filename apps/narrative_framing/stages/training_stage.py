"""
Classifier training stage for narrative framing pipeline.

Trains a transformer-based classifier on LLM-annotated data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from efi_core.pipeline import PipelineStage

from efi_analyser.frames.classifier import FrameClassifierTrainer, FrameLabelSet

from .base import StageContext


class TrainingStage(PipelineStage[StageContext, None]):
    """
    Pipeline stage for training the frame classifier.

    This stage:
    1. Checks if classifier should be trained (enabled in config)
    2. Uses LLM annotations as training data
    3. Trains transformer model
    4. Saves trained model to disk

    Training is optional - if disabled, classification uses LLM directly.
    """

    def should_run(self, input_data: Optional[StageContext]) -> bool:
        """Check if training should run."""
        if input_data is None:
            return False

        config = input_data.config
        state = input_data.state
        paths = input_data.paths

        # Skip if classifier disabled
        if not config.classifier.enabled:
            self.logger.info("Classifier disabled - skipping training")
            return False

        # Skip in regenerate mode
        if config.regenerate_report_only:
            self.logger.info("Regenerate mode - skipping training")
            return False

        # If reload requested and classifier exists, skip training
        if config.reload_classifier:
            if paths.classifier_dir and paths.classifier_dir.exists():
                self.logger.info("Reload from cache requested - skipping training")
                return False
            if not input_data.allow_new_work:
                self.logger.warning("Reload requested but classifier missing and new work disabled")
                return False
            self.logger.warning("Reload requested but classifier missing - will train")

        # Need schema and assignments
        if not state.schema or not state.assignments:
            self.logger.warning("Schema or assignments missing - cannot train")
            return False

        # Run if reload requested or new work allowed
        return input_data.allow_new_work

    def execute(self, input_data: StageContext) -> None:
        """Train the classifier.

        Args:
            input_data: Stage context

        Returns:
            None (model saved to disk)
        """
        config = input_data.config
        state = input_data.state
        paths = input_data.paths

        if not state.schema or not state.assignments:
            raise ValueError("Schema and assignments required for training")

        self.logger.info("Training frame classifier...")

        # Build label set from LLM assignments
        label_set = FrameLabelSet.from_assignments(state.schema, state.assignments, source="llm")
        if not label_set.passages:
            self.logger.warning("Label set is empty; skipping classifier training")
            return None

        # Create classifier spec and update output dir
        spec = config.classifier.to_spec()
        spec.output_dir = str(paths.classifier_dir)

        # Set run name
        spec.run_name = f"{config.corpus}-frame-cls"

        # Create trainer
        trainer = FrameClassifierTrainer(spec)

        # Train and optionally cross-validate
        artifacts = trainer.run(
            label_set=label_set,
            assignments=state.assignments,
            cv_folds=config.classifier.cv_folds if config.classifier.cv_folds and config.classifier.cv_folds >= 2 else None,
        )

        # Save model
        if artifacts.model and paths.classifier_dir:
            paths.classifier_dir.mkdir(parents=True, exist_ok=True)
            artifacts.model.save(paths.classifier_dir)
            self.logger.info(f"Classifier saved to {paths.classifier_dir}")

        return None

    def get_metadata(self) -> dict:
        return {"stage": self.name, "classifier_enabled": True}
