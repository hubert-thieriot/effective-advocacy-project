"""
Classifier training stage for narrative framing pipeline.

Trains a transformer-based classifier on LLM-annotated data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from efi_core.pipeline import PipelineStage

from efi_analyser.frames.classifier import FrameClassifierTrainer

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

        # Skip if classifier disabled
        if not config.classifier.enabled:
            self.logger.info("Classifier disabled - skipping training")
            return False

        # Skip in regenerate mode
        if config.regenerate_report_only:
            self.logger.info("Regenerate mode - skipping training")
            return False

        # Need schema and assignments
        if not state.schema or not state.assignments:
            self.logger.warning("Schema or assignments missing - cannot train")
            return False

        # Run if reload requested or new work allowed
        return config.reload_classifier or input_data.allow_new_work

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

        # Create trainer
        trainer = FrameClassifierTrainer(
            schema=state.schema,
            assignments=state.assignments,
            output_dir=paths.classifier_dir,
            spec=config.classifier.to_spec(),  # Convert Pydantic to FrameClassifierSpec
        )

        # Train
        if config.classifier.cv_folds and config.classifier.cv_folds >= 2:
            self.logger.info(f"Training with {config.classifier.cv_folds}-fold cross-validation")
            trainer.train_with_cv(n_folds=config.classifier.cv_folds)
        else:
            self.logger.info("Training single model")
            trainer.train()

        self.logger.info(f"Classifier saved to {paths.classifier_dir}")

        return None

    def get_metadata(self) -> dict:
        return {"stage": self.name, "classifier_enabled": True}
