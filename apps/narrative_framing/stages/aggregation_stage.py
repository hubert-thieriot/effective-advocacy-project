"""
Aggregation stage for narrative framing pipeline.

Builds temporal and domain aggregations from document classifications.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from efi_core.pipeline import PipelineStage

from apps.narrative_framing.aggregates import Aggregates, AggregatesBuilder
from .base import StageContext


class AggregationStage(PipelineStage[StageContext, Aggregates]):
    """
    Pipeline stage for building aggregations.

    This stage:
    1. Checks if aggregates should be reloaded from cache
    2. Verifies aggregates match current classifications
    3. Builds new aggregates when needed
    4. Saves aggregates for future use

    The stage produces temporal and domain aggregations that are used
    for reporting and visualization.
    """

    def __init__(self, name: str, output_dir: Path):
        super().__init__(name, output_dir)
        self._aggregates_dir: Optional[Path] = None

    def should_run(self, input_data: Optional[StageContext]) -> bool:
        """Check if aggregation should run.

        Aggregation runs if:
        - We're NOT in regenerate-report-only mode (aggregates already exist)
        - Aggregates don't exist or are out of sync
        - OR reload_aggregates is False (rebuild)
        """
        if input_data is None:
            return False

        config = input_data.config
        paths = input_data.paths
        state = input_data.state

        # Store path for later use (before any early returns)
        self._aggregates_dir = paths.aggregates_dir

        # Skip if no classifications or assignments available (induction-only mode)
        if not state.classifications and not state.assignments:
            self.logger.info("No classifications or assignments available - skipping aggregation")
            return False

        # Skip if aggregates directory not configured
        if not paths.aggregates_dir:
            self.logger.warning("Aggregates directory not configured")
            return False

        # In regenerate mode, always skip aggregation (just use cached)
        if config.regenerate_report_only:
            self.logger.info("Regenerate mode - will load cached aggregates")
            return False

        # If reload_aggregates is True, use cached aggregates if available
        if config.reload_aggregates:
            aggregates = Aggregates.load(paths.aggregates_dir)
            if aggregates is not None:
                self.logger.info("Reload from cache requested - will load cached aggregates")
                return False
            self.logger.warning("Reload requested but cached aggregates not found - will build")
            return True

        # If reload_aggregates is False, always rebuild
        self.logger.info("reload_aggregates=False - will rebuild aggregates")
        return True

    def execute(self, input_data: StageContext) -> Aggregates:
        """Build aggregations from classifications.

        Args:
            input_data: Stage context with config, paths, and state

        Returns:
            Aggregates object with temporal and domain aggregations

        Raises:
            ValueError: If schema or classifications are missing
        """
        config = input_data.config
        paths = input_data.paths
        state = input_data.state

        # Check preconditions
        if not state.schema:
            raise ValueError("Schema is required for aggregation")

        # Get classifications - use assignments if classifier disabled
        classifications = state.classifications
        if not classifications and state.assignments:
            self.logger.info("Using LLM annotations for aggregation (classifier disabled)")
            classifications = state.assignments.to_classifications()

        if not classifications:
            raise ValueError("Classifications are required for aggregation")

        # Build aggregates
        self.logger.info("Building aggregations...")

        # Get filter spec if available (from workflow filter)
        filter_spec = input_data.filter_spec

        # Create aggregates directory
        paths.aggregates_dir.mkdir(parents=True, exist_ok=True)

        builder = AggregatesBuilder(
            aggregates_dir=paths.aggregates_dir,
            frame_ids=[frame.frame_id for frame in state.schema.frames],
            corpus_names=input_data.corpus_names,
            application_top_k=config.annotation.top_k,
            min_threshold_weighted=config.agg_min_threshold_weighted or 0.0,
            normalize_weighted=config.agg_normalize_weighted if config.agg_normalize_weighted is not None else False,
            min_threshold_occurrence=config.agg_min_threshold_occurrence or 0.0,
            filter_spec=filter_spec,
        )

        aggregates = builder.build(classifications)

        self.logger.info(
            f"Built {len(aggregates.documents_weighted)} weighted and "
            f"{len(aggregates.documents_occurrence)} occurrence aggregates"
        )

        return aggregates

    def load_cached(self) -> Optional[Aggregates]:
        """Load aggregates from cache."""
        if self._aggregates_dir and self._aggregates_dir.exists():
            aggregates = Aggregates.load(self._aggregates_dir)
            if aggregates:
                self.logger.info(f"Loaded cached aggregates from {self._aggregates_dir}")
                return aggregates
        # Return None if not found (e.g., induction-only mode)
        return None

    def save_result(self, result: Aggregates) -> None:
        """Save aggregates (already saved by AggregatesBuilder)."""
        # AggregatesBuilder already saves to disk
        pass

    def get_metadata(self) -> dict:
        """Get metadata about aggregation."""
        return {
            "stage": self.name,
            "note": "Builds temporal and domain aggregations from classifications"
        }
