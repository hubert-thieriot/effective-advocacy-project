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

    def should_run(self, input_data: Optional[StageContext]) -> bool:
        """Check if aggregation should run.

        Aggregation runs if:
        - We're NOT in regenerate-report-only mode (aggregates already exist)
        - Aggregates don't exist or are out of sync
        - OR reload_aggregates is True
        """
        if input_data is None:
            return False

        config = input_data.config
        paths = input_data.paths

        # Skip if aggregates directory not configured
        if not paths.aggregates_dir:
            self.logger.warning("Aggregates directory not configured")
            return False

        # In regenerate mode, always skip aggregation (just use cached)
        if config.regenerate_report_only:
            self.logger.info("Regenerate mode - will load cached aggregates")
            return False

        # If reload requested, run to rebuild
        if config.reload_aggregates:
            self.logger.info("Reload requested - will rebuild aggregates")
            return True

        # Check if aggregates exist
        aggregates_dir = paths.aggregates_dir
        if not aggregates_dir.exists():
            self.logger.info("Aggregates directory doesn't exist - will build")
            return True

        # Try to load aggregates
        aggregates = Aggregates.load(aggregates_dir)
        if aggregates is None:
            self.logger.info("No cached aggregates found - will build")
            return True

        # Check if aggregates are in sync with classifications
        if input_data.state.classifications:
            classifications = input_data.state.classifications
            if classifications.n_docs > 0:
                agg_doc_ids = {agg.doc_id for agg in aggregates.documents_weighted}
                class_doc_ids = {doc.doc_id for doc in classifications if doc.doc_id}
                overlap = len(agg_doc_ids & class_doc_ids)
                total_agg = len(agg_doc_ids)

                # If less than 50% overlap, rebuild
                if total_agg > 0 and overlap / total_agg < 0.5:
                    self.logger.warning(
                        f"Aggregates out of sync ({overlap}/{total_agg} match) - will rebuild"
                    )
                    return True

        # Aggregates exist and are in sync
        self.logger.info("Aggregates exist and are in sync - skipping")
        return False

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
        filter_spec = None
        if hasattr(input_data, 'filter_spec'):
            filter_spec = input_data.filter_spec

        builder = AggregatesBuilder(
            aggregates_dir=paths.aggregates_dir,
            frame_ids=[frame.frame_id for frame in state.schema.frames],
            corpus_names=input_data.corpus_names,
            application_top_k=config.annotation.top_k,
            min_threshold_weighted=config.agg_min_threshold_weighted,
            normalize_weighted=config.agg_normalize_weighted,
            min_threshold_occurrence=config.agg_min_threshold_occurrence,
            filter_spec=filter_spec,
        )

        aggregates = builder.build(classifications)

        self.logger.info(
            f"Built {len(aggregates.documents_weighted)} weighted and "
            f"{len(aggregates.documents_occurrence)} occurrence aggregates"
        )

        return aggregates

    def load_cached(self) -> Aggregates:
        """Load aggregates from cache."""
        # Note: paths is not directly available here, need to store in __init__ or get from somewhere
        # For now, raise NotImplementedError - caching handled in should_run()
        raise NotImplementedError("Caching handled via should_run() logic")

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
