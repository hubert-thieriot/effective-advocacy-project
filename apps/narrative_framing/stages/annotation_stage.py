"""
Annotation stage for narrative framing pipeline.

Uses LLM to annotate passages with narrative frames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List

from efi_core.pipeline import PipelineStage

from efi_analyser.frames import FrameAssignments, LLMFrameAnnotator
from efi_analyser.frames.classifier import SamplerConfig
from efi_analyser.scorers.openai_interface import OpenAIInterface, OpenAIConfig

from .base import StageContext


class AnnotationStage(PipelineStage[StageContext, FrameAssignments]):
    """
    Pipeline stage for annotating passages with narrative frames.

    This stage:
    1. Loads cached assignments if available
    2. Samples additional passages if needed
    3. Uses LLM to assign frames to passages
    4. Saves assignments to disk

    Annotations are used for training the classifier or as fallback
    when classifier is disabled.
    """

    def __init__(self, name: str, output_dir: Path):
        super().__init__(name, output_dir)
        self._assignments_path: Optional[Path] = None

    def should_run(self, input_data: Optional[StageContext]) -> bool:
        """Check if annotation should run."""
        if input_data is None:
            return False

        config = input_data.config
        paths = input_data.paths

        # Store path for cached loading
        self._assignments_path = paths.assignments_path

        if not input_data.state.schema:
            return False

        # Skip if annotation size is 0 or None
        if not config.annotation.size:
            self.logger.info("Annotation size is 0 or not set - skipping")
            return False

        # In regenerate mode, just load cached
        if config.regenerate_report_only:
            self.logger.info("Regenerate mode - will load cached annotations")
            return False

        # If reload requested, skip execution (load cached)
        # Otherwise run if new work is allowed
        if config.reload_annotation:
            if self._assignments_path and self._assignments_path.exists():
                # Check if cached size is less than target size
                try:
                    cached_assignments = FrameAssignments.load(self._assignments_path)
                    cached_size = cached_assignments.count
                    target_size = config.annotation.size
                    if target_size and cached_size < target_size:
                        self.logger.info(
                            f"Reload requested but cached size ({cached_size}) < target size ({target_size}) - will run"
                        )
                        return input_data.allow_new_work
                except Exception as e:
                    self.logger.warning(f"Failed to check cached assignments size: {e} - will run")
                    return input_data.allow_new_work
                self.logger.info("Reload from cache requested - will load cached annotations")
                return False
            self.logger.warning("Reload requested but cached annotations not found - will run")
            return input_data.allow_new_work

        return input_data.allow_new_work

    def execute(self, input_data: StageContext) -> FrameAssignments:
        """Annotate passages with frames.

        Args:
            input_data: Stage context

        Returns:
            FrameAssignments with annotated passages
        """
        config = input_data.config
        state = input_data.state
        paths = input_data.paths
        schema = state.schema

        if not schema:
            raise ValueError("Schema required for annotation")

        # Initialize assignments
        assignments = FrameAssignments()

        # Load cached assignments if they exist
        should_reload = config.reload_annotation or config.regenerate_report_only
        if should_reload and paths.assignments_path and paths.assignments_path.exists():
            try:
                assignments = FrameAssignments.load(paths.assignments_path)
                self.logger.info(f"Loaded {assignments.count} cached assignments")
            except Exception as e:
                self.logger.warning(f"Failed to load cached assignments: {e}")

        # Add new assignments if allowed and needed
        if input_data.allow_new_work:
            target_count = max(0, config.annotation.size)
            existing_count = assignments.count
            additional_needed = max(0, target_count - existing_count)

            if additional_needed > 0:
                # Collect new candidates
                exclude_ids: List[str] = [a.passage_id for a in assignments]
                if state.induction_samples:
                    exclude_ids.extend(pid for pid, _ in state.induction_samples)

                # Prepare sampler kwargs
                sampler_kwargs = {}
                if input_data.filter:
                    sampler_kwargs = input_data.filter.sampler_kwargs(
                        domain_whitelist=config.filter.document.domain_whitelist
                    )

                try:
                    new_candidates = state.sampler.collect_candidates(
                        SamplerConfig(
                            sample_size=additional_needed,
                            seed=config.seed + 1,
                            exclude_passage_ids=exclude_ids or None,
                            date_from=config.filter.date_from,
                            date_windows=config.filter.date_windows,
                            require_document_keywords=config.filter.document.keywords,
                            **sampler_kwargs,
                        )
                    )
                except ValueError as e:
                    self.logger.warning(f"Unable to gather {additional_needed} new passages: {e}")
                    new_candidates = []

                if new_candidates:
                    # Configure LLM
                    llm_config = OpenAIConfig(
                        model=config.annotation.model,
                        temperature=config.annotation.temperature or 0.0,
                        flex_processing=config.annotation.flex_processing or False,
                    )
                    annotator_client = OpenAIInterface(
                        name="frame_annotation",
                        config=llm_config
                    )

                    # Prepare prompt templates
                    app_sys_t = input_data.app_sys_t or "{system_message}"
                    app_usr_t = input_data.app_usr_t or "{user_message}"

                    resolved_dir = paths.results_dir / "prompts" / "frame_annotation" / "resolved"

                    # Create annotator
                    annotator = LLMFrameAnnotator(
                        llm_client=annotator_client,
                        batch_size=config.annotation.batch_size,
                        max_chars_per_passage=None,
                        chunk_overlap_chars=0,
                        system_template=app_sys_t,
                        user_template=app_usr_t,
                        resolved_messages_dir=resolved_dir,
                        resolved_messages_prefix="frame_annotation_batch",
                    )

                    # Annotate
                    self.logger.info(f"Annotating {len(new_candidates)} new passages...")
                    new_assignments = annotator.batch_assign(
                        schema,
                        new_candidates,
                        top_k=config.annotation.top_k,
                        show_progress=True,
                        progress_desc=f"Annotating frames ({llm_config.model})",
                        relevance_keywords=config.annotation.force_zero_if_no_keywords or config.relevance_keywords,
                        guidance=config.annotation.guidance,
                    )

                    assignments.extend(new_assignments)
                    self.logger.info(
                        f"Added {len(new_assignments)} new assignments "
                        f"(total: {assignments.count})"
                    )

                    # Save
                    if paths.assignments_path:
                        paths.assignments_path.parent.mkdir(parents=True, exist_ok=True)
                        assignments.save(paths.assignments_path)

        # Limit to target size if we have more
        target_count = config.annotation.size
        if target_count and assignments.count > target_count:
            self.logger.info(f"Limiting from {assignments.count} to {target_count}")
            assignments = assignments.select_random(target_count, config.seed + 97)

        return assignments

    def load_cached(self) -> FrameAssignments:
        """Load cached assignments."""
        if self._assignments_path and self._assignments_path.exists():
            assignments = FrameAssignments.load(self._assignments_path)
            self.logger.info(f"Loaded {assignments.count} cached assignments")
            return assignments
        # Return empty assignments if not found (e.g., when annotation size is 0)
        return FrameAssignments()

    def get_metadata(self) -> dict:
        return {"stage": self.name}
