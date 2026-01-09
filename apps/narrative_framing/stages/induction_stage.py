"""
Frame induction stage for narrative framing pipeline.

Discovers narrative frames from a sample of documents using LLM analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from efi_core.pipeline import PipelineStage

from efi_analyser.frames import FrameInducer, FrameSchema
from efi_analyser.frames.classifier import SamplerConfig
from efi_analyser.scorers.openai_interface import OpenAIInterface, OpenAIConfig

from .base import StageContext
from apps.narrative_framing.run import load_schema, save_schema


class InductionStage(PipelineStage[StageContext, FrameSchema]):
    """
    Pipeline stage for inducing narrative frames from document samples.

    This stage:
    1. Checks if schema exists in cache (reload logic)
    2. Samples passages from corpora for analysis
    3. Uses LLM to discover narrative frames
    4. Saves schema to disk

    The induced schema defines the frames used in subsequent annotation,
    training, and classification stages.
    """

    def __init__(self, name: str, output_dir: Path):
        super().__init__(name, output_dir)
        self._schema_path: Optional[Path] = None

    def should_run(self, input_data: Optional[StageContext]) -> bool:
        """Check if induction should run.

        Induction runs if:
        - Schema doesn't exist in cache
        - OR reload_induction is True
        - Unless we're in regenerate-report-only mode (uses cached schema)
        """
        if input_data is None:
            return False

        config = input_data.config
        paths = input_data.paths

        # Store schema path for later use (before any early returns)
        self._schema_path = paths.schema_path

        # In regenerate mode, always skip (use cached)
        if config.regenerate_report_only:
            self.logger.info("Regenerate mode - will load cached schema")
            return False

        # If reload requested, skip execution (load cached)
        if config.reload_induction:
            if paths.schema_path and paths.schema_path.exists():
                self.logger.info("Reload from cache requested - will load cached schema")
                return False
            else:
                self.logger.warning("Reload requested but schema doesn't exist - will run induction")
                return True

        # If schema doesn't exist, run (unless not allowed)
        if not paths.schema_path or not paths.schema_path.exists():
            if not input_data.allow_new_work:
                self.logger.error("Schema not found and induction disabled (read-only mode)")
                return False
            self.logger.info("Schema not found - will run induction")
            return True

        # Schema exists and reload not requested - still run (fresh induction)
        self.logger.info("Schema exists but reload not requested - will run fresh induction")
        return True

    def execute(self, input_data: StageContext) -> FrameSchema:
        """Induce narrative frames from document samples.

        Args:
            input_data: Stage context with config, paths, and state

        Returns:
            FrameSchema with discovered frames

        Raises:
            ValueError: If required configuration is missing
            RuntimeError: If induction fails
        """
        config = input_data.config
        state = input_data.state
        paths = input_data.paths

        self.logger.info("Collecting passages for frame induction...")

        # Prepare relevance keywords (flatten if dict)
        induction_keywords = self._flatten_keywords(config.relevance_keywords)
        if induction_keywords:
            self.logger.info(f"Filtering with {len(induction_keywords)} relevance keywords")

        # Build sampler configuration
        # Note: This requires access to self.filter and self.sampler from workflow
        # For now, we'll need to pass these through the context
        if not state.sampler:
            raise ValueError("Sampler not initialized in workflow state")

        # Prepare sampler kwargs
        sampler_kwargs = {}
        if input_data.filter:
            sampler_kwargs = input_data.filter.sampler_kwargs(
                domain_whitelist=config.filter.document.domain_whitelist
            )

        # Override with relevance keywords if provided
        if induction_keywords:
            sampler_kwargs["keywords"] = induction_keywords

        # Collect sample passages
        sampler_config = SamplerConfig(
            sample_size=config.induction.size,
            seed=config.seed,
            date_from=config.filter.date_from,
            date_windows=config.filter.date_windows,
            require_document_keywords=config.filter.document.keywords,
            **sampler_kwargs,
        )

        induction_samples = state.sampler.collect_chunks(sampler_config)
        self.logger.info(f"Collected {len(induction_samples)} passages for induction")

        # Extract passage texts
        induction_passages = [text for _, text in induction_samples]

        # Configure LLM client
        llm_config = OpenAIConfig(
            model=config.induction.model,
            temperature=config.induction.temperature or 0.0,
        )
        inducer_client = OpenAIInterface(name="frame_induction", config=llm_config)

        # Determine batch size
        if config.induction.batch_size:
            max_per_call = config.induction.batch_size
        else:
            max_per_call = min(config.induction.size, 40)

        # Load prompt templates from context
        ind_sys_t = input_data.ind_sys_t or "{system_message}"
        ind_usr_t = input_data.ind_usr_t or "{user_message}"

        # Create inducer
        inducer = FrameInducer(
            llm_client=inducer_client,
            domain=config.domain,
            frame_target=config.induction.frame_target,
            max_passages_per_call=max_per_call,
            max_total_passages=config.induction.size * 2,
            induction_guidance=config.induction.guidance,
            system_template=ind_sys_t,
            user_template=ind_usr_t,
        )

        # Run induction
        self.logger.info("Running frame induction...")
        schema = inducer.induce(induction_passages)

        # Set schema ID if missing
        if not schema.schema_id and config.domain:
            schema.schema_id = config.domain.replace(" ", "_")

        self.logger.info(f"Induction produced {len(schema.frames)} frames")

        # Store samples in state for reference
        state.induction_samples = induction_samples

        # Save schema
        if paths.schema_path:
            paths.schema_path.parent.mkdir(parents=True, exist_ok=True)
            save_schema(paths.schema_path, schema)
            self.logger.info(f"Saved schema to {paths.schema_path}")

        # Save resolved prompts
        if inducer.emitted_messages and paths.results_dir:
            resolved_dir = paths.results_dir / "prompts" / "induction" / "resolved"
            self._save_resolved_messages(resolved_dir, "induction_call", inducer.emitted_messages)

        return schema

    def load_cached(self) -> FrameSchema:
        """Load schema from cache."""
        if self._schema_path and self._schema_path.exists():
            schema = load_schema(self._schema_path)
            self.logger.info(f"Loaded cached schema from {self._schema_path}")
            return schema
        raise FileNotFoundError(f"Schema not found at {self._schema_path}")

    def _flatten_keywords(self, keywords):
        """Flatten language-specific keywords dict to a single list."""
        if keywords is None:
            return None
        if isinstance(keywords, dict):
            # Flatten all language variants
            return [kw for kws in keywords.values() for kw in kws]
        return keywords

    def _save_resolved_messages(self, output_dir: Path, prefix: str, messages_by_call: list):
        """Save resolved LLM messages to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, messages in enumerate(messages_by_call, start=1):
            output_file = output_dir / f"{prefix}_{i:03d}.json"
            import json
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(messages, f, indent=2, ensure_ascii=False)

    def get_metadata(self) -> dict:
        """Get metadata about induction."""
        return {
            "stage": self.name,
            "schema_path": str(self._schema_path) if self._schema_path else None,
        }
