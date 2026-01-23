"""
Framing stage for discourse analysis pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from efi_core.pipeline import PipelineStage
from efi_analyser.frames import Framer, FramerPaths, FramingResult

from .base import StageContext


class FramingStage(PipelineStage[StageContext, FramingResult]):
    """Run framing (schema, annotation, training, classification)."""

    def should_run(self, input_data: Optional[StageContext]) -> bool:
        if input_data is None:
            return False
        return bool(input_data.config.framing.enabled)

    def execute(self, input_data: StageContext) -> FramingResult:
        config = input_data.config
        state = input_data.state
        paths = input_data.paths

        if not state.sampler or not state.corpora:
            raise ValueError("Corpora and sampler must be initialized before framing.")

        framer_paths = FramerPaths(
            results_dir=paths.results_dir,
            schema_path=paths.framing_schema_path,
            assignments_path=paths.framing_assignments_path,
            classifier_dir=paths.framing_classifier_dir,
            classifications_dir=paths.framing_classifications_dir,
        )

        prompts = {
            "induction_system": input_data.ind_sys_t,
            "induction_user": input_data.ind_usr_t,
            "application_system": input_data.app_sys_t,
            "application_user": input_data.app_usr_t,
        }

        filter_kwargs = None
        if input_data.filter:
            filter_kwargs = input_data.filter.sampler_kwargs(
                domain_whitelist=config.filter.document.domain_whitelist
            )

        relevance_keywords = config.relevance_keywords
        if isinstance(relevance_keywords, dict):
            relevance_keywords = [kw for kws in relevance_keywords.values() for kw in kws]

        framer = Framer(config, framer_paths, prompts=prompts)
        result = framer.run(
            corpora=state.corpora,
            sampler=state.sampler,
            allow_new_work=input_data.allow_new_work,
            filter_kwargs=filter_kwargs,
            relevance_keywords=relevance_keywords,
            doc_keywords=config.filter.document.keywords,
            domain_whitelist=config.filter.document.domain_whitelist,
            date_from=config.filter.date_from,
            date_windows=config.filter.date_windows,
        )

        return result

    def get_metadata(self) -> dict:
        return {"stage": self.name}
