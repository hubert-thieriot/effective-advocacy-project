"""
Discourse analysis pipeline wiring framing, stance detection, analysis, and reporting.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from efi_core.pipeline import Pipeline, PipelineStage

from apps.narrative_framing.filtering import Filter

from .config_models import DiscourseAnalysisConfig
from .stages import (
    CorpusLoadingStage,
    FramingStage,
    NERStage,
    ClaimsAnalysisStage,
    DNAStage,
    AnalysisStage,
    ReportStage,
    StageContext,
    WorkflowPaths,
    WorkflowState,
)


class DiscourseAnalysisPipeline(Pipeline):
    """Multi-stage pipeline for discourse analysis."""

    def __init__(self, config: DiscourseAnalysisConfig, output_dir: Path):
        super().__init__(config, output_dir)
        self.paths = self._init_paths(output_dir)
        self.state = WorkflowState()
        self.corpus_names = list(config.iter_corpus_names())
        self.filter = Filter(
            exclude_regex=config.filter.chunk.exclude_regex,
            exclude_min_hits=config.filter.chunk.exclude_min_hits,
            trim_after_markers=config.filter.chunk.trim_after_markers,
            keywords=config.filter.chunk.keywords,
        )
        self._load_prompts()

    def _init_paths(self, output_dir: Path) -> WorkflowPaths:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        framing_dir = output_dir / "framing"
        ner_dir = output_dir / "ner"
        claims_dir = output_dir / "claims"
        dna_dir = output_dir / "dna"
        report_dir = output_dir / "report"
        plots_dir = report_dir / "plots"

        return WorkflowPaths(
            results_dir=output_dir,
            framing_dir=framing_dir,
            framing_schema_path=framing_dir / "schema.json",
            framing_assignments_path=framing_dir / "assignments.json",
            framing_classifier_dir=framing_dir / "classifier",
            framing_classifications_dir=framing_dir / "classifications",
            ner_dir=ner_dir,
            ner_entities_path=ner_dir / "entities.json",
            ner_consolidated_path=ner_dir / "entities_consolidated.json",
            claims_dir=claims_dir,
            statements_path=claims_dir / "statements.json",
            claims_schema_path=claims_dir / "claims.json",
            agreements_path=claims_dir / "agreements.json",
            dna_dir=dna_dir,
            dna_network_path=dna_dir / "network.pkl",
            dna_coalition_path=dna_dir / "coalition.pkl",
            aggregates_dir=output_dir / "analysis",
            report_dir=report_dir,
            plots_dir=plots_dir,
            html=report_dir / "discourse_report.html",
        )

    def _load_prompts(self) -> None:
        from apps.narrative_framing.run import _resolve_default_prompt_paths, _read_text_or_fail

        try:
            prompt_paths = _resolve_default_prompt_paths()
            self.ind_sys_t = _read_text_or_fail(prompt_paths["induction_system"])
            self.ind_usr_t = _read_text_or_fail(prompt_paths["induction_user"])
            self.app_sys_t = _read_text_or_fail(prompt_paths["application_system"])
            self.app_usr_t = _read_text_or_fail(prompt_paths["application_user"])
        except Exception as exc:
            self.logger.warning(f"Failed to load prompt templates: {exc}")
            self.ind_sys_t = "{system_message}"
            self.ind_usr_t = "{user_message}"
            self.app_sys_t = "{system_message}"
            self.app_usr_t = "{user_message}"

    def build_stages(self) -> List[PipelineStage]:
        context = StageContext(
            config=self.config,
            paths=self.paths,
            state=self.state,
            corpus_names=self.corpus_names,
            allow_new_work=not self.config.regenerate_report_only,
            filter=self.filter,
            filter_spec=self.filter.to_spec(),
            ind_sys_t=self.ind_sys_t,
            ind_usr_t=self.ind_usr_t,
            app_sys_t=self.app_sys_t,
            app_usr_t=self.app_usr_t,
        )

        stages = [
            CorpusLoadingStage("corpus_loading", self.output_dir),
            FramingStage("framing", self.output_dir),
            NERStage("ner", self.output_dir),
            ClaimsAnalysisStage("claims_analysis", self.output_dir),
            DNAStage("dna", self.output_dir),
            AnalysisStage("analysis", self.output_dir),
            ReportStage("report", self.output_dir),
        ]
        self._context = context
        return stages

    def run(self):
        self.logger.info("=" * 60)
        self.logger.info("DISCOURSE ANALYSIS PIPELINE")
        self.logger.info("=" * 60)
        self.logger.info(f"Config: {self.config.corpus}")
        self.logger.info(f"Output: {self.output_dir}")
        self.logger.info("=" * 60)

        self.stages = self.build_stages()
        results = {}

        for stage in self.stages:
            self.logger.info(f"Pipeline stage: {stage.name}")
            result = stage.run(self._context)
            results[stage.name] = result

            if result.error:
                self.logger.error(f"Pipeline stopped due to error in {stage.name}")
                break

            if result.data is not None:
                self._update_state_from_stage(stage.name, result.data)

        return results

    def _update_state_from_stage(self, stage_name: str, data: object) -> None:
        if stage_name == "corpus_loading":
            self.state.corpora = data
        elif stage_name == "framing":
            self.state.frame_schema = data.schema
            self.state.frame_assignments = data.assignments
            self.state.frame_classifications = data.classifications
            self.state.frame_annotation_candidates = data.annotation_candidates
        elif stage_name == "ner":
            self.state.ner_result = data
        elif stage_name == "claims_analysis":
            self.state.claims_result = data
        elif stage_name == "dna":
            dna_result, coalition_result = data
            self.state.dna_result = dna_result
            self.state.coalition_result = coalition_result
        elif stage_name == "analysis":
            self.state.aggregates = data


__all__ = ["DiscourseAnalysisPipeline"]
