"""DNA (Discourse Network Analysis) stage for the discourse analysis pipeline.

Builds the actor-actor network and coalition clustering from claims results.
Both analyses share a single actor set (selected via _select_active_actors) so
that community and coalition groupings are consistent throughout the report.
Coalition K-means labels are used as the authoritative grouping, overwriting
the Louvain community assignments in the network result.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Tuple

from efi_core.pipeline import PipelineStage

from apps.discourse_analysis.analysis.dna_network import (
    _select_active_actors,
    build_dna_network,
    build_coalition_analysis,
    DNANetworkResult,
    CoalitionAnalysisResult,
)
from .base import StageContext

DNAStageOutput = Tuple[Optional[DNANetworkResult], Optional[CoalitionAnalysisResult]]


class DNAStage(PipelineStage[StageContext, DNAStageOutput]):
    """Build DNA network and coalition analysis from claims data.

    Runs after ClaimsAnalysisStage. Both the actor-actor network (Louvain
    communities) and the coalition clustering (K-means) use the same actor
    set, ensuring the report shows a single consistent N-group structure.
    """

    def should_run(self, input_data: Optional[StageContext]) -> bool:
        if input_data is None:
            return False
        cfg = input_data.config.claims_analysis
        return bool(cfg.enabled and cfg.dna.enabled)

    def execute(self, input_data: StageContext) -> DNAStageOutput:
        cfg = input_data.config.claims_analysis
        dna_cfg = cfg.dna
        paths = input_data.paths
        state = input_data.state

        # Use cached results when reload_dna=True, or in report-only mode.
        # reload_dna=False (default) always rebuilds from the claims data.
        use_cache = (
            paths.dna_network_path.exists()
            and paths.dna_coalition_path.exists()
            and (
                dna_cfg.reload_dna
                or input_data.config.regenerate_report_only
                or not input_data.allow_new_work
            )
        )
        if use_cache:
            try:
                dna_result = _load_pickle(paths.dna_network_path)
                coalition_result = _load_pickle(paths.dna_coalition_path)
                self.logger.info(
                    f"Loaded cached DNA: {len(dna_result.nodes)} actors, "
                    f"{coalition_result.best_n} coalitions"
                )
                return dna_result, coalition_result
            except Exception as e:
                self.logger.warning(f"Failed to load cached DNA results: {e}")

        claims_result = state.claims_result
        if not claims_result:
            self.logger.warning("No claims result available for DNA analysis")
            return None, None

        consolidated_ner = (
            state.ner_result.consolidated
            if state.ner_result and state.ner_result.consolidated
            else None
        )

        # Select the shared actor set once — both analyses will use this list.
        active_actors, _ = _select_active_actors(
            claims_result,
            min_statements=dna_cfg.min_statements,
            max_actors=dna_cfg.max_actors,
        )
        self.logger.info(f"DNA: selected {len(active_actors)} actors")

        # Build actor-actor network (edge weights, centrality, Louvain communities).
        dna_result = build_dna_network(
            claims_result=claims_result,
            consolidated_ner=consolidated_ner,
            active_actors=active_actors,
        )

        # Build coalition clustering (K-means on actor × claim position matrix).
        coalition_result = build_coalition_analysis(
            claims_result=claims_result,
            consolidated_ner=consolidated_ner,
            active_actors=active_actors,
            n_clusters_range=tuple(dna_cfg.n_clusters_range),
        )

        # Unify groupings: replace Louvain communities with coalition K-means labels
        # so both the network and coalition sections show the same N groups.
        if dna_result and coalition_result:
            dna_result.communities = dict(coalition_result.best_clustering.labels)
            self.logger.info(
                f"DNA: {len(dna_result.nodes)} actors, "
                f"{coalition_result.best_n} coalitions "
                f"(silhouette: {coalition_result.best_clustering.silhouette_score:.3f})"
            )

        # Persist to cache
        paths.dna_dir.mkdir(parents=True, exist_ok=True)
        try:
            if dna_result:
                _save_pickle(dna_result, paths.dna_network_path)
            if coalition_result:
                _save_pickle(coalition_result, paths.dna_coalition_path)
        except Exception as e:
            self.logger.warning(f"Failed to cache DNA results: {e}")

        return dna_result, coalition_result

    def get_metadata(self) -> dict:
        return {"stage": self.name}


def _save_pickle(obj: object, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _load_pickle(path: Path) -> object:
    with open(path, "rb") as f:
        return pickle.load(f)


__all__ = ["DNAStage"]
