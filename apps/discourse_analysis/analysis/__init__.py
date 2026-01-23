"""Analysis helpers for discourse analysis."""

from .types import DiscourseAggregates, FrameAggregates, StanceAggregates
from .aggregator import build_aggregates, build_frame_aggregates, build_stance_aggregates, DocumentFilter
from .frame_aggregator import FrameAggregator
from .plots import (
    build_color_map,
    plot_global_distribution,
    plot_by_year,
    plot_by_domain,
    plot_by_corpus,
    plot_document_count_by_domain,
)

__all__ = [
    # Types
    "DiscourseAggregates",
    "FrameAggregates",
    "StanceAggregates",
    # Aggregation
    "DocumentFilter",
    "FrameAggregator",
    "build_aggregates",
    "build_frame_aggregates",
    "build_stance_aggregates",
    # Plotting
    "build_color_map",
    "plot_global_distribution",
    "plot_by_year",
    "plot_by_domain",
    "plot_by_corpus",
    "plot_document_count_by_domain",
]
