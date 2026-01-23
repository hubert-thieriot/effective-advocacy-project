"""Aggregation logic for combined frame + stance analysis."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

from efi_analyser.frames.classifier import DocumentClassifications
from efi_analyser.stance.types import StanceAssignments

from .frame_aggregator import DocumentFilter, FrameAggregator
from .types import DiscourseAggregates, FrameAggregates, StanceAggregates


logger = logging.getLogger(__name__)


def build_stance_aggregates(
    frame_classifications: DocumentClassifications,
    stance_results: StanceAssignments,
    valid_doc_ids: Optional[Set[str]] = None,
) -> StanceAggregates:
    """Build stance aggregates from frame classifications and stance results.

    Args:
        frame_classifications: Document classifications with chunk-level probabilities.
        stance_results: Stance assignments to aggregate.
        valid_doc_ids: If provided, only include chunks from these documents.
    """
    frame_by_chunk: Dict[str, str] = {}
    chunk_to_doc: Dict[str, str] = {}
    frame_totals: Dict[str, int] = {}

    for doc in frame_classifications:
        doc_id = str(doc.payload.get("doc_id", "")).strip()
        if not doc_id:
            continue

        # Skip if document not in valid set
        if valid_doc_ids is not None and doc_id not in valid_doc_ids:
            continue

        chunks = doc.payload.get("chunks", []) if isinstance(doc.payload, dict) else []
        for chunk in chunks or []:
            if not isinstance(chunk, dict):
                continue
            chunk_id = str(chunk.get("chunk_id", "")).strip()
            if not chunk_id:
                continue

            chunk_to_doc[chunk_id] = doc_id

            top_frames = chunk.get("top_frames") or []
            if top_frames:
                primary = str(top_frames[0])
            else:
                probs = chunk.get("probabilities") or {}
                primary = max(probs, key=probs.get) if probs else ""
            if not primary:
                continue
            frame_by_chunk[chunk_id] = primary
            frame_totals[primary] = frame_totals.get(primary, 0) + 1

    by_frame_target: Dict[str, Dict[str, Dict[str, int]]] = {}
    overall_by_target: Dict[str, Dict[str, int]] = {}
    target_totals: Dict[str, int] = {}
    included_count = 0
    excluded_count = 0

    for result in stance_results:
        # Check if chunk belongs to a valid document
        if valid_doc_ids is not None:
            doc_id = chunk_to_doc.get(result.chunk_id)
            if doc_id is None or doc_id not in valid_doc_ids:
                excluded_count += 1
                continue

        frame_id = frame_by_chunk.get(result.chunk_id)
        if not frame_id:
            continue

        included_count += 1
        target = result.target
        label = result.label or "neutral"

        by_frame_target.setdefault(frame_id, {}).setdefault(target, {})
        by_frame_target[frame_id][target][label] = by_frame_target[frame_id][target].get(label, 0) + 1

        overall_by_target.setdefault(target, {})
        overall_by_target[target][label] = overall_by_target[target].get(label, 0) + 1
        target_totals[target] = target_totals.get(target, 0) + 1

    if valid_doc_ids is not None and excluded_count > 0:
        logger.info(f"Stance filtering: {included_count} chunks included, {excluded_count} excluded")

    return StanceAggregates(
        by_frame_target=by_frame_target,
        overall_by_target=overall_by_target,
        frame_totals=frame_totals,
        target_totals=target_totals,
        total_chunks=len(frame_by_chunk),
    )


def build_frame_aggregates(
    frame_classifications: DocumentClassifications,
    frame_ids: Optional[List[str]] = None,
    doc_filter: Optional[DocumentFilter] = None,
) -> FrameAggregates:
    """Build frame aggregates from document classifications.

    Args:
        frame_classifications: Document classifications with chunk-level probabilities.
        frame_ids: Optional list of frame IDs to include.
        doc_filter: Optional document filter to apply.

    Returns:
        FrameAggregates with document-level and group-level DataFrames.
    """
    aggregator = FrameAggregator(
        frame_ids=frame_ids,
        doc_filter=doc_filter,
    )

    # Document-level aggregation using length-weighted method
    documents = aggregator.aggregate_documents(
        frame_classifications,
        method="length_weighted",
    )

    # Group-level aggregations
    by_domain = aggregator.by_domain(documents)
    by_year = aggregator.by_year(documents)
    by_corpus = aggregator.by_corpus(documents)
    global_totals = aggregator.global_totals(documents)

    return FrameAggregates(
        documents=documents,
        by_domain=by_domain,
        by_year=by_year,
        by_corpus=by_corpus,
        global_totals=global_totals,
    )


def build_aggregates(
    frame_classifications: DocumentClassifications,
    stance_results: Optional[StanceAssignments] = None,
    frame_ids: Optional[List[str]] = None,
    doc_filter: Optional[DocumentFilter] = None,
) -> DiscourseAggregates:
    """Build combined frame and stance aggregates.

    Args:
        frame_classifications: Document classifications with chunk-level probabilities.
        stance_results: Optional stance assignments for stance aggregation.
        frame_ids: Optional list of frame IDs to include in frame aggregation.
        doc_filter: Optional document filter to apply to both frame and stance aggregation.

    Returns:
        DiscourseAggregates containing frame and stance aggregates.
    """
    frames = build_frame_aggregates(
        frame_classifications,
        frame_ids=frame_ids,
        doc_filter=doc_filter,
    )

    # Get valid doc_ids from the filtered frame aggregates
    valid_doc_ids: Optional[Set[str]] = None
    if doc_filter and not doc_filter.is_empty() and not frames.documents.empty:
        valid_doc_ids = set(frames.documents["doc_id"].unique())

    stances = None
    if stance_results is not None:
        stances = build_stance_aggregates(
            frame_classifications,
            stance_results,
            valid_doc_ids=valid_doc_ids,
        )

    return DiscourseAggregates(frames=frames, stances=stances)


__all__ = ["build_aggregates", "build_frame_aggregates", "build_stance_aggregates", "DocumentFilter"]
