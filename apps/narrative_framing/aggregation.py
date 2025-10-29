"""Document-level aggregation helpers for narrative framing workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Iterable, List, Optional, Protocol, Sequence

import pandas as pd

from efi_core.utils import normalize_date


@dataclass
class DocumentFrameAggregate:
    """Aggregated frame distribution for a single document."""

    doc_id: str
    frame_scores: Dict[str, float]
    total_weight: float
    published_at: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None
    top_frames: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "frame_scores": self.frame_scores,
            "total_weight": self.total_weight,
            "published_at": self.published_at,
            "title": self.title,
            "url": self.url,
            "top_frames": self.top_frames,
        }


class FrameAggregationStrategy(Protocol):
    """Protocol for converting passage-level predictions to document profiles."""

    def accumulate(
        self,
        doc_id: str,
        passage_text: str,
        probabilities: Dict[str, float],
        *,
        published_at: Optional[str] = None,
        title: Optional[str] = None,
        url: Optional[str] = None,
    ) -> None:
        ...

    def finalize(self) -> Sequence[DocumentFrameAggregate]:
        ...


class WeightedFrameAggregator(FrameAggregationStrategy):
    """Aggregate frame probabilities per document using passage length as weights.

    Supports optional thresholding and per-document normalization of scores.
    """

    def __init__(self, frame_ids: Sequence[str], top_k: int = 3, *, min_threshold: float = 0.0, normalize: bool = True) -> None:
        self._frame_ids = list(frame_ids)
        self._top_k = max(1, top_k)
        self._state: Dict[str, Dict[str, object]] = {}
        self._min_threshold = float(min_threshold)
        self._normalize = bool(normalize)

    def accumulate(
        self,
        doc_id: str,
        passage_text: str,
        probabilities: Dict[str, float],
        *,
        published_at: Optional[str] = None,
        title: Optional[str] = None,
        url: Optional[str] = None,
    ) -> None:
        text = passage_text.strip()
        if not text:
            return

        weight = float(max(len(text.split()), 1))
        state = self._state.setdefault(
            doc_id,
            {
                "frame_sums": {frame_id: 0.0 for frame_id in self._frame_ids},
                "weight": 0.0,
                "published_at": None,
                "title": None,
                "url": None,
            },
        )

        frame_sums: Dict[str, float] = state["frame_sums"]  # type: ignore[assignment]
        for frame_id in self._frame_ids:
            frame_sums[frame_id] += weight * float(probabilities.get(frame_id, 0.0))
        state["weight"] = float(state["weight"]) + weight

        if published_at and not state["published_at"]:
            normalized = normalize_date(published_at)
            state["published_at"] = normalized.isoformat() if normalized else published_at
        if title and not state["title"]:
            state["title"] = title
        if url and not state["url"]:
            state["url"] = url

    def finalize(self) -> Sequence[DocumentFrameAggregate]:
        aggregates: List[DocumentFrameAggregate] = []
        for doc_id, state in self._state.items():
            weight = float(state["weight"])  # type: ignore[arg-type]
            frame_sums: Dict[str, float] = state["frame_sums"]  # type: ignore[assignment]
            if weight <= 0.0:
                frame_scores = {frame_id: 0.0 for frame_id in self._frame_ids}
            else:
                frame_scores = {frame_id: frame_sums[frame_id] / weight for frame_id in self._frame_ids}

            # Apply threshold
            if self._min_threshold > 0.0:
                frame_scores = {fid: (val if val >= self._min_threshold else 0.0) for fid, val in frame_scores.items()}

            # Optional per-document normalization to sum to 1.0 (unless all zeros)
            if self._normalize:
                total = sum(frame_scores.values())
                if total > 0:
                    frame_scores = {fid: (val / total) for fid, val in frame_scores.items()}

            ordered = sorted(frame_scores.items(), key=lambda item: item[1], reverse=True)
            top_frames = [frame_id for frame_id, _ in ordered[: self._top_k]]

            aggregates.append(
                DocumentFrameAggregate(
                    doc_id=doc_id,
                    frame_scores=frame_scores,
                    total_weight=weight,
                    published_at=state.get("published_at"),  # type: ignore[arg-type]
                    title=state.get("title"),
                    url=state.get("url"),
                    top_frames=top_frames,
                )
            )

        return sorted(aggregates, key=lambda agg: (agg.published_at or "", agg.doc_id))


class OccurrenceFrameAggregator(FrameAggregationStrategy):
    """Aggregate binary frame presence per document based on a score threshold.

    For each document, compute length-weighted average probabilities per frame,
    then mark presence as 1 if score >= min_threshold else 0.
    """

    def __init__(self, frame_ids: Sequence[str], *, min_threshold: float = 0.0, top_k: int = 3) -> None:
        self._frame_ids = list(frame_ids)
        self._min_threshold = float(min_threshold)
        self._top_k = max(1, top_k)
        self._state: Dict[str, Dict[str, object]] = {}

    def accumulate(
        self,
        doc_id: str,
        passage_text: str,
        probabilities: Dict[str, float],
        *,
        published_at: Optional[str] = None,
        title: Optional[str] = None,
        url: Optional[str] = None,
    ) -> None:
        text = passage_text.strip()
        if not text:
            return
        weight = float(max(len(text.split()), 1))
        state = self._state.setdefault(
            doc_id,
            {
                "frame_sums": {frame_id: 0.0 for frame_id in self._frame_ids},
                "weight": 0.0,
                "published_at": None,
                "title": None,
                "url": None,
            },
        )
        frame_sums: Dict[str, float] = state["frame_sums"]  # type: ignore[assignment]
        for frame_id in self._frame_ids:
            frame_sums[frame_id] += weight * float(probabilities.get(frame_id, 0.0))
        state["weight"] = float(state["weight"]) + weight
        if published_at and not state["published_at"]:
            normalized = normalize_date(published_at)
            state["published_at"] = normalized.isoformat() if normalized else published_at
        if title and not state["title"]:
            state["title"] = title
        if url and not state["url"]:
            state["url"] = url

    def finalize(self) -> Sequence[DocumentFrameAggregate]:
        aggregates: List[DocumentFrameAggregate] = []
        for doc_id, state in self._state.items():
            weight = float(state["weight"])  # type: ignore[arg-type]
            frame_sums: Dict[str, float] = state["frame_sums"]  # type: ignore[assignment]
            if weight <= 0.0:
                scores = {frame_id: 0.0 for frame_id in self._frame_ids}
            else:
                scores = {frame_id: frame_sums[frame_id] / weight for frame_id in self._frame_ids}
            # Threshold to binary presence
            presence = {fid: (1.0 if val >= self._min_threshold and val > 0.0 else 0.0) for fid, val in scores.items()}
            ordered = sorted(presence.items(), key=lambda item: item[1], reverse=True)
            top_frames = [frame_id for frame_id, v in ordered if v > 0.0][: self._top_k]
            aggregates.append(
                DocumentFrameAggregate(
                    doc_id=doc_id,
                    frame_scores=presence,
                    total_weight=1.0,  # count each document equally
                    published_at=state.get("published_at"),  # type: ignore[arg-type]
                    title=state.get("title"),
                    url=state.get("url"),
                    top_frames=top_frames,
                )
            )
        return sorted(aggregates, key=lambda agg: (agg.published_at or "", agg.doc_id))


def build_weighted_time_series(
    aggregates: Sequence[DocumentFrameAggregate],
) -> pd.DataFrame:
    """Return a daily time series of frame importance weighted by document length."""

    records: List[Dict[str, object]] = []
    for aggregate in aggregates:
        if not aggregate.published_at:
            continue
        normalized = normalize_date(aggregate.published_at)
        if not normalized:
            continue
        day_value = date(normalized.year, normalized.month, normalized.day)
        # Skip future-dated documents to avoid distorting time series
        if day_value > date.today():
            continue
        weight = float(aggregate.total_weight)
        for frame_id, score in aggregate.frame_scores.items():
            records.append(
                {
                    "date": day_value,
                    "frame_id": frame_id,
                    "weighted_score": float(score) * weight,
                    "weight": weight,
                }
            )

    if not records:
        return pd.DataFrame(columns=["date", "frame_id", "avg_score", "share"])

    df = pd.DataFrame.from_records(records)
    grouped = (
        df.groupby(["date", "frame_id"], as_index=False)
        .agg(weighted_score=("weighted_score", "sum"), weight=("weight", "sum"))
    )
    grouped["avg_score"] = grouped["weighted_score"] / grouped["weight"].where(grouped["weight"] > 0, 1.0)

    def _normalize(group: pd.DataFrame) -> pd.DataFrame:
        total = group["avg_score"].sum()
        if total <= 0:
            group["share"] = 0.0
        else:
            group["share"] = group["avg_score"] / total
        return group

    normalized = grouped.groupby("date", group_keys=False).apply(_normalize)
    return normalized.sort_values(["date", "frame_id"]).reset_index(drop=True)


def time_series_to_records(df: pd.DataFrame) -> List[Dict[str, object]]:
    """Convert a time series dataframe to JSON-serializable rows."""

    if df.empty:
        return []
    records: List[Dict[str, object]] = []
    for row in df.itertuples(index=False):
        day = row.date.isoformat() if isinstance(row.date, date) else str(row.date)
        records.append(
            {
                "date": day,
                "frame_id": row.frame_id,
                "avg_score": float(row.avg_score),
                "share": float(row.share),
            }
        )
    return records


def compute_global_frame_share(
    aggregates: Sequence[DocumentFrameAggregate],
) -> Dict[str, float]:
    """Return a length-weighted average frame distribution across the corpus."""

    totals: Dict[str, float] = {}
    total_weight = 0.0
    for aggregate in aggregates:
        weight = float(aggregate.total_weight)
        total_weight += weight
        for frame_id, score in aggregate.frame_scores.items():
            totals[frame_id] = totals.get(frame_id, 0.0) + score * weight

    if total_weight <= 0:
        return {}
    return {frame_id: value / total_weight for frame_id, value in totals.items()}


__all__ = [
    "DocumentFrameAggregate",
    "FrameAggregationStrategy",
    "WeightedFrameAggregator",
    "OccurrenceFrameAggregator",
    "build_weighted_time_series",
    "compute_global_frame_share",
    "time_series_to_records",
]
