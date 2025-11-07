"""Document-level aggregation helpers for narrative framing workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence

from efi_core.utils import normalize_date


def _extract_domain(url: Optional[str]) -> Optional[str]:
    """Extract base domain from URL, ignoring subdomains."""
    if not url:
        return None
    from urllib.parse import urlparse
    parsed = urlparse(url)
    netloc = parsed.netloc or parsed.path
    if not netloc:
        return None
    domain = netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    
    # Extract base domain (ignore subdomains)
    # e.g., kota.tribunnews.com -> tribunnews.com
    parts = domain.split('.')
    if len(parts) >= 2:
        # Handle common two-part TLDs like .co.uk, .co.id, .com.au
        if len(parts) >= 3 and parts[-2] in ('co', 'com', 'org', 'net', 'ac', 'gov'):
            return '.'.join(parts[-3:])
        else:
            return '.'.join(parts[-2:])
    
    return domain or None


@dataclass
class DocumentFrameAggregate:
    """Aggregated frame distribution for a single document."""

    doc_id: str
    frame_scores: Dict[str, float]
    total_weight: float
    published_at: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None
    domain: Optional[str] = field(default=None, init=False)
    top_frames: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Auto-extract domain from URL if not provided."""
        if self.domain is None and self.url:
            object.__setattr__(self, 'domain', _extract_domain(self.url))

    def to_dict(self) -> Dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "frame_scores": self.frame_scores,
            "total_weight": self.total_weight,
            "published_at": self.published_at,
            "title": self.title,
            "url": self.url,
            "domain": self.domain,
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


__all__ = [
    "DocumentFrameAggregate",
    "FrameAggregationStrategy",
    "WeightedFrameAggregator",
    "OccurrenceFrameAggregator",
]

