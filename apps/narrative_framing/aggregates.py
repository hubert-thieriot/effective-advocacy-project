"""Aggregation helpers for narrative framing.

This module centralises loading, building, and saving of aggregation outputs
derived from document-level frame scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Any

import json
import pandas as pd

from apps.narrative_framing.aggregation_document import (
    DocumentFrameAggregate,
    FrameAggregationStrategy,
    WeightedFrameAggregator,
    OccurrenceFrameAggregator,
)
from apps.narrative_framing.aggregation_temporal import (
    TemporalAggregator,
    period_aggregates_to_records,
)
from apps.narrative_framing.aggregation_domain import DomainAggregator
from apps.narrative_framing.filtering import FilterSpec, filter_chunks as nf_filter_chunks
from efi_analyser.frames.classifier import DocumentClassification, DocumentClassifications


def load_document_aggregates(path: Path) -> List[DocumentFrameAggregate]:
    """Load document aggregates from a single JSON file."""
    if not path.exists():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    records: Iterable[Dict[str, object]] = payload

    aggregates: List[DocumentFrameAggregate] = []
    for item in records:
        aggregates.append(
            DocumentFrameAggregate(
                doc_id=item["doc_id"],
                frame_scores={k: float(v) for k, v in item.get("frame_scores", {}).items()},
                total_weight=float(item.get("total_weight", 0.0)),
                published_at=item.get("published_at"),
                title=item.get("title"),
                url=item.get("url"),
                top_frames=list(item.get("top_frames", [])),
            )
        )
    return aggregates


def save_aggregates_json(path: Path, data: object) -> None:
    """Save aggregation data to JSON file."""
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _compute_corpus_aggregates(
    doc_aggs: List[DocumentFrameAggregate],
    *,
    keep_documents_with_no_frames: bool,
    weight_by_document_weight: bool,
) -> List[Dict[str, object]]:
    """Compute per-corpus aggregates from document-level aggregates."""
    if not doc_aggs:
        return []

    df = pd.DataFrame([agg.to_dict() for agg in doc_aggs])
    for col in ["frame_scores", "total_weight", "doc_id"]:
        if col not in df.columns:
            return []

    if "corpus" not in df.columns:
        df["corpus"] = None

    from efi_analyser.frames.identifiers import split_global_doc_id

    df["corpus"] = df.apply(
        lambda r: (
            r["corpus"]
            if pd.notna(r["corpus"]) and r["corpus"]
            else split_global_doc_id(str(r["doc_id"]))[0]
        ),
        axis=1,
    )
    df["corpus"] = df["corpus"].apply(
        lambda x: x if isinstance(x, str) and x.strip() else "default"
    )

    df["has_frames"] = df["frame_scores"].apply(
        lambda x: any(float(v or 0.0) > 0.0 for v in x.values())
    )
    if not keep_documents_with_no_frames:
        df = df[df["has_frames"]]
    if df.empty:
        return []

    if weight_by_document_weight:
        df["weight"] = df["total_weight"]
    else:
        df["weight"] = 1.0

    x = pd.DataFrame(pd.json_normalize(df["frame_scores"]).stack()).reset_index(level=1)
    x.columns = ["frame", "value"]
    df_long = df.drop(columns=["frame_scores"]).join(x)

    df_long["weighted_value"] = df_long["value"] * df_long["weight"]

    mean_weight_per_corpus = (
        df_long.drop_duplicates(subset=["corpus", "doc_id"])
        .groupby("corpus")["weight"]
        .mean()
    )
    grouped = df_long.groupby(["corpus", "frame"])["weighted_value"].mean()
    grouped = grouped / mean_weight_per_corpus
    results_df = grouped.unstack(fill_value=0.0)

    doc_counts = df.groupby("corpus")["doc_id"].nunique().to_dict()

    results: List[Dict[str, object]] = []
    for corpus_name, row in results_df.iterrows():
        frame_scores = {str(col): float(row[col]) for col in row.index}
        results.append(
            {
                "corpus": str(corpus_name),
                "frame_scores": frame_scores,
                "document_count": int(doc_counts.get(corpus_name, 0)),
            }
        )
    return results


def build_all_aggregates(
    aggregates_dir: Path,
    document_aggregates_weighted: List[DocumentFrameAggregate],
    document_aggregates_occurrence: List[DocumentFrameAggregate],
    frame_ids: List[str],
) -> Dict[str, object]:
    """Build all aggregation combinations and save them to files.

    Returns a dictionary with all aggregates for passing to the report.
    """
    aggregates: Dict[str, object] = {}

    print("Building document aggregates...")
    if document_aggregates_weighted:
        save_aggregates_json(
            aggregates_dir / "documents_weighted.json",
            [agg.to_dict() for agg in document_aggregates_weighted],
        )
        aggregates["documents_weighted"] = document_aggregates_weighted
    if document_aggregates_occurrence:
        save_aggregates_json(
            aggregates_dir / "documents_occurrence.json",
            [agg.to_dict() for agg in document_aggregates_occurrence],
        )
        aggregates["documents_occurrence"] = document_aggregates_occurrence

    print("Building temporal aggregates...")
    for keep_zeros in [True, False]:
        key_suffix = "with_zeros" if keep_zeros else "without_zeros"
        agg = TemporalAggregator(
            period="year",
            weight_by_document_weight=True,
            keep_documents_with_no_frames=keep_zeros,
        )
        yearly_result = agg.aggregate(document_aggregates_weighted)
        save_aggregates_json(
            aggregates_dir / f"year_weighted_{key_suffix}.json",
            [
                {
                    "period_id": p.period_id,
                    "frame_scores": p.frame_scores,
                    "document_count": p.document_count,
                }
                for p in yearly_result
            ],
        )
        aggregates[f"year_weighted_{key_suffix}"] = yearly_result

    print("Building domain aggregates...")
    for keep_zeros in [True, False]:
        key_suffix = "with_zeros" if keep_zeros else "without_zeros"

        domain_agg = DomainAggregator(
            keep_documents_with_no_frames=keep_zeros,
            weight_by_document_weight=True,
            avg_or_sum="avg",
        )
        domain_aggregates = domain_agg.aggregate(document_aggregates_weighted)

        domain_frame_summaries = [
            {
                "domain": da.domain,
                "count": da.document_count,
                "shares": da.frame_scores,
            }
            for da in domain_aggregates
        ]

        save_aggregates_json(
            aggregates_dir / f"domain_weighted_{key_suffix}.json",
            domain_frame_summaries,
        )
        aggregates[f"domain_weighted_{key_suffix}"] = domain_frame_summaries

    print("Building corpus aggregates...")
    for keep_zeros in [True, False]:
        key_suffix = "with_zeros" if keep_zeros else "without_zeros"
        corpus_weighted = _compute_corpus_aggregates(
            document_aggregates_weighted,
            keep_documents_with_no_frames=keep_zeros,
            weight_by_document_weight=True,
        )
        if corpus_weighted:
            save_aggregates_json(
                aggregates_dir / f"corpus_weighted_{key_suffix}.json",
                corpus_weighted,
            )
            aggregates[f"corpus_weighted_{key_suffix}"] = corpus_weighted

    print("Building global aggregates...")
    for keep_zeros in [True, False]:
        key_suffix = "with_zeros" if keep_zeros else "without_zeros"
        agg = TemporalAggregator(
            period="all",
            weight_by_document_weight=True,
            keep_documents_with_no_frames=keep_zeros,
        )
        global_result = agg.aggregate(document_aggregates_weighted)
        if global_result:
            global_data = {
                "period_id": global_result[0].period_id,
                "frame_scores": global_result[0].frame_scores,
                "document_count": global_result[0].document_count,
            }
            save_aggregates_json(
                aggregates_dir / f"global_weighted_{key_suffix}.json",
                global_data,
            )
            aggregates[f"global_weighted_{key_suffix}"] = global_result

    print("Building occurrence aggregates...")
    if document_aggregates_occurrence:
        for keep_zeros in [True, False]:
            key_suffix = "with_zeros" if keep_zeros else "without_zeros"
            agg = TemporalAggregator(
                period="all",
                weight_by_document_weight=False,
                keep_documents_with_no_frames=keep_zeros,
            )
            global_result = agg.aggregate(document_aggregates_occurrence)
            if global_result:
                global_data = {
                    "period_id": global_result[0].period_id,
                    "frame_scores": global_result[0].frame_scores,
                    "document_count": global_result[0].document_count,
                }
                save_aggregates_json(
                    aggregates_dir / f"global_occurrence_{key_suffix}.json",
                    global_data,
                )
                aggregates[f"global_occurrence_{key_suffix}"] = global_result

        for keep_zeros in [True, False]:
            key_suffix = "with_zeros" if keep_zeros else "without_zeros"
            agg = TemporalAggregator(
                period="year",
                weight_by_document_weight=False,
                keep_documents_with_no_frames=keep_zeros,
            )
            yearly_result = agg.aggregate(document_aggregates_occurrence)
            save_aggregates_json(
                aggregates_dir / f"year_occurrence_{key_suffix}.json",
                [
                    {
                        "period_id": p.period_id,
                        "frame_scores": p.frame_scores,
                        "document_count": p.document_count,
                    }
                    for p in yearly_result
                ],
            )
            aggregates[f"year_occurrence_{key_suffix}"] = yearly_result

        for keep_zeros in [True, False]:
            key_suffix = "with_zeros" if keep_zeros else "without_zeros"
            corpus_occ = _compute_corpus_aggregates(
                document_aggregates_occurrence,
                keep_documents_with_no_frames=keep_zeros,
                weight_by_document_weight=False,
            )
            if corpus_occ:
                save_aggregates_json(
                    aggregates_dir / f"corpus_occurrence_{key_suffix}.json",
                    corpus_occ,
                )
                aggregates[f"corpus_occurrence_{key_suffix}"] = corpus_occ

    print("Building time series with 30-day rolling average...")
    temporal_agg = TemporalAggregator(
        period="day",
        weight_by_document_weight=True,
        avg_or_sum="avg",
        rolling_window=30,
    )
    frame_timeseries_aggregates = temporal_agg.aggregate(document_aggregates_weighted)
    frame_timeseries_records = period_aggregates_to_records(frame_timeseries_aggregates)
    save_aggregates_json(
        aggregates_dir / "time_series_30day.json",
        frame_timeseries_records,
    )
    aggregates["time_series_30day"] = frame_timeseries_records
    return aggregates


@dataclass
class Aggregates:
    """Container for document-level and derived aggregates."""

    documents_weighted: List[DocumentFrameAggregate] = field(default_factory=list)
    documents_occurrence: List[DocumentFrameAggregate] = field(default_factory=list)
    named: Dict[str, object] = field(default_factory=dict)

    @property
    def all_aggregates(self) -> Dict[str, object]:
        data: Dict[str, object] = dict(self.named)
        if self.documents_weighted:
            data.setdefault("documents_weighted", self.documents_weighted)
        if self.documents_occurrence:
            data.setdefault("documents_occurrence", self.documents_occurrence)
        return data

    def save(self, aggregates_dir: Path) -> None:
        """Persist aggregates to JSON files under ``aggregates_dir``."""
        aggregates_dir.mkdir(parents=True, exist_ok=True)
        if self.documents_weighted:
            save_aggregates_json(
                aggregates_dir / "documents_weighted.json",
                [agg.to_dict() for agg in self.documents_weighted],
            )
        if self.documents_occurrence:
            save_aggregates_json(
                aggregates_dir / "documents_occurrence.json",
                [agg.to_dict() for agg in self.documents_occurrence],
            )
        for key, value in self.named.items():
            if key in {"documents_weighted", "documents_occurrence"}:
                continue
            save_aggregates_json(aggregates_dir / f"{key}.json", value)

    @classmethod
    def load(cls, aggregates_dir: Path) -> Optional["Aggregates"]:
        """Load aggregates from existing JSON files, if available."""
        if not aggregates_dir or not aggregates_dir.exists():
            return None

        weighted_path = aggregates_dir / "documents_weighted.json"
        if not weighted_path.exists():
            return None
        documents_weighted = load_document_aggregates(weighted_path)

        occurrence_path = aggregates_dir / "documents_occurrence.json"
        documents_occurrence: List[DocumentFrameAggregate] = []
        if occurrence_path.exists():
            documents_occurrence = load_document_aggregates(occurrence_path)

        named: Dict[str, object] = {}
        for child in aggregates_dir.glob("*.json"):
            stem = child.stem
            if stem in {"documents_weighted", "documents_occurrence"}:
                continue
            try:
                named[stem] = json.loads(child.read_text(encoding="utf-8"))
            except Exception:
                continue

        return cls(
            documents_weighted=documents_weighted,
            documents_occurrence=documents_occurrence,
            named=named,
        )


@dataclass
class AggregatesBuilder:
    """Build Aggregates from classified documents and configuration."""

    aggregates_dir: Path
    frame_ids: Sequence[str]
    corpus_names: Sequence[str]
    application_top_k: int
    min_threshold_weighted: float
    normalize_weighted: bool
    min_threshold_occurrence: float
    filter_spec: FilterSpec

    def build(self, classifications: DocumentClassifications) -> Aggregates:
        """Compute document-level and derived aggregates, and write them to disk."""
        documents_weighted, documents_occurrence = self._build_document_aggregates(
            classifications
        )

        if len(self.corpus_names) == 1:
            single_corpus = self.corpus_names[0]
            for agg in documents_weighted:
                try:
                    if getattr(agg, "corpus", None) is None:
                        object.__setattr__(agg, "corpus", single_corpus)
                except Exception:
                    pass
            for agg in documents_occurrence:
                try:
                    if getattr(agg, "corpus", None) is None:
                        object.__setattr__(agg, "corpus", single_corpus)
                except Exception:
                    pass

        named = build_all_aggregates(
            self.aggregates_dir,
            documents_weighted,
            documents_occurrence,
            list(self.frame_ids),
        )
        return Aggregates(
            documents_weighted=documents_weighted,
            documents_occurrence=documents_occurrence,
            named=named,
        )

    def _build_document_aggregates(
        self, classifications: DocumentClassifications
    ) -> tuple[List[DocumentFrameAggregate], List[DocumentFrameAggregate]]:
        """Build weighted and occurrence document aggregates from classifications."""
        if not classifications:
            return [], []

        weighted_agg = WeightedFrameAggregator(
            list(self.frame_ids),
            top_k=self.application_top_k,
            min_threshold=self.min_threshold_weighted,
            normalize=self.normalize_weighted,
        )
        occurrence_agg = OccurrenceFrameAggregator(
            list(self.frame_ids),
            min_threshold=self.min_threshold_occurrence,
            top_k=self.application_top_k,
        )

        weighted = list(
            _aggregate_classified_documents(
                documents=classifications,
                aggregator=weighted_agg,
                filter_spec=self.filter_spec,
            )
        )
        occurrence = list(
            _aggregate_classified_documents(
                documents=classifications,
                aggregator=occurrence_agg,
                filter_spec=self.filter_spec,
            )
        )
        return weighted, occurrence


def _aggregate_classified_documents(
    documents: Sequence[DocumentClassification | Dict[str, object]],
    *,
    aggregator: Optional[FrameAggregationStrategy],
    filter_spec: FilterSpec,
) -> Sequence[DocumentFrameAggregate]:
    if not documents:
        return []

    active_aggregator = aggregator
    if active_aggregator is None:
        frame_ids: List[str] = []
        if documents:
            first = documents[0]
            if isinstance(first, DocumentClassification):
                payload = first.payload
            else:
                payload = first
            frame_ids = list(
                next(iter(payload.get("chunks", [])), {})
                .get("probabilities", {})
                .keys()
            )
        active_aggregator = WeightedFrameAggregator(frame_ids)

    spec = filter_spec

    for doc_record in documents:
        if isinstance(doc_record, DocumentClassification):
            payload = doc_record.payload
        else:
            payload = doc_record

        doc_id = str(payload.get("doc_id"))
        chunks = payload.get("chunks", [])
        if not doc_id or not isinstance(chunks, Sequence):
            continue

        filtered_chunks: List[Dict[str, object]] = nf_filter_chunks(chunks, spec)
        if not filtered_chunks:
            continue

        if spec.keywords is not None:
            has_keyword = False
            for chunk in filtered_chunks:
                if not isinstance(chunk, dict):
                    continue
                text = str(chunk.get("text", "")).lower()
                if any(kw in text for kw in spec.keywords):
                    has_keyword = True
                    break
            if not has_keyword:
                continue

        published_at = payload.get("published_at")
        title = payload.get("title")
        url = payload.get("url")
        for chunk in filtered_chunks:
            if not isinstance(chunk, dict):
                continue
            passage_text = str(chunk.get("text", ""))
            if not passage_text.strip():
                continue
            probabilities = chunk.get("probabilities", {})
            if isinstance(probabilities, dict):
                probs = {fid: float(val) for fid, val in probabilities.items()}
            else:
                probs = {}
            active_aggregator.accumulate(
                doc_id=doc_id,
                passage_text=passage_text,
                probabilities=probs,
                published_at=published_at,
                title=title,
                url=url,
            )

    return active_aggregator.finalize()


__all__ = [
    "Aggregates",
    "AggregatesBuilder",
    "build_all_aggregates",
    "load_document_aggregates",
]

