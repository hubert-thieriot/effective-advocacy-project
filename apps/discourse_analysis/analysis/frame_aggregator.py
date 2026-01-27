"""Frame classification aggregation with DataFrame output.

Provides document-level and group-level aggregation for frame classifications.
"""

from __future__ import annotations

import logging
from typing import List, Literal, Optional

import pandas as pd

from efi_analyser.frames.classifier import DocumentClassifications
from efi_analyser.frames.filter import DocumentFilter, extract_domain


logger = logging.getLogger(__name__)

AggregationMethod = Literal["length_weighted", "occurrence"]


class FrameAggregator:
    """Aggregates frame classifications into DataFrames.

    Two-stage aggregation:
    1. Document-level: aggregate chunks into per-document frame weights
    2. Group-level: aggregate documents by domain, year, corpus, etc.
    """

    def __init__(
        self,
        frame_ids: Optional[List[str]] = None,
        occurrence_threshold: float = 0.1,
        doc_filter: Optional[DocumentFilter] = None,
    ):
        """Initialize aggregator.

        Args:
            frame_ids: List of frame IDs to include. If None, uses all frames found.
            occurrence_threshold: Threshold for occurrence method (frame present if prob > threshold).
            doc_filter: Optional document filter to apply during aggregation.
        """
        self.frame_ids = frame_ids
        self.occurrence_threshold = occurrence_threshold
        self.doc_filter = doc_filter or DocumentFilter()

    def aggregate_documents(
        self,
        classifications: DocumentClassifications,
        method: AggregationMethod = "length_weighted",
    ) -> pd.DataFrame:
        """Aggregate chunk-level classifications to document-level frame weights.

        Args:
            classifications: Document classifications with chunk-level probabilities.
            method: Aggregation method.
                - "length_weighted": weight by chunk word count, normalize to sum=1
                - "occurrence": binary 1/0 if frame prob > threshold

        Returns:
            DataFrame with columns: doc_id, date, domain, corpus, frame_id, weight, method
            Long format with one row per (document, frame) pair.
        """
        rows = []
        filter_stats = {
            "total": 0,
            "keyword_filtered": 0,
            "domain_filtered": 0,
            "date_filtered": 0,
            "passed": 0,
        }

        for doc in classifications:
            payload = doc.payload
            doc_id = str(payload.get("doc_id", "")).strip()
            if not doc_id:
                continue

            filter_stats["total"] += 1

            # Extract metadata
            published_at = payload.get("published_at")
            date = self._parse_date(published_at)
            year = date.year if date else None
            url = payload.get("url")
            domain = extract_domain(url)
            corpus = payload.get("corpus")

            # Get chunks for text content
            chunks = payload.get("chunks", [])
            if not isinstance(chunks, list) or not chunks:
                continue

            # Apply filters
            if not self._passes_filters(payload, chunks, domain, date, filter_stats):
                continue

            filter_stats["passed"] += 1

            # Aggregate chunks to document-level frame weights
            frame_weights = self._aggregate_chunks(chunks, method)
            if not frame_weights:
                continue

            # Add rows in long format
            for frame_id, weight in frame_weights.items():
                if self.frame_ids and frame_id not in self.frame_ids:
                    continue
                rows.append({
                    "doc_id": doc_id,
                    "date": date,
                    "year": year,
                    "domain": domain,
                    "corpus": corpus,
                    "frame_id": frame_id,
                    "weight": weight,
                    "method": method,
                })

        # Log filter statistics
        self._log_filter_stats(filter_stats)

        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=[
                "doc_id", "date", "year", "domain", "corpus", "frame_id", "weight", "method"
            ])
        return df

    def _passes_filters(
        self,
        payload: dict,
        chunks: List[dict],
        domain: Optional[str],
        date: Optional[pd.Timestamp],
        stats: dict,
    ) -> bool:
        """Check if document passes all configured filters."""
        f = self.doc_filter
        if f.is_empty():
            return True

        # Keyword filter: document text must contain at least one keyword
        if f.keywords:
            doc_text = self._get_document_text(payload, chunks).lower()
            if not any(kw.lower() in doc_text for kw in f.keywords):
                stats["keyword_filtered"] += 1
                return False

        # Domain whitelist filter
        if f.domain_whitelist:
            if not domain or domain.lower() not in [d.lower() for d in f.domain_whitelist]:
                stats["domain_filtered"] += 1
                return False

        # Date filters
        if date is not None:
            # date_from filter
            if f.date_from:
                try:
                    date_from = pd.to_datetime(f.date_from)
                    if date < date_from:
                        stats["date_filtered"] += 1
                        return False
                except Exception:
                    pass

            # date_windows filter (document must be within at least one window)
            if f.date_windows:
                in_window = False
                for start, end in f.date_windows:
                    try:
                        window_start = pd.to_datetime(start)
                        window_end = pd.to_datetime(end)
                        if window_start <= date <= window_end:
                            in_window = True
                            break
                    except Exception:
                        continue
                if not in_window:
                    stats["date_filtered"] += 1
                    return False
        elif f.date_from or f.date_windows:
            # Document has no date but date filter is configured
            stats["date_filtered"] += 1
            return False

        return True

    def _get_document_text(self, payload: dict, chunks: List[dict]) -> str:
        """Extract full document text from payload or chunks."""
        # Try title + chunks text
        parts = []
        title = payload.get("title", "")
        if title:
            parts.append(str(title))

        for chunk in chunks:
            if isinstance(chunk, dict):
                text = chunk.get("text", "")
                if text:
                    parts.append(str(text))

        return " ".join(parts)

    def _log_filter_stats(self, stats: dict) -> None:
        """Log filtering statistics."""
        total = stats["total"]
        passed = stats["passed"]
        if total == 0:
            return

        filtered = total - passed
        if filtered == 0 and self.doc_filter.is_empty():
            return  # No filtering configured or needed

        logger.info(f"Document filtering: {passed} of {total} documents passed ({100*passed/total:.1f}%)")
        if stats["keyword_filtered"] > 0:
            logger.info(f"  - keyword filter removed: {stats['keyword_filtered']} documents")
        if stats["domain_filtered"] > 0:
            logger.info(f"  - domain filter removed: {stats['domain_filtered']} documents")
        if stats["date_filtered"] > 0:
            logger.info(f"  - date filter removed: {stats['date_filtered']} documents")

    def _aggregate_chunks(
        self,
        chunks: List[dict],
        method: AggregationMethod,
    ) -> dict[str, float]:
        """Aggregate chunk probabilities to document-level frame weights."""
        if method == "length_weighted":
            return self._length_weighted_aggregate(chunks)
        elif method == "occurrence":
            return self._occurrence_aggregate(chunks)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def _length_weighted_aggregate(self, chunks: List[dict]) -> dict[str, float]:
        """Weight each chunk by word count, normalize so weights sum to 1."""
        frame_weighted_sums: dict[str, float] = {}
        total_weight = 0.0

        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            probs = chunk.get("probabilities") or {}
            if not probs:
                continue

            # Word count as weight (fall back to 1 if not available)
            text = chunk.get("text", "")
            word_count = len(text.split()) if text else 1
            weight = max(word_count, 1)
            total_weight += weight

            for frame_id, prob in probs.items():
                frame_weighted_sums[frame_id] = frame_weighted_sums.get(frame_id, 0.0) + prob * weight

        if total_weight == 0 or not frame_weighted_sums:
            return {}

        # Normalize to get average probability per frame
        return {fid: ws / total_weight for fid, ws in frame_weighted_sums.items()}

    def _occurrence_aggregate(self, chunks: List[dict]) -> dict[str, float]:
        """Binary presence: 1.0 if any chunk has prob > threshold, else 0.0."""
        frame_max_probs: dict[str, float] = {}

        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            probs = chunk.get("probabilities") or {}
            for frame_id, prob in probs.items():
                frame_max_probs[frame_id] = max(frame_max_probs.get(frame_id, 0.0), prob)

        return {
            fid: 1.0 if max_prob > self.occurrence_threshold else 0.0
            for fid, max_prob in frame_max_probs.items()
        }

    def _parse_date(self, value) -> Optional[pd.Timestamp]:
        """Parse date from various formats."""
        if value is None:
            return None
        try:
            return pd.to_datetime(value)
        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Group-level aggregation methods
    # -------------------------------------------------------------------------

    def by_domain(
        self,
        doc_df: pd.DataFrame,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """Aggregate document-level weights by domain.

        Args:
            doc_df: Output from aggregate_documents().
            normalize: If True, normalize weights within each domain to sum to 1.

        Returns:
            DataFrame with columns: domain, frame_id, weight, doc_count
        """
        return self._group_aggregate(doc_df, group_col="domain", normalize=normalize)

    def co_occurrence(
        self,
        classifications: DocumentClassifications,
        *,
        threshold: float = 0.3,
    ) -> pd.DataFrame:
        """Compute document-level frame co-occurrence matrix.

        A frame is considered present in a document if any chunk has prob >= threshold.

        Args:
            classifications: Document classifications with chunk-level probabilities.
            threshold: Minimum per-chunk probability to count a frame as present.

        Returns:
            DataFrame with columns: frame_id, co_frame_id, count, cooccurrence_rate
        """
        filter_stats = {
            "total": 0,
            "keyword_filtered": 0,
            "domain_filtered": 0,
            "date_filtered": 0,
            "passed": 0,
        }

        co_counts: dict[tuple[str, str], int] = {}
        observed_frames: set[str] = set()
        total_docs = 0

        for doc in classifications:
            payload = doc.payload
            doc_id = str(payload.get("doc_id", "")).strip()
            if not doc_id:
                continue

            filter_stats["total"] += 1

            published_at = payload.get("published_at")
            date = self._parse_date(published_at)
            url = payload.get("url")
            domain = extract_domain(url)

            chunks = payload.get("chunks", [])
            if not isinstance(chunks, list) or not chunks:
                continue

            if not self._passes_filters(payload, chunks, domain, date, filter_stats):
                continue

            filter_stats["passed"] += 1
            total_docs += 1

            frames_present: set[str] = set()
            for chunk in chunks:
                if not isinstance(chunk, dict):
                    continue
                probs = chunk.get("probabilities") or {}
                if isinstance(probs, dict) and probs:
                    for frame_id, prob in probs.items():
                        try:
                            prob_val = float(prob)
                        except (TypeError, ValueError):
                            continue
                        if prob_val >= threshold:
                            frames_present.add(str(frame_id))
                else:
                    top_frames = chunk.get("top_frames") or []
                    for frame_id in top_frames:
                        if frame_id:
                            frames_present.add(str(frame_id))

            if self.frame_ids:
                frames_present = {fid for fid in frames_present if fid in self.frame_ids}

            if not frames_present:
                continue

            observed_frames.update(frames_present)
            frame_list = sorted(frames_present)
            for left in frame_list:
                for right in frame_list:
                    key = (left, right)
                    co_counts[key] = co_counts.get(key, 0) + 1

        frame_ids = self.frame_ids or sorted(observed_frames)
        if not frame_ids:
            return pd.DataFrame(columns=["frame_id", "co_frame_id", "count", "cooccurrence_rate"])

        rows = []
        denom = total_docs if total_docs > 0 else 1
        for left in frame_ids:
            for right in frame_ids:
                count = co_counts.get((left, right), 0)
                rows.append({
                    "frame_id": left,
                    "co_frame_id": right,
                    "count": count,
                    "cooccurrence_rate": count / denom,
                })

        return pd.DataFrame(rows)

    def by_year(
        self,
        doc_df: pd.DataFrame,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """Aggregate document-level weights by year.

        Args:
            doc_df: Output from aggregate_documents().
            normalize: If True, normalize weights within each year to sum to 1.

        Returns:
            DataFrame with columns: year, frame_id, weight, doc_count
        """
        return self._group_aggregate(doc_df, group_col="year", normalize=normalize)

    def by_domain_year(
        self,
        doc_df: pd.DataFrame,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """Aggregate document-level weights by domain and year.

        Args:
            doc_df: Output from aggregate_documents().
            normalize: If True, normalize weights within each domain-year to sum to 1.

        Returns:
            DataFrame with columns: domain, year, frame_id, weight, doc_count
        """
        return self._group_aggregate_multi(
            doc_df,
            group_cols=["domain", "year"],
            normalize=normalize,
        )

    def by_corpus(
        self,
        doc_df: pd.DataFrame,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """Aggregate document-level weights by corpus.

        Args:
            doc_df: Output from aggregate_documents().
            normalize: If True, normalize weights within each corpus to sum to 1.

        Returns:
            DataFrame with columns: corpus, frame_id, weight, doc_count
        """
        return self._group_aggregate(doc_df, group_col="corpus", normalize=normalize)

    def global_totals(
        self,
        doc_df: pd.DataFrame,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """Aggregate document-level weights globally (all documents).

        Args:
            doc_df: Output from aggregate_documents().
            normalize: If True, normalize weights to sum to 1.

        Returns:
            DataFrame with columns: frame_id, weight, doc_count
        """
        if doc_df.empty:
            return pd.DataFrame(columns=["frame_id", "weight", "doc_count"])

        doc_count = doc_df["doc_id"].nunique()
        grouped = doc_df.groupby("frame_id", as_index=False)["weight"].mean()
        grouped["doc_count"] = doc_count

        if normalize:
            total = grouped["weight"].sum()
            if total > 0:
                grouped["weight"] = grouped["weight"] / total

        return grouped

    def _group_aggregate(
        self,
        doc_df: pd.DataFrame,
        group_col: str,
        normalize: bool = True,
    ) -> pd.DataFrame:
        """Generic group aggregation helper."""
        if doc_df.empty:
            return pd.DataFrame(columns=[group_col, "frame_id", "weight", "doc_count"])

        # Filter out rows with missing group values
        df = doc_df[doc_df[group_col].notna()].copy()
        if df.empty:
            return pd.DataFrame(columns=[group_col, "frame_id", "weight", "doc_count"])

        # Count documents per group
        doc_counts = df.groupby(group_col)["doc_id"].nunique().reset_index()
        doc_counts.columns = [group_col, "doc_count"]

        # Average weight per frame within each group
        grouped = df.groupby([group_col, "frame_id"], as_index=False)["weight"].mean()

        # Normalize within each group if requested
        if normalize:
            group_totals = grouped.groupby(group_col)["weight"].transform("sum")
            grouped["weight"] = grouped["weight"] / group_totals.where(group_totals > 0, 1.0)

        # Merge doc counts
        result = grouped.merge(doc_counts, on=group_col, how="left")
        return result

    def _group_aggregate_multi(
        self,
        doc_df: pd.DataFrame,
        group_cols: List[str],
        normalize: bool = True,
    ) -> pd.DataFrame:
        """Generic multi-column group aggregation helper."""
        if doc_df.empty:
            return pd.DataFrame(columns=[*group_cols, "frame_id", "weight", "doc_count"])

        df = doc_df.dropna(subset=group_cols).copy()
        if df.empty:
            return pd.DataFrame(columns=[*group_cols, "frame_id", "weight", "doc_count"])

        doc_counts = df.groupby(group_cols)["doc_id"].nunique().reset_index(name="doc_count")
        grouped = df.groupby([*group_cols, "frame_id"], as_index=False)["weight"].mean()

        if normalize:
            group_totals = grouped.groupby(group_cols)["weight"].transform("sum")
            grouped["weight"] = grouped["weight"] / group_totals.where(group_totals > 0, 1.0)

        result = grouped.merge(doc_counts, on=group_cols, how="left")
        return result


__all__ = ["FrameAggregator", "AggregationMethod", "DocumentFilter", "extract_domain"]
