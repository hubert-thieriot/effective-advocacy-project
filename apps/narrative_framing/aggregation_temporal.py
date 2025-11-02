"""Temporal aggregation for narrative framing workflows.

Groups DocumentFrameAggregate by time periods (day/week/month/year) and
computes aggregate metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import pandas as pd

from efi_core.utils import normalize_date
from .aggregation_document import DocumentFrameAggregate


@dataclass
class PeriodAggregate:
    """Aggregated frame metrics for a time period."""

    period_id: str  # "2023-01-01", "2023", etc.
    frame_scores: Dict[str, float]  # frame_id -> aggregated value
    document_count: int


class TemporalAggregator:
    """Aggregate document aggregates by time period."""

    def __init__(
        self,
        period: Literal["day", "week", "month", "year", "all"],
        *,
        keep_documents_with_no_frames: bool = False,
        avg_or_sum: Literal["avg", "sum"] = "avg",
        weight_by_document_weight: bool = True,
        rolling_window: Optional[int] = None,
    ):
        """Initialize temporal aggregator.

        Args:
            period: Time period type for aggregation, or "all" for global aggregation
            keep_documents_with_no_frames: If False, exclude documents where all frame scores are 0
            normalize_each_period: Whether to normalize scores within each period
            weight_by_document: Whether to weight by document length/weight
            avg_or_sum: Whether to average or sum the frame scores
        """
        self.period = period
        self.keep_documents_with_no_frames = keep_documents_with_no_frames
        self.avg_or_sum = avg_or_sum
        self.weight_by_document_weight = weight_by_document_weight
        self.rolling_window = rolling_window

    def aggregate(
        self,
        aggregates: Sequence[DocumentFrameAggregate]
    ) -> List[PeriodAggregate]:
        """Aggregate document aggregates by time period or globally.

        Args:
            aggregates: Document-level aggregates

        Returns:
            List of PeriodAggregate objects, one per period. Each PeriodAggregate contains:
                - period_id: Period identifier (date, year, month, week, or "all")
                - frame_scores: Dictionary mapping frame_id to aggregated value
                - document_count: Number of documents in this period
            
            Returns empty list if no aggregates.
        """
        if not aggregates:
            return []
        
        # Transform to pandas DataFrame
        df = pd.DataFrame([agg.to_dict() for agg in aggregates])
        df = df[['frame_scores', 'published_at', 'total_weight', 'doc_id']]
        
        # Define indicator for documents with frames
        df['has_frames'] = df['frame_scores'].apply(lambda x: any(score > 0.0 for score in x.values()))
        
        # Create period field
        def get_period_id(x):
            normalized = normalize_date(x)
            if normalized is None:
                return None
            d = normalized.date()
            if self.period == "day":
                return d
            elif self.period == "week":
                # Use pandas Timestamp to floor to week (Monday)
                return pd.Timestamp(d).floor('W-MON').date()
            elif self.period == "month":
                # Use pandas Timestamp to floor to first of month
                return pd.Timestamp(d).replace(day=1).date()
            elif self.period == "year":
                return d.year
            elif self.period == "all":
                return "all"
            else:
                raise ValueError(f"Unknown period type: {self.period}")
        
        df['period_id'] = df['published_at'].apply(get_period_id)
        
        # Filter out None period_ids (documents without valid dates)
        df = df[df['period_id'].notna()]
        
        # keep_documents_with_no_frames or not
        if not self.keep_documents_with_no_frames:
            df = df[df['has_frames']]
        
        # weight_by_document or not
        if self.weight_by_document_weight:
            df.loc[:, 'weight'] = df['total_weight']
        else:
            df.loc[:, 'weight'] = 1
        
        # Explode it long format, so each frame score is a separate row
        df = df.reset_index(drop=True) # Important to reset index to avoid mismatched indices
        x = pd.DataFrame(pd.json_normalize(df['frame_scores']).stack()).reset_index(level=1)
        x.columns = ['frame', 'value']
        df_long = df.drop(columns=['frame_scores']).join(x)
        
        # Group by period
        groups = df_long.groupby('period_id')
        
        # Aggregate each period
        def _aggregate_period(group):
            # Multiply the value by the weight
            group['value'] = group['value'] * group['weight']
            # To ensure the weight is distinct on doc_id, sum only the weights for unique doc_id values in this group
            mean_weight = group.drop_duplicates(subset="doc_id")['weight'].mean()
            if self.avg_or_sum == "avg":
                return group.groupby('frame')['value'].mean() / mean_weight
            elif self.avg_or_sum == "sum":
                return group.groupby('frame')['value'].sum() / mean_weight
            else:
                raise ValueError(f"Unknown avg_or_sum type: {self.avg_or_sum}")
            
        results = groups.apply(_aggregate_period, include_groups=False)
        
        # Handle empty results
        if results.empty:
            return []
 
        # Always convert to long format consistently
        # reset_index() converts period_id from index to column
        # melt() converts frame columns to a 'frame' column
        results = results.reset_index().melt(
            id_vars=['period_id'],
            var_name='frame',
            value_name='value'
        )
        
        # Apply rolling window if specified
        if self.rolling_window and self.rolling_window > 1 and self.period != "all":
            # Results is now in long format, pivot to wide for rolling
            results_pivot = results.pivot(index='period_id', columns='frame', values='value')
            
            # Convert period_id to datetime index if needed
            if self.period == "day":
                results_pivot.index = pd.to_datetime(results_pivot.index)
            elif self.period == "week":
                # For week, convert date objects to datetime
                results_pivot.index = pd.to_datetime(results_pivot.index)
            elif self.period == "month":
                # Convert to period index
                results_pivot.index = pd.to_datetime(results_pivot.index)
            elif self.period == "year":
                results_pivot.index = pd.to_datetime(results_pivot.index.astype(str) + '-01-01', format='%Y-%m-%d')
            
            # Fill missing dates with zeros for proper rolling
            if isinstance(results_pivot.index, pd.DatetimeIndex):
                freq_map = {"day": "D", "week": "W-MON", "month": "MS", "year": "YS"}
                freq = freq_map.get(self.period, "D")
                date_range = pd.date_range(results_pivot.index.min(), results_pivot.index.max(), freq=freq)
                results_pivot = results_pivot.reindex(date_range, fill_value=0.0)
                
                # Apply rolling window
                results_pivot = results_pivot.rolling(window=self.rolling_window, min_periods=1).mean()
            
            # Melt back to long format
            results = results_pivot.melt(ignore_index=False, var_name='frame', value_name='value').reset_index()
            # Rename index column back to period_id
            if "index" in results.columns:
                results = results.rename(columns={"index": "period_id"})
            elif "period_id" not in results.columns:
                results = results.rename(columns={results.columns[0]: "period_id"})
        
        # Ensure consistent column order: period_id, frame, value
        results = results[['period_id', 'frame', 'value']]
        
        # Count documents per period (need to get this from original data)
        # Group by period_id and count unique doc_ids
        period_doc_counts = df.groupby('period_id')['doc_id'].nunique().to_dict()
        
        # Convert DataFrame to List[PeriodAggregate]
        period_aggregates: List[PeriodAggregate] = []
        for period_id in results['period_id'].unique():
            period_data = results[results['period_id'] == period_id]
            frame_scores = dict(zip(period_data['frame'], period_data['value']))
            document_count = period_doc_counts.get(period_id, 0)
            
            # Convert period_id to string consistently
            period_id_str = str(period_id)
            if isinstance(period_id, date):
                period_id_str = period_id.isoformat()
            elif hasattr(period_id, 'isoformat'):  # datetime or Timestamp
                # For datetime, use date part only (YYYY-MM-DD) for consistency
                if hasattr(period_id, 'date'):
                    period_id_str = period_id.date().isoformat()
                else:
                    period_id_str = period_id.isoformat()
                # If it's a datetime string, extract just the date part
                if 'T' in period_id_str:
                    period_id_str = period_id_str.split('T')[0]
            
            period_aggregates.append(
                PeriodAggregate(
                    period_id=period_id_str,
                    frame_scores=frame_scores,
                    document_count=document_count
                )
            )
        
        return period_aggregates


def period_aggregates_to_records(period_aggregates: List[PeriodAggregate]) -> List[Dict[str, object]]:
    """Convert a list of PeriodAggregate to JSON-serializable records.
    
    Expands each PeriodAggregate into multiple records (one per frame), suitable
    for time series visualization.
    
    Args:
        period_aggregates: List of PeriodAggregate objects
        
    Returns:
        List of dictionaries with keys: date, frame_id, value
    """
    return [
        {"date": period_agg.period_id, "frame_id": frame_id, "value": float(value)}
        for period_agg in period_aggregates
        for frame_id, value in period_agg.frame_scores.items()
    ]


__all__ = [
    "PeriodAggregate",
    "TemporalAggregator",
    "period_aggregates_to_records",
]

