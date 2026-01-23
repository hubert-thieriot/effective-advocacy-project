"""Domain aggregation for narrative framing workflows.

Groups DocumentFrameAggregate by media domain/source and computes
aggregate metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence

import pandas as pd

from .aggregation_document import DocumentFrameAggregate


@dataclass
class DomainAggregate:
    """Aggregated frame metrics for a media domain."""

    domain: str
    frame_scores: Dict[str, float]  # frame_id -> aggregated value
    document_count: int


class DomainAggregator:
    """Aggregate document aggregates by media domain."""

    def __init__(
        self,
        *,
        keep_documents_with_no_frames: bool = False,
        avg_or_sum: Literal["avg", "sum"] = "avg",
        weight_by_document_weight: bool = True,
        weight_by_relevance: bool = True,
    ):
        """Initialize domain aggregator.

        Args:
            keep_documents_with_no_frames: If False, exclude documents where all frame scores are 0
            avg_or_sum: Whether to average or sum the frame scores
            weight_by_document_weight: Whether to weight by document length/weight
            weight_by_relevance: Whether to weight by document relevance (sum of raw frame probs).
                When True, irrelevant documents (low raw probability sums) contribute less.
        """
        self.keep_documents_with_no_frames = keep_documents_with_no_frames
        self.avg_or_sum = avg_or_sum
        self.weight_by_document_weight = weight_by_document_weight
        self.weight_by_relevance = weight_by_relevance

    def aggregate(
        self,
        aggregates: Sequence[DocumentFrameAggregate],
    ) -> List[DomainAggregate]:
        """Aggregate document aggregates by domain.

        Args:
            aggregates: Document-level aggregates

        Returns:
            List of DomainAggregate objects, one per domain. Each DomainAggregate contains:
                - domain: Domain identifier
                - frame_scores: Dictionary mapping frame_id to aggregated value
                - document_count: Number of documents in this domain
            
            Returns empty list if no aggregates.
        """
        if not aggregates:
            return []
        
        # Transform to pandas DataFrame
        df = pd.DataFrame([agg.to_dict() for agg in aggregates])
        # Include relevance_weight if available (default to 1.0 for backwards compatibility)
        cols = ['frame_scores', 'total_weight', 'doc_id', 'domain']
        if 'relevance_weight' in df.columns:
            cols.append('relevance_weight')
        else:
            df['relevance_weight'] = 1.0
        df = df[cols]
        
        # Filter out documents without domain
        df = df[df['domain'].notna()]
        
        if df.empty:
            return []
        
        # Define indicator for documents with frames
        df['has_frames'] = df['frame_scores'].apply(lambda x: any(score > 0.0 for score in x.values()))
        
        # keep_documents_with_no_frames or not
        if not self.keep_documents_with_no_frames:
            df = df[df['has_frames']]
        
        if df.empty:
            return []

        # Compute combined weight: document_weight * relevance_weight (if enabled)
        if self.weight_by_document_weight:
            base_weight = df['total_weight']
        else:
            base_weight = pd.Series(1.0, index=df.index)

        if self.weight_by_relevance:
            # Multiply by relevance_weight so irrelevant docs contribute less
            df['weight'] = base_weight * df['relevance_weight']
        else:
            df['weight'] = base_weight
        
        # Explode it long format, so each frame score is a separate row
        x = pd.DataFrame(pd.json_normalize(df['frame_scores']).stack()).reset_index(level=1)
        x.columns = ['frame', 'value']
        df_long = df.drop(columns=['frame_scores']).join(x)
        
        # Group by domain
        groups = df_long.groupby('domain')
        
        # Aggregate each domain
        def _aggregate_domain(group):
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
            
        results = groups.apply(_aggregate_domain, include_groups=False)
        
        # Handle empty results
        if results.empty:
            return []
        
        # Convert Series to DataFrame if needed (happens when domains have different frame sets)
        if isinstance(results, pd.Series):
            # Series with MultiIndex - unstack to convert to DataFrame
            results = results.unstack(fill_value=0.0)
        
        # Always convert to long format consistently
        # reset_index() converts domain from index to column
        # melt() converts frame columns to a 'frame' column
        results = results.reset_index().melt(
            id_vars=['domain'],
            var_name='frame',
            value_name='value'
        )
        
        # Calculate document counts per domain (use filtered df)
        domain_doc_counts = df.groupby('domain')['doc_id'].nunique().to_dict()
        
        # Convert DataFrame to List[DomainAggregate]
        domain_aggregates: List[DomainAggregate] = []
        for domain in results['domain'].unique():
            domain_data = results[results['domain'] == domain]
            frame_scores = dict(zip(domain_data['frame'], domain_data['value']))
            document_count = domain_doc_counts.get(domain, 0)
            
            domain_aggregates.append(
                DomainAggregate(
                    domain=str(domain),
                    frame_scores=frame_scores,
                    document_count=document_count
                )
            )
        
        return domain_aggregates


__all__ = [
    "DomainAggregate",
    "DomainAggregator",
]

