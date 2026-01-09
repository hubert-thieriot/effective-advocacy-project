# Narrative Framing Aggregation Structure

## Final Structure

```
apps/narrative_framing/
├── aggregation_document.py
│   ├── DocumentFrameAggregate (dataclass)
│   ├── FrameAggregationStrategy (protocol)
│   ├── WeightedFrameAggregator
│   └── OccurrenceFrameAggregator
│
├── aggregation_temporal.py
│   ├── PeriodAggregate (dataclass)
│   ├── TemporalAggregator
│   │   ├── aggregate() → List[PeriodAggregate]
│   │   ├── to_dataframe() → pd.DataFrame (for frame time series)
│   │   └── to_corpus_volume_dataframe() → pd.DataFrame (for corpus volume)
│   └── time_series_to_records() helper
│
└── aggregation_domain.py
    ├── DomainAggregate (dataclass)
    └── DomainAggregator
        └── aggregate() → Tuple[List[Tuple[str, int]], List[Dict]]
```

## TemporalAggregator Features

### Period Types
- `"day"` - Daily aggregation
- `"week"` - Weekly (Monday-start)
- `"month"` - Monthly
- `"year"` - Yearly
- `"all"` - Global (all documents, single aggregate)

### Metrics
- `"avg_score"` - Weighted average scores
- `"share"` - Normalized shares (sum to 1)
- `"count"` - Document counts where frame appears
- `"mentions"` - Total mentions (can be >1 per document)

### Options
- `keep_documents_with_no_frames` - If False, exclude documents where all frame scores are 0
- `normalize_each_period` - Whether to normalize scores within each period
- `weight_by_document` - Whether to weight by document length/weight
- `rolling_window` - Optional rolling window size for smoothing (e.g., 7 or 30)

### Methods
- `aggregate()` - Returns List[PeriodAggregate]
- `to_dataframe()` - Returns DataFrame with [date, frame_id, avg_score, share]
- `to_corpus_volume_dataframe()` - Returns DataFrame with [date, document_count, smoothed_count]

## Key Design Decisions

1. **No duplicate code**: `period="all"` handled as special group, uses same `_aggregate_period` method
2. **Consistent return types**: All temporal aggregation returns `List[PeriodAggregate]`
3. **Proper date filling**: When rolling_window is used, missing dates are filled with 0.0
4. **Auto domain extraction**: DocumentFrameAggregate extracts domain from URL automatically

## Implementation Complete

All aggregation logic is now in place and ready to use!

