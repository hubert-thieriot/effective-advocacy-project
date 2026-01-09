# Added Article Volume Charts to Reports

## Summary

Added a new "Article Volume Over Time" chart to both **word count** and **narrative framing** reports. This chart shows the **total number of articles per day** with a **7-day rolling average**, filling missing days with zeros.

## Why This Was Needed

The discrepancy between 7-day and 30-day trends highlighted that different metrics were being displayed. The new volume chart explicitly shows **absolute article counts over time**, which helps users:

1. **Understand volume spikes**: See when article counts surge (e.g., Indonesia August 2023)
2. **Compare to trends**: Understand why intensity trends (Average Frame Score) might not show spikes even when volume does
3. **Context for analysis**: See the data availability over time

## Changes Made

### Word Count Report (`apps/word_count/report.py`)
- Added `_render_total_docs_timeseries()` function
- Uses existing `daily_total_docs` data (articles per day regardless of theme)
- New section: "Article Volume Over Time" placed **before** "Daily Trend by Theme"
- 7-day rolling average with zero-filling for missing days

### Narrative Framing Report (`apps/narrative_framing/report.py`)
- Added `_render_plotly_total_docs_timeseries()` function
- Uses `document_aggregates_occurrence` (preferred) or `document_aggregates_weighted` (fallback)
- New section: "Article Volume Over Time" placed **before** "Time Series"
- 7-day rolling average with zero-filling for missing days
- Added explanatory CSS for chart explanations

### Improvements to Existing Explanations
- Changed "averaged daily" → "Grouped by day, then normalized" for clarity
- Added explanatory text boxes above each time series plot

## Technical Details

### Chart Properties
- **Data**: Raw article counts per day
- **Transformation**: 7-day rolling average
- **Zero-filling**: Missing days filled with 0 using `asfreq("D", fill_value=0)`
- **Color**: Dark blue (#1E3D58) to match report theme
- **Y-axis**: "Articles per day (7-day avg)"
- **Y-axis range**: Auto-scaled

### Code References

Word count:
```68:111:apps/word_count/report.py
def _render_total_docs_timeseries(daily_total_docs: Dict[str, int]) -> str:
    """Render total documents per day with 7-day rolling average."""
    ...
    df = df.asfreq("D", fill_value=0)
    df["smooth"] = df["count"].rolling(window=7, min_periods=1).mean()
    ...
```

Narrative framing:
```1404:1461:apps/narrative_framing/report.py
def _render_plotly_total_docs_timeseries(
    aggregates: Optional[Sequence[DocumentFrameAggregate]],
) -> str:
    """Render total documents per day with 7-day rolling average."""
    ...
    df = df.set_index("date").asfreq("D", fill_value=0)
    df["smooth"] = df["count"].rolling(window=7, min_periods=1).mean()
    ...
```

## Testing

Both reports should now display the volume chart when regenerated. For Indonesia data, expect to see the August 2023 spike clearly visible in the volume chart but not in the Average Frame Score chart (as expected).









