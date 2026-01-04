# Why Average Frame Score Doesn't Show the Spike (Indonesia Narrative Framing)

## Summary

The **Average Frame Score Over Time** chart in narrative framing reports shows **normalized, averaged intensity** per document, not absolute volume. This is why it doesn't show the spike that word count's 7-day trend shows.

## What's Happening

### Word Count 7-Day Trend
- **Metric**: Raw document count per day mentioning themes
- **Data**: `daily_counts` — absolute counts
- **August 2023 spike**: ~14x increase (July: 300 docs, August: 4,406 docs)
- **Why it spikes**: Directly tracks volume changes

### Narrative Framing Average Frame Score
- **Metric**: Length-weighted average frame strength per document (0-1 scale)
- **Data**: Normalized per-document scores, then averaged daily
- **August 2023**: No spike
- **Why it doesn't spike**: Three layers of averaging/normalization

## The Three Layers That Mask the Spike

### Layer 1: Per-Document Normalization
```python
# From WeightedFrameAggregator (line 124-127)
if self._normalize:
    total = sum(frame_scores.values())
    if total > 0:
        frame_scores = {fid: (val / total) for fid, val in frame_scores.items()}
```
Default: `agg_normalize_weighted=True`

Each document's frame scores sum to 1.0, making scores proportional within the document rather than absolute.

### Layer 2: Per-Document Averaging
```python
# From WeightedFrameAggregator (line 117)
frame_scores = {frame_id: frame_sums[frame_id] / weight for frame_id in self._frame_ids}
```

Scores are averaged by document length, so intensity is averaged across the document.

### Layer 3: Daily Averaging
```python
# From build_weighted_time_series (line 258)
grouped["avg_score"] = grouped["weighted_score"] / grouped["weight"].where(grouped["weight"] > 0, 1.0)
```

Then, per-day averages are taken across all documents that day.

## Why This Hides Volume Spikes

If August saw 15x more documents but similar intensity:
- **July**: 100 docs with natural_factors scores averaging 0.25 → daily avg ≈ 0.25
- **August**: 1,500 docs with natural_factors scores averaging 0.24 → daily avg ≈ 0.24

Volume changes are invisible because intensity is normalized per document.

## What DOES Change in Average Frame Score

Average frame score reflects **qualitative changes in discourse**, e.g.:
- Increased focus on a frame
- Stronger discussion of a frame
- Changed topical balance across frames

It does not reflect pure volume shifts.

## Comparison Table

| Metric | What it Shows | Type | Sensitive To Volume? |
|--------|---------------|------|---------------------|
| **Word Count 7-Day** | Documents per day | Absolute count | ✅ Yes |
| **Word Count 30-Day** | Share of docs per day | Proportional | ❌ No |
| **NF Average Frame Score** | Average intensity | Averaged & normalized | ❌ No |
| **NF Frame Share** | Relative importance | Normalized | ❌ No |
| **NF Occurrence** | Share mentioning frame | Binary presence | ❌ No |

## Code References

```124:127:apps/narrative_framing/aggregation.py
# Per-document normalization
if self._normalize:
    total = sum(frame_scores.values())
    if total > 0:
        frame_scores = {fid: (val / total) for fid, val in frame_scores.items()}
```

```258:apps/narrative_framing/aggregation.py
# Daily averaging
grouped["avg_score"] = grouped["weighted_score"] / grouped["weight"].where(grouped["weight"] > 0, 1.0)
```

## Conclusion

**Average Frame Score** is designed to show **how intensely frames are discussed**, not **how many articles mention them**. It's a measure of discourse quality/intensity, not volume.

To see volume spikes, use:
- **Article Volume Over Time** (new chart in both reports)
- **Word Count 7-Day Trend** (absolute counts by theme)
- **NF Occurrence charts** (count of articles, not averaged scores)

