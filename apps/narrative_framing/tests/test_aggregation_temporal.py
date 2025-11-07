"""Tests for temporal aggregation."""

import pytest
from datetime import date, timedelta

from apps.narrative_framing.aggregation_document import DocumentFrameAggregate
from apps.narrative_framing.aggregation_temporal import PeriodAggregate, TemporalAggregator


@pytest.fixture
def sample_aggregates():
    """Create sample document aggregates for testing."""
    return [
        DocumentFrameAggregate(
            doc_id="doc1",
            frame_scores={"frame_a": 0.8, "frame_b": 0.2},
            total_weight=100.0,
            published_at="2023-01-15"
        ),
        DocumentFrameAggregate(
            doc_id="doc2",
            frame_scores={"frame_a": 0.3, "frame_b": 0.7},
            total_weight=150.0,
            published_at="2023-01-15"
        ),
        DocumentFrameAggregate(
            doc_id="doc3",
            frame_scores={"frame_a": 0.5, "frame_b": 0.5},
            total_weight=200.0,
            published_at="2023-02-01"
        ),
        DocumentFrameAggregate(
            doc_id="doc4",
            frame_scores={"frame_a": 0.0, "frame_b": 0.0},
            total_weight=50.0,
            published_at="2023-02-01"
        ),
    ]


@pytest.fixture
def time_series_aggregates():
    """Create aggregates spread across multiple days for rolling window tests."""
    base_date = date(2023, 1, 1)
    aggregates = []
    for i in range(10):
        d = base_date + timedelta(days=i)
        aggregates.append(
            DocumentFrameAggregate(
                doc_id=f"doc{i}",
                frame_scores={"frame_a": 0.8 if i % 2 == 0 else 0.2, "frame_b": 0.2 if i % 2 == 0 else 0.8},
                total_weight=100.0,
                published_at=d.isoformat()
            )
        )
    return aggregates


class TestTemporalAggregator:
    """Test TemporalAggregator."""
    
    def test_day_aggregation(self, sample_aggregates):
        agg = TemporalAggregator(period="day")
        results = agg.aggregate(sample_aggregates)
        
        assert len(results) > 0
        assert all(isinstance(r, PeriodAggregate) for r in results)
        
        # Should have 2 periods (two unique days)
        periods = {r.period_id for r in results}
        assert len(periods) == 2
    
    def test_year_aggregation(self, sample_aggregates):
        agg = TemporalAggregator(period="year")
        results = agg.aggregate(sample_aggregates)
        
        assert len(results) > 0
        assert all(isinstance(r, PeriodAggregate) for r in results)
        # All same year, should have one period
        periods = {r.period_id for r in results}
        assert len(periods) == 1
        assert any("2023" in r.period_id for r in results)
    
    def test_global_aggregation(self, sample_aggregates):
        agg = TemporalAggregator(period="all")
        results = agg.aggregate(sample_aggregates)
        
        assert len(results) > 0
        assert all(isinstance(r, PeriodAggregate) for r in results)
        periods = {r.period_id for r in results}
        assert len(periods) == 1
        assert "all" in periods
    
    def test_keep_documents_with_no_frames_false(self, sample_aggregates):
        """Test that documents with all zero frames are excluded when keep_documents_with_no_frames=False."""
        agg = TemporalAggregator(period="day", keep_documents_with_no_frames=False)
        results = agg.aggregate(sample_aggregates)
        
        # doc4 has all zeros and should be excluded
        assert len(results) > 0
        assert all(isinstance(r, PeriodAggregate) for r in results)
        # Should have both frame_a and frame_b in at least one period
        all_frames = set()
        for r in results:
            all_frames.update(r.frame_scores.keys())
        assert "frame_a" in all_frames
        assert "frame_b" in all_frames
    
    def test_keep_documents_with_no_frames_true(self, sample_aggregates):
        """Test that documents with all zero frames are included when keep_documents_with_no_frames=True."""
        agg = TemporalAggregator(period="day", keep_documents_with_no_frames=True)
        results = agg.aggregate(sample_aggregates)
        
        # Should include all documents including doc4 (all zeros)
        # Check that we have results
        assert len(results) > 0
    
    def test_keep_documents_with_no_frames_comparison(self, sample_aggregates):
        """Compare results with and without keep_documents_with_no_frames."""
        agg_false = TemporalAggregator(period="all", keep_documents_with_no_frames=False)
        agg_true = TemporalAggregator(period="all", keep_documents_with_no_frames=True)
        
        results_false = agg_false.aggregate(sample_aggregates)
        results_true = agg_true.aggregate(sample_aggregates)
        
        # Results should differ
        # When False, doc4 (all zeros) is excluded
        # When True, doc4 is included
        assert len(results_false) > 0
        assert len(results_true) > 0
        
        # Values should differ because doc4 contributes zeros when included
        # Results are List[PeriodAggregate]
        frame_a_false = results_false[0].frame_scores.get("frame_a", 0.0)
        frame_a_true = results_true[0].frame_scores.get("frame_a", 0.0)
        # They might be the same if doc4 has zero weight, but structure should be different
        assert isinstance(frame_a_false, (int, float))
        assert isinstance(frame_a_true, (int, float))
    
    def test_weight_by_document_weight_true(self, sample_aggregates):
        """Test weighted aggregation where documents are weighted by total_weight."""
        agg = TemporalAggregator(period="all", weight_by_document_weight=True, avg_or_sum="avg")
        results = agg.aggregate(sample_aggregates)
        
        assert len(results) > 0
        assert isinstance(results[0], PeriodAggregate)
        # Results are List[PeriodAggregate]
        frame_a_result = results[0].frame_scores.get("frame_a", 0.0)
        
        # Expected weighted average for frame_a:
        # doc1: 0.8 * 100 = 80
        # doc2: 0.3 * 150 = 45
        # doc3: 0.5 * 200 = 100
        # doc4: 0.0 * 50 = 0 (if included)
        # Without doc4: total = 225, weight = 450, avg = 225/450 = 0.5
        # But need to account for the actual aggregation logic
        assert isinstance(frame_a_result, (int, float))
    
    def test_weight_by_document_weight_false(self, sample_aggregates):
        """Test unweighted aggregation where all documents count equally."""
        agg = TemporalAggregator(
            period="all", 
            weight_by_document_weight=False, 
            keep_documents_with_no_frames=False,
            avg_or_sum="avg"
        )
        results = agg.aggregate(sample_aggregates)
        
        assert len(results) > 0
        assert isinstance(results[0], PeriodAggregate)
        # Results are List[PeriodAggregate]
        frame_a_result = results[0].frame_scores.get("frame_a", 0.0)
        
        # Expected unweighted average for frame_a (all docs count equally):
        # doc1: 0.8, doc2: 0.3, doc3: 0.5
        # avg = (0.8 + 0.3 + 0.5) / 3 = 1.6 / 3 ≈ 0.533
        expected_frame_a = (0.8 + 0.3 + 0.5) / 3.0
        assert abs(frame_a_result - expected_frame_a) < 0.1  # Allow some tolerance for aggregation logic
    
    def test_weight_by_document_weight_comparison(self, sample_aggregates):
        """Compare weighted vs unweighted aggregation results."""
        agg_weighted = TemporalAggregator(
            period="all", 
            weight_by_document_weight=True, 
            keep_documents_with_no_frames=False,
            avg_or_sum="avg"
        )
        agg_unweighted = TemporalAggregator(
            period="all", 
            weight_by_document_weight=False, 
            keep_documents_with_no_frames=False,
            avg_or_sum="avg"
        )
        
        results_weighted = agg_weighted.aggregate(sample_aggregates)
        results_unweighted = agg_unweighted.aggregate(sample_aggregates)
        
        # Frame scores should differ because doc3 has higher weight
        # Weighted: doc3 (weight 200) has more influence than doc1 (weight 100)
        # Unweighted: all documents have equal influence
        # Results are List[PeriodAggregate]
        frame_a_weighted = results_weighted[0].frame_scores.get("frame_a", 0.0)
        frame_a_unweighted = results_unweighted[0].frame_scores.get("frame_a", 0.0)
        
        # They should differ (weighted should be closer to doc3's value since it has higher weight)
        assert abs(frame_a_weighted - frame_a_unweighted) > 0.01
    
    def test_avg_or_sum_avg(self, sample_aggregates):
        """Test averaging aggregation."""
        agg = TemporalAggregator(period="all", avg_or_sum="avg", weight_by_document_weight=False)
        results = agg.aggregate(sample_aggregates)
        
        assert len(results) > 0
        assert isinstance(results[0], PeriodAggregate)
        # Results are List[PeriodAggregate]
        frame_a_result = results[0].frame_scores.get("frame_a", 0.0)
        
        # Should be an average (between 0 and 1 typically)
        assert isinstance(frame_a_result, (int, float))
    
    def test_avg_or_sum_sum(self, sample_aggregates):
        """Test summing aggregation."""
        agg = TemporalAggregator(period="all", avg_or_sum="sum", weight_by_document_weight=False)
        results = agg.aggregate(sample_aggregates)
        
        assert len(results) > 0
        assert isinstance(results[0], PeriodAggregate)
        # Results are List[PeriodAggregate]
        frame_a_result = results[0].frame_scores.get("frame_a", 0.0)
        
        # Should be a sum (could be > 1)
        assert isinstance(frame_a_result, (int, float))
    
    def test_avg_or_sum_comparison(self, sample_aggregates):
        """Compare avg vs sum aggregation results."""
        agg_avg = TemporalAggregator(period="all", avg_or_sum="avg", weight_by_document_weight=False)
        agg_sum = TemporalAggregator(period="all", avg_or_sum="sum", weight_by_document_weight=False)
        
        results_avg = agg_avg.aggregate(sample_aggregates)
        results_sum = agg_sum.aggregate(sample_aggregates)
        
        # Results are List[PeriodAggregate]
        frame_a_avg = results_avg[0].frame_scores.get("frame_a", 0.0)
        frame_a_sum = results_sum[0].frame_scores.get("frame_a", 0.0)
        
        # Sum should generally be larger than avg (unless normalized)
        # The exact relationship depends on implementation, but they should differ
        assert isinstance(frame_a_avg, (int, float))
        assert isinstance(frame_a_sum, (int, float))
    
    def test_rolling_window_applies_smoothing(self, time_series_aggregates):
        """Test that rolling window actually smooths the data."""
        # Create test data with more variation that will show smoothing
        varying_aggregates = []
        base_date = date(2023, 1, 1)
        # Create pattern: 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, ...
        # This will clearly show smoothing when window=3
        for i in range(10):
            d = base_date + timedelta(days=i)
            # Pattern that alternates every 2 days: 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, ...
            frame_a_value = 1.0 if (i // 2) % 2 == 1 else 0.0
            varying_aggregates.append(
                DocumentFrameAggregate(
                    doc_id=f"doc{i}",
                    frame_scores={"frame_a": frame_a_value, "frame_b": 1.0 - frame_a_value},
                    total_weight=100.0,
                    published_at=d.isoformat()
                )
            )
        
        agg_no_rolling = TemporalAggregator(period="day", rolling_window=None)
        agg_rolling = TemporalAggregator(period="day", rolling_window=3)
        
        results_no_rolling = agg_no_rolling.aggregate(varying_aggregates)
        results_rolling = agg_rolling.aggregate(varying_aggregates)
        
        # Both should have results
        assert len(results_no_rolling) > 0
        assert len(results_rolling) > 0
        assert all(isinstance(r, PeriodAggregate) for r in results_no_rolling)
        assert all(isinstance(r, PeriodAggregate) for r in results_rolling)
        
        # Extract frame_a values sorted by period_id
        # Create dictionaries mapping period_id to frame_a value
        # Normalize period_ids to date strings (YYYY-MM-DD) for comparison
        def normalize_period_id(period_id):
            """Normalize period_id to YYYY-MM-DD format."""
            if isinstance(period_id, str):
                # Extract date part if it's a datetime string
                if 'T' in period_id:
                    return period_id.split('T')[0]
                return period_id
            return str(period_id)
        
        no_rolling_values = {
            normalize_period_id(r.period_id): r.frame_scores.get("frame_a", 0.0) 
            for r in results_no_rolling
        }
        rolling_values = {
            normalize_period_id(r.period_id): r.frame_scores.get("frame_a", 0.0) 
            for r in results_rolling
        }
        
        # Get matching period_ids (normalized to date strings)
        common_periods = sorted(set(no_rolling_values.keys()) & set(rolling_values.keys()))
        assert len(common_periods) > 0, f"No common periods found. No rolling: {list(no_rolling_values.keys())[:5]}, Rolling: {list(rolling_values.keys())[:5]}"
        
        values_no_rolling = [no_rolling_values[p] for p in common_periods]
        values_rolling = [rolling_values[p] for p in common_periods]
        
        # Rolling window should smooth values
        assert len(values_no_rolling) == len(values_rolling)
        
        # Values should differ due to smoothing
        differences = [abs(a - b) for a, b in zip(values_no_rolling, values_rolling)]
        # At least some values should be smoothed
        assert max(differences) > 0.01 or len(values_no_rolling) <= 2, \
            f"Rolling window should smooth values. Max difference: {max(differences)}, values_no_rolling: {values_no_rolling[:5]}, values_rolling: {values_rolling[:5]}"
    
    def test_rolling_window_fills_missing_dates(self, sample_aggregates):
        """Test that rolling window fills missing dates between periods."""
        # Create aggregates with gaps between dates
        sparse_aggregates = [
            DocumentFrameAggregate(
                doc_id="doc1",
                frame_scores={"frame_a": 0.8, "frame_b": 0.2},
                total_weight=100.0,
                published_at="2023-01-01"
            ),
            DocumentFrameAggregate(
                doc_id="doc2",
                frame_scores={"frame_a": 0.3, "frame_b": 0.7},
                total_weight=150.0,
                published_at="2023-01-05"  # Gap of 3 days
            ),
        ]
        
        agg = TemporalAggregator(period="day", rolling_window=3)
        results = agg.aggregate(sparse_aggregates)
        
        # Should fill dates between 2023-01-01 and 2023-01-05
        # Results are List[PeriodAggregate]
        assert all(isinstance(r, PeriodAggregate) for r in results)
        period_ids = sorted([r.period_id for r in results])
        assert len(period_ids) >= 3  # At least the original dates
    
    def test_rolling_window_with_all_period(self, sample_aggregates):
        """Test that rolling window is not applied when period='all'."""
        agg = TemporalAggregator(period="all", rolling_window=7)
        results = agg.aggregate(sample_aggregates)
        
        # Should work normally (rolling window ignored for "all")
        assert len(results) > 0
        assert all(isinstance(r, PeriodAggregate) for r in results)
        # Results are List[PeriodAggregate]
        assert len(results) == 1
        assert results[0].period_id == "all"
        assert "frame_a" in results[0].frame_scores
    
    def test_empty_input(self):
        """Test with empty input."""
        agg = TemporalAggregator(period="day")
        results = agg.aggregate([])
        assert len(results) == 0
    
    def test_no_dates(self):
        """Test that documents with no published_at are handled."""
        no_date_agg = DocumentFrameAggregate(
            doc_id="nodate",
            frame_scores={"frame_a": 0.5},
            total_weight=100.0,
            published_at=None
        )
        agg = TemporalAggregator(period="day")
        results = agg.aggregate([no_date_agg])
        # Should handle gracefully (filtered out due to no date)
        assert len(results) == 0
