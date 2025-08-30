#!/usr/bin/env python3
"""
Integration tests for URL blacklist functionality in the corpus building pipeline
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from efi_corpus.builders.base import BaseCorpusBuilder
from efi_corpus.types import BuilderParams, DiscoveryItem


class MockCorpusBuilder(BaseCorpusBuilder):
    """Mock implementation for integration testing"""
    
    def discover(self, params):
        """Return test discovery items"""
        return [
            DiscoveryItem(
                url="https://example.com/weather-today-aqi-and-rain-forecast",
                canonical_url="https://example.com/weather-today-aqi-and-rain-forecast",
                title="Weather Report",
                published_at="2024-01-01"
            ),
            DiscoveryItem(
                url="https://example.com/air-quality-report",
                canonical_url="https://example.com/air-quality-report",
                title="Air Quality Report", 
                published_at="2024-01-01"
            ),
            DiscoveryItem(
                url="https://livemint.com/article",
                canonical_url="https://livemint.com/article",
                title="Article from Livemint",
                published_at="2024-01-01"
            ),
            DiscoveryItem(
                url="https://example.com/pm25-data",
                canonical_url="https://example.com/pm25-data",
                title="PM2.5 Data",
                published_at="2024-01-01"
            )
        ]
    
    def fetch_raw(self, url, stable_id):
        """Mock fetch_raw method"""
        return b"mock content", {"mime": "text/html"}, "html"
    
    def parse_text(self, raw_bytes, raw_ext, url):
        """Mock parse_text method"""
        return {
            "text": "This is mock text content for testing purposes with sufficient length to pass quality gates. " * 20,  # Repeat to get >400 chars
            "title": "Mock Title",
            "published_at": "2024-01-01"
        }


class TestURLBlacklistIntegration:
    """Test URL blacklist integration in the pipeline"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.temp_dir = Path("temp_test_corpus_integration")
        self.temp_dir.mkdir(exist_ok=True)
        
        # Mock the fetcher and corpus handle
        with patch('efi_corpus.builders.base.Fetcher'), \
             patch('efi_corpus.builders.base.CorpusHandle'):
            self.builder = MockCorpusBuilder(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_url_blacklist_in_pipeline(self):
        """Test that URL blacklist filtering works in the actual pipeline"""
        # Create params with URL blacklist
        params = BuilderParams(
            keywords=["air quality"],
            date_from="2024-01-01",
            date_to="2024-01-31",
            extra={
                "url_blacklist": ["weather-today-aqi-and-rain-forecast"],
                "domain_blacklist": ["livemint.com"]
            }
        )
        
        # Mock the corpus methods to avoid actual processing
        with patch.object(self.builder.corpus, 'has_doc', return_value=False), \
             patch.object(self.builder.corpus, 'add_doc'):
            
            # Run the pipeline and capture output
            from io import StringIO
            import sys
            
            # Capture stdout to verify debug messages
            captured_output = StringIO()
            sys.stdout = captured_output
            
            try:
                result = self.builder.run(params=params)
                output = captured_output.getvalue()
                
                # Verify that the blacklisted items were filtered out
                # Check debug output shows correct filtering
                assert "Filtered out 1 items from blacklisted domains: ['livemint.com']" in output
                assert "Filtered out 1 items with blacklisted URL patterns: ['weather-today-aqi-and-rain-forecast']" in output
                
                # Verify that only 2 items were discovered (after filtering)
                assert "Discovered: 2" in output
                
                # Verify that the correct URLs were processed
                assert "Processing URL: https://example.com/air-quality-report" in output
                assert "Processing URL: https://example.com/pm25-data" in output
                
                # Verify blacklisted URLs were NOT processed
                assert "Processing URL: https://example.com/weather-today-aqi-and-rain-forecast" not in output
                assert "Processing URL: https://livemint.com/article" not in output
                
            finally:
                sys.stdout = sys.__stdout__
    
    def test_domain_blacklist_in_pipeline(self):
        """Test that domain blacklist filtering works in the actual pipeline"""
        # Create params with only domain blacklist
        params = BuilderParams(
            keywords=["air quality"],
            date_from="2024-01-01",
            date_to="2024-01-31",
            extra={
                "domain_blacklist": ["livemint.com"]
            }
        )
        
        # Mock the corpus methods
        with patch.object(self.builder.corpus, 'has_doc', return_value=False), \
             patch.object(self.builder.corpus, 'add_doc'):
            
            # Run the pipeline and capture output
            from io import StringIO
            import sys
            
            # Capture stdout to verify debug messages
            captured_output = StringIO()
            sys.stdout = captured_output
            
            try:
                result = self.builder.run(params=params)
                output = captured_output.getvalue()
                
                # Verify that the blacklisted domain was filtered out
                assert "Filtered out 1 items from blacklisted domains: ['livemint.com']" in output
                
                # Verify that only 3 items were discovered (after filtering)
                assert "Discovered: 3" in output
                
                # Verify that the correct URLs were processed
                assert "Processing URL: https://example.com/weather-today-aqi-and-rain-forecast" in output
                assert "Processing URL: https://example.com/air-quality-report" in output
                assert "Processing URL: https://example.com/pm25-data" in output
                
                # Verify blacklisted domain was NOT processed
                assert "Processing URL: https://livemint.com/article" not in output
                
            finally:
                sys.stdout = sys.__stdout__
    
    def test_no_blacklist_in_pipeline(self):
        """Test that pipeline works normally when no blacklists are configured"""
        # Create params with no blacklists
        params = BuilderParams(
            keywords=["air quality"],
            date_from="2024-01-01",
            date_to="2024-01-31"
        )
        
        # Mock the corpus methods
        with patch.object(self.builder.corpus, 'has_doc', return_value=False), \
             patch.object(self.builder.corpus, 'add_doc'):
            
            # Run the pipeline and capture output
            from io import StringIO
            import sys
            
            # Capture stdout to verify debug messages
            captured_output = StringIO()
            sys.stdout = captured_output
            
            try:
                result = self.builder.run(params=params)
                output = captured_output.getvalue()
                
                # Verify that no filtering occurred
                assert "No domain blacklist configured" in output
                assert "No URL blacklist configured" in output
                
                # Verify that all 4 items were discovered
                assert "Discovered: 4" in output
                
                # Verify that all URLs were processed
                assert "Processing URL: https://example.com/weather-today-aqi-and-rain-forecast" in output
                assert "Processing URL: https://example.com/air-quality-report" in output
                assert "Processing URL: https://livemint.com/article" in output
                assert "Processing URL: https://example.com/pm25-data" in output
                
            finally:
                sys.stdout = sys.__stdout__


if __name__ == "__main__":
    pytest.main([__file__])
