"""
Tests for URL collectors
"""

import pytest
from unittest.mock import Mock, patch
from ..collectors.crea_collector import CREAPublicationsCollector


class TestCREAPublicationsCollector:
    """Test CREA publications collector"""
    
    def test_crea_collector_initialization(self):
        """Test collector initialization"""
        collector = CREAPublicationsCollector()
        assert collector.name == "crea_publications"
        assert collector.base_url == "https://energyandcleanair.org"
        assert collector.max_sources is None
    
    def test_crea_collector_with_max_sources(self):
        """Test collector with max_sources limit"""
        collector = CREAPublicationsCollector(max_sources=50)
        assert collector.max_sources == 50
    
    def test_crea_collector_finds_oldest_publication(self):
        """Test that the collector can find the oldest publication"""
        collector = CREAPublicationsCollector()
        
        # Mock the session to avoid actual HTTP requests during testing
        with patch.object(collector, 'session') as mock_session:
            # Mock successful response for main publications page
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_response.content = self._get_mock_publications_html()
            mock_session.get.return_value = mock_response
            
            # Collect URLs
            urls = collector.collect_urls()
            
            # Verify the oldest publication is found
            oldest_publication = "https://energyandcleanair.org/publication/air-pollution-in-china-2019/"
            assert oldest_publication in urls, f"Oldest publication {oldest_publication} not found in collected URLs"
            
            # Should find the oldest publication
            assert len(urls) >= 2, f"Expected at least 2 publications, found {len(urls)}"
    
    def test_crea_collector_handles_pagination(self):
        """Test that the collector handles CREA's custom pagination"""
        collector = CREAPublicationsCollector()
        
        # Mock the session
        with patch.object(collector, 'session') as mock_session:
            # Mock main page response
            mock_main_response = Mock()
            mock_main_response.raise_for_status.return_value = None
            mock_main_response.content = self._get_mock_publications_html()
            
            # Mock pagination page response
            mock_page_response = Mock()
            mock_page_response.raise_for_status.return_value = None
            mock_page_response.content = self._get_mock_pagination_html()
            
            # Set up mock to return different responses for different URLs
            def mock_get(url, **kwargs):
                if "query-bf51d214-page=2" in url:
                    return mock_page_response
                return mock_main_response
            
            mock_session.get.side_effect = mock_get
            
            # Collect URLs
            urls = collector.collect_urls()
            
            # Should find publications from both main page and pagination
            assert len(urls) >= 4, f"Expected at least 4 publications with pagination, found {len(urls)}"
    
    def _get_mock_publications_html(self):
        """Get mock HTML content for publications page"""
        return b"""
        <html>
        <body>
            <div class="gb-query-loop-item post-12345 publication">
                <div class="gb-container">
                    <a class="gb-container-link" href="https://energyandcleanair.org/publication/air-pollution-in-china-2019/">
                        Air Pollution in China 2019
                    </a>
                </div>
            </div>
            <div class="gb-query-loop-item post-12346 publication">
                <div class="gb-container">
                    <a class="gb-container-link" href="https://energyandcleanair.org/publication/recent-publication-2025/">
                        Recent Publication 2025
                    </a>
                </div>
            </div>
            <div class="wpgb-pagination">
                <a href="/publications/?cst=&query-bf51d214-page=2">2</a>
                <a href="/publications/?cst=&query-bf51d214-page=3">3</a>
            </div>
        </body>
        </html>
        """
    
    def _get_mock_pagination_html(self):
        """Get mock HTML content for pagination page"""
        return b"""
        <html>
        <body>
            <div class="gb-query-loop-item post-12347 publication">
                <div class="gb-container">
                    <a class="gb-container-link" href="https://energyandcleanair.org/publication/another-old-publication-2018/">
                        Another Old Publication 2018
                    </a>
                </div>
            </div>
            <div class="gb-query-loop-item post-12348 publication">
                <div class="gb-container">
                    <a class="gb-container-link" href="https://energyandcleanair.org/publication/yet-another-publication-2017/">
                        Yet Another Publication 2017
                    </a>
                </div>
            </div>
        </body>
        </html>
        """
    
    def test_crea_collector_generates_pagination_urls(self):
        """Test that the collector generates CREA pagination URLs correctly"""
        collector = CREAPublicationsCollector()
        
        # Test the pagination URL generation
        pagination_urls = collector._generate_crea_pagination_urls("https://energyandcleanair.org")
        
        # Should generate URLs for pages 2-20
        assert len(pagination_urls) == 19
        
        # Check first few URLs
        assert "https://energyandcleanair.org/publications/?cst=&query-bf51d214-page=2" in pagination_urls
        assert "https://energyandcleanair.org/publications/?cst=&query-bf51d214-page=3" in pagination_urls
        assert "https://energyandcleanair.org/publications/?cst=&query-bf51d214-page=20" in pagination_urls
        
        # Should not generate page 1 (already covered by main page)
        assert "https://energyandcleanair.org/publications/?cst=&query-bf51d214-page=1" not in pagination_urls
    
    def test_crea_collector_identifies_publication_urls(self):
        """Test that the collector correctly identifies publication URLs"""
        collector = CREAPublicationsCollector()
        
        # Valid publication URLs
        valid_urls = [
            "https://energyandcleanair.org/publication/air-pollution-in-china-2019/",
            "https://energyandcleanair.org/publication/recent-study-2025/",
            "https://energyandcleanair.org/report/important-analysis/",
            "https://energyandcleanair.org/analysis/key-findings/"
        ]
        
        for url in valid_urls:
            assert collector._is_publication_url(url), f"URL {url} should be identified as publication"
        
        # Invalid URLs
        invalid_urls = [
            "https://energyandcleanair.org/publications/",
            "https://energyandcleanair.org/about-us/",
            "https://otherwebsite.org/publication/test/",
            "https://energyandcleanair.org/news/article/"
        ]
        
        for url in invalid_urls:
            assert not collector._is_publication_url(url), f"URL {url} should not be identified as publication"
