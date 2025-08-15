"""Tests for the domain-aware rate limiter"""

import time
import pytest
from unittest.mock import patch

from ..rate_limiter import DomainRateLimiter, DomainRateLimit, RateLimitedSession


class TestDomainRateLimiter:
    """Test the DomainRateLimiter class"""
    
    def test_init_with_default_config(self):
        """Test initialization with default configuration"""
        rate_limiter = DomainRateLimiter()
        assert rate_limiter.default_config.requests_per_minute == 3
        assert rate_limiter.default_config.min_interval == 20.0
    
    def test_init_with_custom_config(self):
        """Test initialization with custom configuration"""
        custom_config = DomainRateLimit(requests_per_minute=5, min_interval=10.0)
        rate_limiter = DomainRateLimiter(custom_config)
        assert rate_limiter.default_config.requests_per_minute == 5
        assert rate_limiter.default_config.min_interval == 10.0
    
    def test_set_domain_config(self):
        """Test setting custom configuration for a domain"""
        rate_limiter = DomainRateLimiter()
        custom_config = DomainRateLimit(requests_per_minute=1, min_interval=60.0)
        
        rate_limiter.set_domain_config("crea.org", custom_config)
        config = rate_limiter.get_domain_config("crea.org")
        
        assert config.requests_per_minute == 1
        assert config.min_interval == 60.0
    
    def test_get_domain_config_default(self):
        """Test getting default configuration for unknown domain"""
        rate_limiter = DomainRateLimiter()
        config = rate_limiter.get_domain_config("unknown.com")
        
        assert config.requests_per_minute == 3
        assert config.min_interval == 20.0
    
    def test_extract_domain(self):
        """Test domain extraction from URLs"""
        rate_limiter = DomainRateLimiter()
        
        assert rate_limiter._extract_domain("https://example.com/page") == "example.com"
        assert rate_limiter._extract_domain("http://sub.example.com:8080/path") == "sub.example.com"
        assert rate_limiter._extract_domain("https://example.com") == "example.com"
    
    def test_rate_limiting_basic(self):
        """Test basic rate limiting functionality"""
        # Use very short intervals for testing
        config = DomainRateLimit(requests_per_minute=10, min_interval=0.1)
        rate_limiter = DomainRateLimiter(config)
        
        # First request should not wait
        start_time = time.time()
        rate_limiter.wait_if_needed("https://example.com/page1")
        first_request_time = time.time() - start_time
        
        # Second request should wait for min interval
        start_time = time.time()
        rate_limiter.wait_if_needed("https://example.com/page2")
        second_request_time = time.time() - start_time
        
        assert first_request_time < 0.1  # Should be fast
        assert second_request_time >= 0.09  # Should wait ~0.1s
    
    def test_rate_limiting_per_minute(self):
        """Test requests per minute limit"""
        # Use very short intervals for testing
        config = DomainRateLimit(requests_per_minute=2, min_interval=0.05)
        rate_limiter = DomainRateLimiter(config)
        
        # First two requests should be fast
        rate_limiter.wait_if_needed("https://example.com/page1")
        rate_limiter.wait_if_needed("https://example.com/page2")
        
        # Third request should trigger rate limit
        start_time = time.time()
        rate_limiter.wait_if_needed("https://example.com/page3")
        third_request_time = time.time() - start_time
        
        assert third_request_time >= 0.9  # Should wait for minute reset (but much shorter in test)


class TestRateLimitedSession:
    """Test the RateLimitedSession class"""
    
    @patch('requests.Session')
    def test_session_creation(self, mock_session_class):
        """Test that RateLimitedSession wraps requests.Session correctly"""
        mock_session = mock_session_class.return_value
        mock_session.headers = {'User-Agent': 'test'}
        
        rate_limiter = DomainRateLimiter()
        session = RateLimitedSession(rate_limiter)
        
        # Should delegate to underlying session
        assert session.headers == {'User-Agent': 'test'}
    
    @patch('requests.Session')
    def test_get_with_rate_limiting(self, mock_session_class):
        """Test that GET requests apply rate limiting"""
        mock_session = mock_session_class.return_value
        mock_response = mock_session.get.return_value
        
        rate_limiter = DomainRateLimiter()
        session = RateLimitedSession(rate_limiter)
        
        # Mock the rate limiter to avoid actual delays in tests
        with patch.object(rate_limiter, 'wait_if_needed') as mock_wait:
            response = session.get("https://example.com/page")
            
            mock_wait.assert_called_once_with("https://example.com/page")
            mock_session.get.assert_called_once_with("https://example.com/page")
            assert response == mock_response
