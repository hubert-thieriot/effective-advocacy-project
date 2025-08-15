"""
Domain-aware rate limiter for HTTP requests in findings extraction
"""

import time
import logging
from typing import Dict, Optional
from dataclasses import dataclass
from urllib.parse import urlparse
from collections import defaultdict
import threading
import requests

logger = logging.getLogger(__name__)


@dataclass
class DomainRateLimit:
    """Rate limit configuration for a specific domain"""
    requests_per_minute: int = 3
    min_interval: float = 20.0  # Minimum seconds between requests


class DomainRateLimiter:
    """
    Simple rate limiter that tracks and enforces limits per domain
    
    This ensures that we don't overwhelm any single website with too many
    requests in a short time period, which could lead to being blocked.
    """
    
    def __init__(self, default_config: Optional[DomainRateLimit] = None):
        """
        Initialize the domain rate limiter
        
        Args:
            default_config: Default rate limit configuration for new domains
        """
        self.default_config = default_config or DomainRateLimit()
        self.domain_configs: Dict[str, DomainRateLimit] = {}
        self.domain_trackers: Dict[str, Dict] = defaultdict(lambda: {
            'last_request_time': 0.0,
            'request_count': 0,
            'window_start': time.time()
        })
        self.lock = threading.Lock()
        
        logger.info(f"Initialized DomainRateLimiter with default config: {self.default_config}")
    
    def set_domain_config(self, domain: str, config: DomainRateLimit):
        """Set custom rate limit configuration for a specific domain"""
        with self.lock:
            self.domain_configs[domain] = config
            logger.info(f"Set rate limit config for {domain}: {config}")
    
    def wait_if_needed(self, url: str):
        """
        Wait if necessary to respect rate limits for the domain
        
        Args:
            url: URL being requested (domain will be extracted)
        """
        domain = self._extract_domain(url)
        config = self.get_domain_config(domain)
        tracker = self.domain_trackers[domain]
        
        with self.lock:
            current_time = time.time()
            
            # Check if we need to reset the window (1 minute)
            if current_time - tracker['window_start'] >= 60.0:
                tracker['request_count'] = 0
                tracker['window_start'] = current_time
            
            # Check if we've hit the rate limit
            if tracker['request_count'] >= config.requests_per_minute:
                # Wait until the next minute
                wait_time = 60.0 - (current_time - tracker['window_start'])
                if wait_time > 0:
                    logger.info(f"Rate limit reached for {domain}, waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
                    tracker['request_count'] = 0
                    tracker['window_start'] = time.time()
            
            # Check minimum interval between requests
            time_since_last = current_time - tracker['last_request_time']
            if time_since_last < config.min_interval:
                wait_time = config.min_interval - time_since_last
                logger.info(f"Enforcing min interval for {domain}, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
            
            # Update tracking
            tracker['last_request_time'] = time.time()
            tracker['request_count'] += 1
            
            logger.debug(f"Request to {domain} (count: {tracker['request_count']}/{config.requests_per_minute})")
    
    def get_domain_config(self, domain: str) -> DomainRateLimit:
        """Get rate limit configuration for a domain (creates default if not exists)"""
        if domain not in self.domain_configs:
            return self.default_config
        return self.domain_configs[domain]
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Remove port number if present
            if ':' in domain:
                domain = domain.split(':')[0]
            
            return domain
        except Exception as e:
            logger.warning(f"Could not parse domain from URL {url}: {e}")
            return 'unknown'


class RateLimitedSession:
    """
    Wrapper around requests.Session that automatically applies rate limiting
    
    This can be used as a drop-in replacement for requests.Session in
    extractors and other components that make HTTP requests.
    """
    
    def __init__(self, rate_limiter: Optional[DomainRateLimiter] = None, **session_kwargs):
        """
        Initialize rate-limited session
        
        Args:
            rate_limiter: DomainRateLimiter instance, or None for defaults
            **session_kwargs: Arguments to pass to requests.Session
        """
        # Create default rate limiter if none provided
        if rate_limiter is None:
            default_config = DomainRateLimit(requests_per_minute=3, min_interval=20.0)
            rate_limiter = DomainRateLimiter(default_config)
        
        self.rate_limiter = rate_limiter
        self.session = requests.Session(**session_kwargs)
    
    def get(self, url: str, **kwargs):
        """Make GET request with rate limiting"""
        self.rate_limiter.wait_if_needed(url)
        return self.session.get(url, **kwargs)
    
    def post(self, url: str, **kwargs):
        """Make POST request with rate limiting"""
        self.rate_limiter.wait_if_needed(url)
        return self.session.post(url, **kwargs)
    
    def __getattr__(self, name):
        """Delegate other attributes to the underlying session"""
        return getattr(self.session, name)
