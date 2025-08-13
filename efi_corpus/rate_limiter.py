"""
Rate limiter for API requests
"""

import time
from typing import Optional
from dataclasses import dataclass


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    requests_per_minute: int = 3  # MediaCloud default
    min_interval: float = 20.0    # Minimum seconds between requests
    burst_size: int = 1           # Allow burst of requests
    enabled: bool = True          # Can be disabled for testing


class RateLimiter:
    """Rate limiter that respects API limits and can be disabled for testing"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.last_request_time = 0.0
        self.request_count = 0
        self.window_start = time.time()
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits"""
        if not self.config.enabled:
            return
        
        current_time = time.time()
        
        # Check if we need to reset the window
        if current_time - self.window_start >= 60.0:
            self.request_count = 0
            self.window_start = current_time
        
        # Check if we've hit the rate limit
        if self.request_count >= self.config.requests_per_minute:
            # Wait until the next minute
            wait_time = 60.0 - (current_time - self.window_start)
            if wait_time > 0:
                time.sleep(wait_time)
                self.request_count = 0
                self.window_start = time.time()
        
        # Check minimum interval between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.config.min_interval:
            wait_time = self.config.min_interval - time_since_last
            time.sleep(wait_time)
        
        # Update tracking
        self.last_request_time = time.time()
        self.request_count += 1
    
    def wait_for_retry(self, retry_delay: float = 60.0):
        """Wait for a retry after an error"""
        if not self.config.enabled:
            return
        
        time.sleep(retry_delay)
        # Reset counters after retry delay
        self.request_count = 0
        self.window_start = time.time()
    
    def disable(self):
        """Disable rate limiting (useful for testing)"""
        self.config.enabled = False
    
    def enable(self):
        """Enable rate limiting"""
        self.config.enabled = True
