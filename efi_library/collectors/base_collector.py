"""
Base URL collector for discovering content sources
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseURLCollector(ABC):
    """Abstract base class for URL collectors"""
    
    def __init__(self, name: str, max_sources: Optional[int] = None):
        """
        Initialize the collector
        
        Args:
            name: Name of the collector
            max_sources: Maximum number of sources to collect (None for all)
        """
        self.name = name
        self.max_sources = max_sources
    
    @abstractmethod
    def collect_urls(self) -> List[str]:
        """
        Collect URLs from the source
        
        Returns:
            List of discovered URLs
        """
        pass
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collector"""
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'max_sources': self.max_sources
        }
    
    def collect_urls_limited(self) -> List[str]:
        """
        Collect URLs with optional limit
        
        Returns:
            List of discovered URLs (limited if max_sources is set)
        """
        urls = self.collect_urls()
        
        if self.max_sources and len(urls) > self.max_sources:
            logger.info(f"Limiting collection to {self.max_sources} sources (found {len(urls)})")
            urls = urls[:self.max_sources]
        
        return urls
