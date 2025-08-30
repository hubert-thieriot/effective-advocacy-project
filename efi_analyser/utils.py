"""
Utility functions and classes for the EFI Analyser.
"""

import re
from typing import Dict, List
from urllib.parse import urlparse


class MediaSourceMapper:
    """
    Maps URLs to media sources by extracting and grouping domain information.
    
    This utility helps categorize documents by their source media outlet,
    grouping related domains together for analysis.
    """
    
    def __init__(self):
        """Initialize the media source mapper."""
        # Define media source patterns and their associated domains
        self.media_patterns = {
            "BBC": [
                r"bbc\.com",
                r"bbc\.co\.uk",
                r"bbc\.in",
                r"bbc\.com\.au"
            ],
            "CNN": [
                r"cnn\.com",
                r"cnn\.es",
                r"cnn\.ar",
                r"cnn\.de"
            ],
            "Reuters": [
                r"reuters\.com",
                r"reuters\.co\.uk",
                r"reuters\.in"
            ],
            "Associated Press": [
                r"ap\.org",
                r"apnews\.com"
            ],
            "The Guardian": [
                r"theguardian\.com",
                r"guardian\.co\.uk"
            ],
            "New York Times": [
                r"nytimes\.com",
                r"nytimes\.co\.uk"
            ],
            "Washington Post": [
                r"washingtonpost\.com",
                r"wapo\.st"
            ],
            "Al Jazeera": [
                r"aljazeera\.com",
                r"aljazeera\.net"
            ],
            "Times of India": [
                r"timesofindia\.indiatimes\.com",
                r"timesofindia\.com"
            ],
            "The Hindu": [
                r"thehindu\.com",
                r"thehindu\.in"
            ],
            "Hindustan Times": [
                r"hindustantimes\.com",
                r"ht\.com"
            ],
            "The Indian Express": [
                r"indianexpress\.com",
                r"expressindia\.com"
            ],
            "South China Morning Post": [
                r"scmp\.com",
                r"scmp\.hk"
            ],
            "China Daily": [
                r"chinadaily\.com\.cn",
                r"chinadaily\.cn"
            ],
            "Xinhua": [
                r"xinhuanet\.com",
                r"news\.cn"
            ],
            "Jakarta Post": [
                r"thejakartapost\.com",
                r"jakpost\.com"
            ],
            "Kompas": [
                r"kompas\.com",
                r"kompas\.id"
            ],
            "Dawn": [
                r"dawn\.com",
                r"dawn\.pk"
            ],
            "The Express Tribune": [
                r"tribune\.com\.pk",
                r"express\.com\.pk"
            ],
            "Daily Star": [
                r"thedailystar\.net",
                r"thedailystar\.com"
            ],
            "The Daily Star": [
                r"dailystar\.co\.uk",
                r"dailystar\.com"
            ],
            "News24": [
                r"news24\.com",
                r"news24\.co\.za"
            ],
            "Mail & Guardian": [
                r"mg\.co\.za",
                r"mg\.com"
            ]
        }
        
        # Compile regex patterns for efficiency
        self.compiled_patterns = {}
        for media_source, patterns in self.media_patterns.items():
            self.compiled_patterns[media_source] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]
    
    def get_media_source(self, url: str) -> str:
        """
        Extract media source from a URL.
        
        Args:
            url: The URL to analyze
            
        Returns:
            The media source name, or empty string if not recognized
        """
        if not url or not isinstance(url, str):
            return ""
        
        try:
            # Parse the URL
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Check against known media patterns
            for media_source, patterns in self.compiled_patterns.items():
                for pattern in patterns:
                    if pattern.search(domain):
                        return media_source
            
            # If no pattern matches, return the base domain
            if domain:
                # Extract the main domain (e.g., "example.com" from "sub.example.com")
                domain_parts = domain.split('.')
                if len(domain_parts) >= 2:
                    return f"{domain_parts[-2]}.{domain_parts[-1]}"
                return domain
            
            return ""
            
        except Exception:
            return ""
    
    def get_media_sources(self, urls: List[str]) -> Dict[str, List[str]]:
        """
        Group URLs by media source.
        
        Args:
            urls: List of URLs to categorize
            
        Returns:
            Dictionary mapping media sources to lists of URLs
        """
        media_groups = {}
        
        for url in urls:
            media_source = self.get_media_source(url)
            if media_source:
                if media_source not in media_groups:
                    media_groups[media_source] = []
                media_groups[media_source].append(url)
        
        return media_groups
    
    def get_media_source_counts(self, urls: List[str]) -> Dict[str, int]:
        """
        Count URLs by media source.
        
        Args:
            urls: List of URLs to count
            
        Returns:
            Dictionary mapping media sources to counts
        """
        counts = {}
        
        for url in urls:
            media_source = self.get_media_source(url)
            if media_source:
                counts[media_source] = counts.get(media_source, 0) + 1
        
        return counts
