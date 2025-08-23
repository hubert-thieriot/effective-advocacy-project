"""
Base finding extractor with caching mechanism
"""

import os
import json
import hashlib
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from efi_core.types import Finding, LibraryDocumentWFindings
from ..types import ExtractionConfig

logger = logging.getLogger(__name__)


class BaseFindingExtractor(ABC):
    """Base class for finding extractors with caching"""
    
    def __init__(self, 
                 name: str,
                 extraction_config: Optional[ExtractionConfig] = None,
                 cache_dir: str = "cache",
                 rate_limiter: Optional['DomainRateLimiter'] = None):
        """
        Initialize the base extractor
        
        Args:
            name: Name of the extractor
            extraction_config: Configuration for extraction
            cache_dir: Base cache directory
            rate_limiter: Optional rate limiter for HTTP requests
        """
        self.name = name
        self.extraction_config = extraction_config or ExtractionConfig()
        self.cache_dir = Path(cache_dir)
        self.rate_limiter = rate_limiter
        
        # Set up cache directories
        self.cache_base = self.cache_dir / f"finding_extraction_{self.name.lower()}"
        self.cache_base.mkdir(parents=True, exist_ok=True)
    
    def extract_findings(self, url: str, **kwargs) -> Optional[LibraryDocumentWFindings]:
        """
        Extract findings with caching
        
        Args:
            url: URL to extract findings from
            **kwargs: Additional arguments for extraction
            
        Returns:
            DocumentFindings object or None if extraction failed
        """
        # Generate cache key
        cache_key = self._generate_cache_key(url, **kwargs)
        cache_path = self.cache_base / cache_key
        
        # Check cache first
        cached_result = self._load_from_cache(cache_path)
        if cached_result:
            logger.info(f"Using cached findings for {url}")
            return cached_result
        
        # Extract findings
        logger.info(f"Extracting findings for {url} using {self.name}")
        try:
            result = self._extract_findings_impl(url, **kwargs)
            
            # Cache the result
            if result:
                self._save_to_cache(cache_path, result)
                logger.info(f"Successfully extracted and cached findings for {url}")
            else:
                logger.warning(f"No findings extracted for {url}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting findings for {url}: {e}")
            return None
    
    @abstractmethod
    def _extract_findings_impl(self, url: str, **kwargs) -> Optional[LibraryDocumentWFindings]:
        """
        Implementation of findings extraction
        
        Args:
            url: URL to extract findings from
            **kwargs: Additional arguments for extraction
            
        Returns:
            DocumentFindings object or None if extraction failed
        """
        pass
    
    def _generate_cache_key(self, url: str, **kwargs) -> str:
        """Generate cache key from URL and arguments"""
        # Create a string representation of the arguments
        args_str = json.dumps(kwargs, sort_keys=True, default=str)
        
        # Combine URL and arguments
        combined = f"{url}|{args_str}"
        
        # Generate hash
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _load_from_cache(self, cache_path: Path) -> Optional[LibraryDocumentWFindings]:
        """Load findings from cache"""
        try:
            if not cache_path.exists():
                return None
            
            # Load cached data
            with open(cache_path / "findings.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert back to LibraryDocumentWFindings
            return self._dict_to_findings(data)
            
        except Exception as e:
            logger.warning(f"Error loading from cache {cache_path}: {e}")
            return None
    
    def _save_to_cache(self, cache_path: Path, findings: LibraryDocumentWFindings):
        """Save findings to cache"""
        try:
            # Create cache directory
            cache_path.mkdir(parents=True, exist_ok=True)
            
            # Convert to dictionary
            data = self._findings_to_dict(findings)
            
            # Save to cache
            with open(cache_path / "findings.json", 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving to cache {cache_path}: {e}")
    
    def _findings_to_dict(self, doc_findings: LibraryDocumentWFindings) -> Dict[str, Any]:
        """Convert LibraryDocumentWFindings to dictionary for caching"""
        # Handle published_at field - it might be a string or datetime
        published_at = None
        if doc_findings.published_at:
            if hasattr(doc_findings.published_at, 'isoformat'):
                # It's a datetime object
                published_at = doc_findings.published_at.isoformat()
            else:
                # It's already a string
                published_at = str(doc_findings.published_at)
        
        return {
            'doc_id': doc_findings.doc_id,
            'url': doc_findings.url,
            'title': doc_findings.title,
            'published_at': published_at,
            'language': doc_findings.language,
            'extraction_date': doc_findings.extraction_date.isoformat(),
            'findings': [
                {
                    'text': f.text,
                    'confidence': f.confidence,
                    'category': f.category,
                    'keywords': f.keywords
                }
                for f in doc_findings.findings
            ],
            'metadata': doc_findings.metadata
        }
    
    def _dict_to_findings(self, data: Dict[str, Any]) -> LibraryDocumentWFindings:
        """Convert dictionary back to LibraryDocumentWFindings"""
        
        findings = []
        for f_data in data.get('findings', []):
            findings.append(Finding(
                text=f_data.get('text', ''),
                confidence=f_data.get('confidence'),
                category=f_data.get('category'),
                keywords=f_data.get('keywords')
            ))
        
        # Parse dates
        published_at = None
        if data.get('published_at'):
            try:
                published_at = datetime.fromisoformat(data['published_at'])
            except:
                pass
        
        extraction_date = datetime.now()
        if data.get('extraction_date'):
            try:
                extraction_date = datetime.fromisoformat(data['extraction_date'])
            except:
                pass
        
        return LibraryDocumentWFindings(
            doc_id=data.get('doc_id') or Finding.generate_doc_id(data.get('url','')),
            url=data.get('url', ''),
            title=data.get('title'),
            published_at=published_at,
            language=data.get('language'),
            extraction_date=extraction_date,
            findings=findings,
            metadata=data.get('metadata', {})
        )
    
    def clear_cache(self):
        """Clear the extractor's cache"""
        try:
            import shutil
            if self.cache_base.exists():
                shutil.rmtree(self.cache_base)
                logger.info(f"Cleared cache for {self.name}")
        except Exception as e:
            logger.error(f"Error clearing cache for {self.name}: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            if not self.cache_base.exists():
                return {'cached_items': 0, 'cache_size_bytes': 0}
            
            cached_items = 0
            cache_size = 0
            
            for item in self.cache_base.rglob('*'):
                if item.is_file():
                    cached_items += 1
                    cache_size += item.stat().st_size
            
            return {
                'cached_items': cached_items,
                'cache_size_bytes': cache_size,
                'cache_size_mb': round(cache_size / (1024 * 1024), 2)
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats for {self.name}: {e}")
            return {}
