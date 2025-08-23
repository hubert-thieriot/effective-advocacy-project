"""
Finding extractor that extracts text first, then sends to LLM
"""

import logging
from typing import Optional, Dict, Any
from .base_extractor import BaseFindingExtractor
from ..types import ExtractionConfig
from efi_core.types import LibraryDocumentWFindings
from ..url_processor import URLProcessor


class FindingExtractorFromText(BaseFindingExtractor):
    """Extract findings by first extracting text, then sending to LLM"""
    
    def __init__(self, 
                 extraction_config: Optional[ExtractionConfig] = None,
                 cache_dir: str = "cache",
                 rate_limiter: Optional['DomainRateLimiter'] = None):
        """
        Initialize the text-based extractor
        
        Args:
            extraction_config: Configuration for LLM extraction
            cache_dir: Base cache directory
            rate_limiter: Optional rate limiter for HTTP requests
        """
        super().__init__("text", extraction_config, cache_dir, rate_limiter)
        self.url_processor = URLProcessor(rate_limiter=rate_limiter)
    
    def _extract_findings_impl(self, url: str, **kwargs) -> Optional[LibraryDocumentWFindings]:
        """
        Extract findings by processing text content
        
        Args:
            url: URL to extract findings from
            **kwargs: Additional arguments
            
        Returns:
            DocumentFindings object or None if extraction failed
        """
        logger = logging.getLogger(__name__)
        try:
            # Step 1: Extract text content from URL
            logger.info(f"Extracting text content from {url}")
            content_result = self.url_processor.process_url(url)
            
            if not content_result.get('success'):
                logger.error(f"Failed to extract content from {url}: {content_result.get('error')}")
                return None
            
            text = content_result.get('text', '')
            metadata = content_result.get('metadata', {})
            
            if not text.strip():
                logger.warning(f"No text content extracted from {url}")
                return None
            
            logger.info(f"Extracted {len(text)} characters from {url}")
            
            # Step 2: Extract findings using LLM
            logger.info(f"Extracting findings from text using LLM")
            from .findings_extractor import FindingsExtractor
            
            extractor = FindingsExtractor(self.extraction_config)
            doc_findings = extractor.extract_findings(text, url, metadata)
            
            if not doc_findings.findings:
                logger.warning(f"No findings extracted from {url}")
            
            return doc_findings
            
        except Exception as e:
            logger.error(f"Error extracting findings from text for {url}: {e}")
            return None
    
    def close(self):
        """Close the URL processor"""
        self.url_processor.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
