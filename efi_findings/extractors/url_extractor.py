"""
Finding extractor that sends URLs directly to LLM
"""

import logging
from typing import Optional, Dict, Any
from .base_extractor import BaseFindingExtractor
from ..types import DocumentFindings, ExtractionConfig


class FindingExtractorFromUrl(BaseFindingExtractor):
    """Extract findings by sending URL directly to LLM"""
    
    def __init__(self, 
                 extraction_config: Optional[ExtractionConfig] = None,
                 cache_dir: str = "cache"):
        """
        Initialize the URL-based extractor
        
        Args:
            extraction_config: Configuration for LLM extraction
            cache_dir: Base cache directory
        """
        super().__init__("url", extraction_config, cache_dir)
    
    def _extract_findings_impl(self, url: str, **kwargs) -> Optional[DocumentFindings]:
        """
        Extract findings by sending URL directly to LLM
        
        Args:
            url: URL to extract findings from
            **kwargs: Additional arguments
            
        Returns:
            DocumentFindings object or None if extraction failed
        """
        logger = logging.getLogger(__name__)
        try:
            logger.info(f"Extracting findings directly from URL using LLM")
            
            # Create a custom prompt for URL-based extraction
            system_prompt = """You are an expert analyst specializing in extracting key findings from documents accessible via URLs. 
Your task is to analyze the content at the given URL and identify the most important insights, conclusions, and key points.

Focus on:
- Quantitative findings (numbers, statistics, percentages)
- Policy implications and recommendations
- Environmental and economic impacts
- Key trends and developments
- Critical challenges and solutions

Present your findings as a clear, structured list. Each finding should be a complete, standalone statement that captures one key insight."""

            extraction_prompt = """Please analyze the content at the following URL and extract the key findings.

URL: {url}

Extract 5-10 key findings from this document. Each finding should be a complete, factual statement that captures an important insight or conclusion.

Present your findings as a JSON array of strings, where each string is one finding:

[
  "Finding 1 description",
  "Finding 2 description",
  "Finding 3 description"
]

Focus on the most significant and actionable insights from the document."""

            # Use the LLM extractor with custom prompts
            from .findings_extractor import FindingsExtractor
            
            # Create custom config with URL-specific prompts
            custom_config = self.extraction_config
            custom_config.system_prompt = system_prompt
            custom_config.extraction_prompt = extraction_prompt.format(url=url)
            
            extractor = FindingsExtractor(custom_config)
            
            # For URL-based extraction, we pass an empty text and let the LLM handle the URL
            # The LLM will need to be configured to handle URLs directly
            doc_findings = extractor.extract_findings("", url, {})
            
            if not doc_findings.findings:
                logger.warning(f"No findings extracted from URL {url}")
            
            return doc_findings
            
        except Exception as e:
            logger.error(f"Error extracting findings from URL {url}: {e}")
            return None
