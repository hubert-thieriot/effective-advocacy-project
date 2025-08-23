"""
LLM-based extractor for key findings
"""

import os
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from efi_core.types import Finding, LibraryDocumentWFindings
from ..types import ExtractionConfig
from ..utils import normalize_date
from ..url_processor import URLProcessor

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class FindingsExtractor:
    """Extract key findings from document text using LLM"""
    
    def __init__(self, config: Optional[ExtractionConfig] = None):
        """
        Initialize the extractor
        
        Args:
            config: Configuration for extraction
        """
        self.config = config or ExtractionConfig()
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Set default prompts if not provided
        if not self.config.system_prompt:
            self.config.system_prompt = self._get_default_system_prompt()
        
        if not self.config.extraction_prompt:
            self.config.extraction_prompt = self._get_default_extraction_prompt()
    
    def extract_findings(self, text: str, url: str, metadata: Optional[Dict[str, Any]] = None) -> LibraryDocumentWFindings:
        """
        Extract key findings from document text
        
        Args:
            text: Document text to analyze
            url: Source URL
            metadata: Additional document metadata
            
        Returns:
            DocumentFindings object with extracted findings
        """
        try:
            # Prepare the extraction prompt
            prompt = self.config.extraction_prompt.format(
                text=text[:8000],  # Limit text length to avoid token limits
                url=url
            )
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            # Parse the response to extract findings and publication date
            findings, llm_publication_date = self._parse_findings_response(response.choices[0].message.content)
            
            # Use LLM-extracted date if available, otherwise fall back to metadata
            publication_date = normalize_date(llm_publication_date or metadata.get('publish_date'))
            
            # Create LibraryDocumentWFindings object
            doc_findings = LibraryDocumentWFindings(
                doc_id=Finding.generate_doc_id(url),
                url=url,
                title=metadata.get('title') if metadata else None,
                published_at=publication_date,
                language=metadata.get('language') if metadata else None,
                findings=findings,
                metadata=metadata or {}
            )
            
            logger.info(f"Successfully extracted {len(findings)} findings from {url}")
            return doc_findings
            
        except Exception as e:
            logger.error(f"Error extracting findings from {url}: {e}")
            # Return empty findings on error
            return LibraryDocumentWFindings(
                doc_id=Finding.generate_doc_id(url),
                url=url,
                findings=[],
                metadata={**(metadata or {}), 'metadata_error': str(e)}
            )
    
    def _parse_findings_response(self, response_text: str) -> Tuple[List[Finding], Optional[str]]:
        """Parse the LLM response to extract findings and publication date"""
        try:
            # Split response into lines and look for findings
            lines = response_text.strip().split('\n')
            findings = []
            llm_publication_date = None
            
            current_finding = ""
            finding_number = 0
            
            for line in lines:
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Look for publication date
                if not llm_publication_date:
                    llm_publication_date = self._extract_publication_date_from_response(line)
                
                # Look for numbered findings (e.g., "1.", "2.", etc.)
                if line and line[0].isdigit() and line[1] in ['.', ')', ':']:
                    # Save previous finding if exists
                    if current_finding.strip():
                        findings.append(Finding(text=current_finding.strip()))
                        current_finding = ""
                    
                    # Start new finding
                    current_finding = line[line.find('.')+1:].strip()
                    finding_number += 1
                else:
                    # Continue building current finding
                    if current_finding:
                        current_finding += " " + line
            
            # Add the last finding
            if current_finding.strip():
                findings.append(Finding(text=current_finding.strip()))
            
            # If no numbered findings found, try to split by other patterns
            if not findings:
                # Split by double newlines or other separators
                sections = response_text.split('\n\n')
                for section in sections:
                    section = section.strip()
                    if section and len(section) > 50:  # Minimum length for a finding
                        findings.append(Finding(text=section))
            
            return findings, llm_publication_date
            
        except Exception as e:
            logger.error(f"Error parsing findings response: {e}")
            return [], None
    
    def _extract_publication_date_from_response(self, response_text: str) -> Optional[str]:
        """Extract publication date from LLM response"""
        try:
            # Look for date patterns in the response
            import re
            
            # Common date patterns
            date_patterns = [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # DD/MM/YYYY or DD-MM-YYYY
                r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',  # YYYY/MM/DD or YYYY-MM-DD
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',  # Month DD, YYYY
                r'\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b',  # DD Month YYYY
                r'\b(?:Published|Date|Released):\s*([^\n]+)',  # "Published: Date"
                r'\b(?:Date|Published):\s*(\d{4}-\d{2}-\d{2})',  # ISO format
            ]
            
            for pattern in date_patterns:
                matches = re.findall(pattern, response_text, re.IGNORECASE)
                if matches:
                    date_str = matches[0]
                    # Clean up the date string
                    date_str = re.sub(r'[^\w\s\-/]', '', date_str).strip()
                    
                    # Try to parse the date
                    parsed_date = self._parse_date_string(date_str)
                    if parsed_date:
                        return parsed_date.isoformat()
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting publication date from response: {e}")
            return None
    
    def _parse_date_string(self, date_str: str) -> Optional[datetime]:
        """Parse various date string formats"""
        try:
            from datetime import datetime
            
            # Try ISO format first
            try:
                return datetime.fromisoformat(date_str)
            except ValueError:
                pass
            
            # Try other common formats
            formats = [
                '%Y-%m-%d',
                '%d/%m/%Y',
                '%m/%d/%Y',
                '%d-%m-%Y',
                '%m-%d-%Y',
                '%B %d, %Y',
                '%b %d, %Y',
                '%d %B %Y',
                '%d %b %Y'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            
            return None
            
        except Exception as e:
            logger.debug(f"Error parsing date string '{date_str}': {e}")
            return None
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for extraction"""
        return """You are an expert research analyst specializing in energy, environment, and climate policy. 
Your task is to extract key findings from research documents and reports.

Key findings should be:
- Specific, factual statements supported by the document
- Important insights, conclusions, or policy implications
- Quantified results where available (numbers, percentages, dates)
- Clear and concise (1-2 sentences each)

Focus on the most significant findings that would be relevant for policy makers, researchers, or journalists."""
    
    def _get_default_extraction_prompt(self) -> str:
        """Get default extraction prompt"""
        return """Please analyze the following text and extract the key findings. 

Document URL: {url}

Text Content:
{text}

Please provide:
1. A numbered list of key findings (aim for 5-10 findings)
2. If you can identify a publication date, include it in your response

Format your response as:
1. [First key finding]
2. [Second key finding]
3. [Third key finding]
...

Publication Date: [Date if identifiable, otherwise "Not specified"]"""
