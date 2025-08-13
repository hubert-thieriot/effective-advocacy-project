"""
URL processor for handling PDF and webpage URLs
"""

import requests
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import urlparse
from datetime import datetime
import PyPDF2
import io
from newspaper import Article
from bs4 import BeautifulSoup
import logging
from .utils import normalize_date

logger = logging.getLogger(__name__)


class URLProcessor:
    """Process URLs to extract text content from PDFs and webpages"""
    
    def __init__(self, timeout: int = 30, user_agent: Optional[str] = None):
        """
        Initialize URL processor
        
        Args:
            timeout: Request timeout in seconds
            user_agent: Custom user agent string
        """
        self.timeout = timeout
        self.user_agent = user_agent or "Mozilla/5.0 (compatible; EFI-Findings/1.0)"
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': self.user_agent})
    
    def process_url(self, url: str) -> Dict[str, Any]:
        """
        Process a URL and extract text content
        
        Args:
            url: URL to process
            
        Returns:
            Dictionary with extracted content and metadata
        """
        try:
            parsed_url = urlparse(url)
            
            if url.lower().endswith('.pdf') or 'pdf' in parsed_url.path.lower():
                return self._process_pdf(url)
            else:
                return self._process_webpage(url)
                
        except Exception as e:
            logger.error(f"Error processing URL {url}: {e}")
            return {
                'success': False,
                'error': str(e),
                'text': '',
                'metadata': {}
            }
    
    def _process_pdf(self, url: str) -> Dict[str, Any]:
        """Process PDF URL and extract text"""
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Check if it's actually a PDF
            if not response.headers.get('content-type', '').lower().startswith('application/pdf'):
                logger.warning(f"URL {url} doesn't return PDF content")
                return self._process_webpage(url)
            
            # Extract text from PDF
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return {
                'success': True,
                'text': text.strip(),
                'metadata': {
                    'content_type': 'pdf',
                    'page_count': len(pdf_reader.pages),
                    'file_size': len(response.content)
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF {url}: {e}")
            return {
                'success': False,
                'error': str(e),
                'text': '',
                'metadata': {}
            }
    
    def _process_webpage(self, url: str) -> Dict[str, Any]:
        """Process webpage URL and extract text"""
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            # Extract main text content
            text = article.text
            
            # If newspaper3k didn't extract much text, try alternative approach
            if len(text.strip()) < 100:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                
                # Simple text extraction as fallback
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text content
                text = soup.get_text()
                
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Safely extract metadata attributes
            metadata = {
                'content_type': 'webpage',
                'title': getattr(article, 'title', None),
                'publish_date': None,
                'authors': getattr(article, 'authors', []),
                'language': getattr(article, 'language', None)
            }
            
            # Enhanced date extraction
            metadata['publish_date'] = self._extract_publication_date(article, url)
            
            return {
                'success': True,
                'text': text.strip(),
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error processing webpage {url}: {e}")
            # Try fallback method
            try:
                logger.info(f"Trying fallback text extraction for {url}")
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text content
                text = soup.get_text()
                
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                if text.strip():
                    return {
                        'success': True,
                        'text': text.strip(),
                        'metadata': {
                            'content_type': 'webpage',
                            'title': soup.title.string if soup.title else None,
                            'publish_date': self._extract_publication_date_from_soup(soup, url),
                            'authors': [],
                            'language': None
                        }
                    }
            except Exception as fallback_error:
                logger.error(f"Fallback extraction also failed for {url}: {fallback_error}")
            
            return {
                'success': False,
                'error': str(e),
                'text': '',
                'metadata': {}
            }
    
    def _extract_publication_date(self, article, url: str) -> Optional[datetime]:
        """Extract publication date from newspaper3k article or fallback to HTML"""
        try:
            # Try newspaper3k publish_date first
            if hasattr(article, 'publish_date') and article.publish_date:
                return normalize_date(article.publish_date)
            
            # Fallback: try to extract from HTML
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            return self._extract_publication_date_from_soup(soup, url)
            
        except Exception as e:
            logger.debug(f"Error extracting publication date from {url}: {e}")
            return None
    
    def _extract_publication_date_from_soup(self, soup, url: str) -> Optional[datetime]:
        """Extract publication date from BeautifulSoup object"""
        try:
            # Common date selectors for different websites
            date_selectors = [
                # Meta tags
                'meta[property="article:published_time"]',
                'meta[name="publish_date"]',
                'meta[name="date"]',
                'meta[property="og:updated_time"]',
                'meta[property="og:published_time"]',
                
                # Schema.org structured data
                'time[datetime]',
                '[itemprop="datePublished"]',
                '[itemprop="dateCreated"]',
                
                # Common CSS classes
                '.publish-date',
                '.post-date',
                '.article-date',
                '.date',
                '.timestamp',
                
                # CREA-specific selectors
                '.entry-date',
                '.publication-date',
                '.report-date'
            ]
            
            # Try meta tags first (most reliable)
            for selector in date_selectors[:5]:  # Meta tags only
                element = soup.select_one(selector)
                if element:
                    date_str = element.get('content') or element.get('datetime')
                    if date_str:
                        parsed_date = normalize_date(date_str)
                        if parsed_date:
                            return parsed_date
            
            # Try visible date elements
            for selector in date_selectors[5:]:
                element = soup.select_one(selector)
                if element:
                    date_text = element.get_text().strip()
                    if date_text:
                        parsed_date = normalize_date(date_text)
                        if parsed_date:
                            return parsed_date
            
            # Try to find any date-like text in the content
            date_patterns = [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # DD/MM/YYYY or DD-MM-YYYY
                r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',  # YYYY/MM/DD or YYYY-MM-DD
                r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b',  # Month DD, YYYY
                r'\b\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b'  # DD Month YYYY
            ]
            
            content_text = soup.get_text()
            for pattern in date_patterns:
                import re
                matches = re.findall(pattern, content_text, re.IGNORECASE)
                if matches:
                    # Take the first match that looks like a publication date
                    for match in matches[:3]:  # Check first 3 matches
                        parsed_date = normalize_date(match)
                        if parsed_date:
                            # Prefer dates that are recent (not too old)
                            if parsed_date.year >= 2020:  # Reasonable publication year
                                return parsed_date
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting date from HTML for {url}: {e}")
            return None
    
    def _parse_date_string(self, date_str: str) -> Optional[datetime]:
        """Parse various date string formats"""
        if not date_str:
            return None
        
        # Common date formats
        date_formats = [
            '%Y-%m-%d',           # 2025-08-13
            '%Y/%m/%d',           # 2025/08/13
            '%d/%m/%Y',           # 13/08/2025
            '%d-%m-%Y',           # 13-08-2025
            '%B %d, %Y',          # August 13, 2025
            '%b %d, %Y',          # Aug 13, 2025
            '%d %B %Y',           # 13 August 2025
            '%d %b %Y',           # 13 Aug 2025
            '%Y-%m-%dT%H:%M:%S', # ISO format
            '%Y-%m-%dT%H:%M:%SZ' # ISO format with Z
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        
        # Try parsing with dateutil if available (more flexible)
        try:
            from dateutil import parser
            return parser.parse(date_str)
        except ImportError:
            pass
        except Exception:
            pass
        
        return None
    
    def close(self):
        """Close the session"""
        self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
