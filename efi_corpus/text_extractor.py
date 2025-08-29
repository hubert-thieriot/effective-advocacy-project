"""
Text extraction from various content types
"""

import re
from typing import Dict, Any, Optional
from pathlib import Path
import json
import time

try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False


class TextExtractor:
    """Extract text content from various document formats"""
    
    def __init__(self):
        self.error_phrases = [
            "verify you are human",
            "verification successful",
            "waiting for",
            "to respond",
            "security check",
            "captcha",
            "access denied",
            "403 forbidden",
            "404 not found",
            "page not found",
            "robot check",
            "javascript is disabled",
            "enable javascript",
            "cookies are disabled",
            "please enable cookies",
            "cloudflare",
            "ddos protection",
            "checking your browser",
            "please wait",
            "security challenge"
        ]
        
        # Create failed extraction directory - use project root relative path
        # Try to find the project root by looking for pyproject.toml
        project_root = Path(__file__).parent.parent
        self.failed_dir = project_root / "debug" / "failed_to_extract"
        self.failed_dir.mkdir(parents=True, exist_ok=True)
    
    def _save_failed_extraction(self, url: str, raw_bytes: bytes, raw_ext: str, 
                               extraction_attempts: Dict[str, Any], error: str = None):
        """Save failed extraction details for debugging"""
        try:
            # Create a safe filename from URL
            safe_url = url.replace("://", "_").replace("/", "_").replace("?", "_").replace("&", "_")
            safe_url = "".join(c for c in safe_url if c.isalnum() or c in "_-")
            if len(safe_url) > 100:
                safe_url = safe_url[:100]
            
            timestamp = int(time.time())
            failed_dir = self.failed_dir / f"{timestamp}_{safe_url}"
            failed_dir.mkdir(exist_ok=True)
            
            # Save the raw content
            if raw_ext == 'html':
                (failed_dir / "content.html").write_bytes(raw_bytes)
            else:
                (failed_dir / f"content.{raw_ext}").write_bytes(raw_bytes)
            
            # Save extraction details
            debug_info = {
                "url": url,
                "timestamp": timestamp,
                "raw_ext": raw_ext,
                "content_size": len(raw_bytes),
                "extraction_attempts": extraction_attempts,
                "error": error
            }
            (failed_dir / "debug.json").write_text(json.dumps(debug_info, indent=2), encoding="utf-8")
            
            print(f"Failed extraction saved to: {failed_dir}")
            
        except Exception as e:
            print(f"Warning: Could not save failed extraction: {e}")
    
    def extract_text(self, raw_bytes: bytes, raw_ext: str, url: str = "") -> Dict[str, Any]:
        """
        Extract text and metadata from raw content
        
        Args:
            raw_bytes: Raw document content
            raw_ext: File extension indicating content type
            url: Source URL for context
            
        Returns:
            Dict with keys: text, title, published_at, language, authors
        """
        extraction_attempts = {}
        
        if raw_ext.lower() in ['html', 'htm']:
            result = self._extract_from_html(raw_bytes, url, extraction_attempts)
        elif raw_ext.lower() == 'pdf':
            result = self._extract_from_pdf(raw_bytes, url, extraction_attempts)
        elif raw_ext.lower() in ['txt', 'text']:
            result = self._extract_from_text(raw_bytes, url, extraction_attempts)
        else:
            # Try to detect content type and extract accordingly
            result = self._extract_auto_detect(raw_bytes, url, extraction_attempts)
        
        # If extraction failed, save for debugging
        if not result or not result.get("text"):
            self._save_failed_extraction(url, raw_bytes, raw_ext, extraction_attempts, 
                                      "No text extracted or extraction failed")
        
        return result
    
    def _extract_from_html(self, html_bytes: bytes, url: str = "", extraction_attempts: Dict[str, Any] = {}) -> Dict[str, Any]:
        """Extract text from HTML content using multiple strategies"""
        try:
            html = html_bytes.decode('utf-8', errors='ignore')
        except UnicodeDecodeError:
            try:
                html = html_bytes.decode('latin-1', errors='ignore')
            except UnicodeDecodeError:
                html = html_bytes.decode('cp1252', errors='ignore')
        
        if not html:
            extraction_attempts["html_decode"] = "Failed to decode HTML"
            return self._empty_result()
        
        extraction_attempts["html_decode"] = f"Success, length: {len(html)}"
        
        # Strategy 1: Try newspaper3k if available
        if NEWSPAPER_AVAILABLE:
            try:
                article = Article(url)
                article.download(input_html=html)
                article.parse()
                text = article.text
                
                extraction_attempts["newspaper3k"] = {
                    "success": True,
                    "text_length": len(text),
                    "title": article.title,
                    "authors": article.authors
                }
                
                if self._is_valid_content(text):
                    return {
                        "text": text,
                        "title": article.title,
                        "published_at": article.publish_date.isoformat() if article.publish_date else None,
                        "language": article.meta_lang or "en",
                        "authors": article.authors or []
                    }
                else:
                    extraction_attempts["newspaper3k"]["validation_failed"] = True
            except Exception as e:
                extraction_attempts["newspaper3k"] = {
                    "success": False,
                    "error": str(e)
                }
                pass  # Fall back to BeautifulSoup
        
        # Strategy 2: BeautifulSoup extraction
        if BEAUTIFULSOUP_AVAILABLE:
            try:
                text = self._extract_with_beautifulsoup(html, url)
                
                # If BeautifulSoup extraction didn't get much content, try JavaScript paywall handling
                if not text or len(text) < 1000:
                    text = self._extract_before_javascript_paywall(html, url)
                
                extraction_attempts["beautifulsoup"] = {
                    "success": True,
                    "text_length": len(text)
                }
                
                # If we got substantial text, return it even if validation fails
                if text and len(text) > 500:  # Lower threshold for substantial content
                    return {
                        "text": text,
                        "title": self._extract_title_from_html(html),
                        "published_at": None,  # Would need more sophisticated extraction
                        "language": "en",  # Would need language detection
                        "authors": []  # Would need author extraction
                    }
                else:
                    extraction_attempts["beautifulsoup"]["validation_failed"] = True
            except Exception as e:
                extraction_attempts["beautifulsoup"] = {
                    "success": False,
                    "error": str(e)
                }
                pass
        
        # Strategy 3: Basic regex-based extraction
        try:
            text = self._extract_with_regex(html)
            extraction_attempts["regex"] = {
                "success": True,
                "text_length": len(text)
            }
            
            if self._is_valid_content(text):
                return {
                    "text": text,
                    "title": self._extract_title_from_html(html),
                    "published_at": None,
                    "language": "en",
                    "authors": []
                }
            else:
                extraction_attempts["regex"]["validation_failed"] = True
        except Exception as e:
            extraction_attempts["regex"] = {
                "success": False,
                "error": str(e)
            }
        
        # All strategies failed
        extraction_attempts["final_result"] = "All extraction strategies failed"
        return self._empty_result()
    
    def _extract_before_javascript_paywall(self, html: str, url: str) -> str:
        """Extract text before JavaScript paywall kicks in for sites like Indian Express"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script tags and other dynamic content
        for unwanted in soup.select('script, style, .paywall, .premium-content, .subscription-required'):
            unwanted.decompose()
        
        # Site-specific extraction for known paywall sites
        url_lower = url.lower()
        
        if 'indianexpress.com' in url_lower:
            # Indian Express specific selectors that work before JS runs
            selectors = [
                '.story-details', '.story-content', '.article-content', 
                '.entry-content', '.post-content', '.content-area',
                'main', '.main-content', '.article-body'
            ]
            
            for selector in selectors:
                elements = soup.select(selector)
                if elements:
                    # Find the largest content block
                    content_element = max(elements, key=lambda e: len(e.get_text()))
                    text = content_element.get_text(separator='\n\n')
                    
                    # Clean up the text
                    lines = [line.strip() for line in text.splitlines() if line.strip()]
                    cleaned_text = '\n\n'.join(lines)
                    
                    if len(cleaned_text) > 1000:  # Ensure substantial content
                        return cleaned_text
        
        elif 'hindustantimes.com' in url_lower:
            # Hindustan Times specific selectors
            selectors = [
                '.story-content', '.story-body', '.article-content', 
                '.content', '.main-content', '.story-details',
                'main', 'article'
            ]
            
            for selector in selectors:
                elements = soup.select(selector)
                if elements:
                    content_element = max(elements, key=lambda e: len(e.get_text()))
                    text = content_element.get_text(separator='\n\n')
                    lines = [line.strip() for line in text.splitlines() if line.strip()]
                    cleaned_text = '\n\n'.join(lines)
                    
                    if len(cleaned_text) > 1000:
                        return cleaned_text
        
        # Fallback: try to get all text content
        text = soup.get_text(separator='\n\n')
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return '\n\n'.join(lines)
    
    def _extract_with_beautifulsoup(self, html: str, url: str = "") -> str:
        """Extract text using BeautifulSoup with content selection"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted elements
        for unwanted in soup.select('script, style, nav, header, footer, .ads, .comments, .sidebar, .social-share, .related-posts'):
            unwanted.decompose()
        
        # Common article content selectors (expanded list)
        content_selectors = [
            'article', '.article', '.post', '.content', '.story', '.entry',
            '.article-content', '.article-body', '.entry-content', '.post-content',
            '.story-content', '.story-body', '.main-content', '#article-body',
            '.news-content', '.article__content', '.article__body',
            # Add more specific selectors for news sites
            '.entry-content', '.post-body', '.content-area', '.main-article',
            '.article-text', '.story-text', '.news-text', '.content-body'
        ]
        
        # Try to find content using selectors
        content_element = None
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                # Find the largest content block
                content_element = max(elements, key=lambda e: len(e.get_text()))
                break
        
        if content_element:
            # Extract text from the content element
            paragraphs = content_element.find_all('p')
            if paragraphs:
                text = '\n\n'.join([p.get_text().strip() for p in paragraphs])
            else:
                text = content_element.get_text(separator='\n\n')
        else:
            # Fallback: look for content within article tags
            article = soup.find('article')
            if article:
                paragraphs = article.find_all('p')
                if paragraphs:
                    text = '\n\n'.join([p.get_text().strip() for p in paragraphs])
                else:
                    text = article.get_text(separator='\n\n')
            else:
                # Fallback: get all paragraphs from the page
                paragraphs = soup.find_all('p')
                if paragraphs:
                    # Filter out very short paragraphs that might be navigation/ads
                    valid_paragraphs = [p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 20]
                    text = '\n\n'.join(valid_paragraphs)
                else:
                    # Last resort: get all text
                    text = soup.get_text(separator='\n\n')
        
        # Clean up the text
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return '\n\n'.join(lines)
    
    def _extract_with_regex(self, html: str) -> str:
        """Basic regex-based text extraction as fallback"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', html)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove common HTML entities
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)
        text = re.sub(r'&#\d+;', ' ', text)
        return text.strip()
    
    def _extract_title_from_html(self, html: str) -> Optional[str]:
        """Extract title from HTML"""
        # Try to find title tag
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', html, re.IGNORECASE)
        if title_match:
            return title_match.group(1).strip()
        
        # Try to find h1 tag
        h1_match = re.search(r'<h1[^>]*>([^<]+)</h1>', html, re.IGNORECASE)
        if h1_match:
            return h1_match.group(1).strip()
        
        return None
    
    def _extract_from_pdf(self, pdf_bytes: bytes, url: str = "", extraction_attempts: Dict[str, Any] = {}) -> Dict[str, Any]:
        """Extract text from PDF content"""
        # TODO: Implement PDF text extraction
        # This would require PyPDF2, pdfplumber, or similar
        return self._empty_result()
    
    def _extract_from_text(self, text_bytes: bytes, url: str = "", extraction_attempts: Dict[str, Any] = {}) -> Dict[str, Any]:
        """Extract text from plain text content"""
        try:
            text = text_bytes.decode('utf-8', errors='ignore')
        except UnicodeDecodeError:
            try:
                text = text_bytes.decode('latin-1', errors='ignore')
            except UnicodeDecodeError:
                text = text_bytes.decode('cp1252', errors='ignore')
        
        if self._is_valid_content(text):
            return {
                "text": text,
                "title": None,
                "published_at": None,
                "language": "en",
                "authors": []
            }
        
        return self._empty_result()
    
    def _extract_auto_detect(self, raw_bytes: bytes, url: str = "", extraction_attempts: Dict[str, Any] = {}) -> Dict[str, Any]:
        """Auto-detect content type and extract accordingly"""
        # Try to detect if it's HTML
        if raw_bytes.startswith(b'<!DOCTYPE') or raw_bytes.startswith(b'<html'):
            return self._extract_from_html(raw_bytes, url, extraction_attempts)
        
        # Try to detect if it's PDF
        if raw_bytes.startswith(b'%PDF'):
            return self._extract_from_pdf(raw_bytes, url, extraction_attempts)
        
        # Assume it's text
        return self._extract_from_text(raw_bytes, url, extraction_attempts)
    
    def _is_valid_content(self, text: str) -> bool:
        """Determine if extracted content is valid"""
        if not text or len(text) < 200:
            return False
        
        # Check for error phrases
        text_lower = text.lower()
        for phrase in self.error_phrases:
            if phrase in text_lower:
                return False
        
        # Check for reasonable word count
        words = text.split()
        if len(words) < 20:  # Reduced from 100 for testing
            return False
        
        return True
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure"""
        return {
            "text": "",
            "title": None,
            "published_at": None,
            "language": "en",
            "authors": []
        }
