"""
Text extraction from various content types
"""

import re
from typing import Dict, Any, Optional
from pathlib import Path
import json
import time

from newspaper import Article
from bs4 import BeautifulSoup

# BeautifulSoup is always available since it's imported
BEAUTIFULSOUP_AVAILABLE = True

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
            from efi_core.utils import DateTimeEncoder
            (failed_dir / "debug.json").write_text(json.dumps(debug_info, indent=2, cls=DateTimeEncoder), encoding="utf-8")
            
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
        newspaper_text = None
        newspaper_title = None
        newspaper_authors = []
        newspaper_published_at = None
        try:
            article = Article(url)
            article.download(input_html=html)
            article.parse()
            newspaper_text = article.text
            
            extraction_attempts["newspaper3k"] = {
                "success": True,
                "text_length": len(newspaper_text),
                "title": article.title,
                "authors": article.authors
            }
            
            if self._is_valid_content(newspaper_text):
                newspaper_title = article.title
                newspaper_authors = article.authors or []
                newspaper_published_at = article.publish_date.isoformat() if article.publish_date else None
            else:
                extraction_attempts["newspaper3k"]["validation_failed"] = True
                newspaper_text = None  # Don't use invalid content
        except Exception as e:
            extraction_attempts["newspaper3k"] = {
                "success": False,
                "error": str(e)
            }
            pass  # Fall back to BeautifulSoup
        
        # Strategy 2: BeautifulSoup extraction
        try:
            bs_text = self._extract_with_beautifulsoup(html, url)
            
            # If BeautifulSoup extraction didn't get much content, try JavaScript paywall handling
            if not bs_text or len(bs_text) < 1000:
                bs_text = self._extract_before_javascript_paywall(html, url)
            
            extraction_attempts["beautifulsoup"] = {
                "success": True,
                "text_length": len(bs_text) if bs_text else 0
            }
            
            # Choose the best extraction: prefer longer content
            # If BeautifulSoup extracted significantly more content, use it instead
            if bs_text and len(bs_text) > 500:
                # Use BeautifulSoup if it has substantially more content (at least 2x) than newspaper3k
                if not newspaper_text or len(bs_text) >= len(newspaper_text) * 1.5:
                    # Try to extract published date from both HTML and JSON-LD
                    published_at = self._extract_published_date_from_html(html)
                    if not published_at and BEAUTIFULSOUP_AVAILABLE:
                        soup = BeautifulSoup(html, 'html.parser')
                        published_at = self._extract_published_date_from_json_ld(soup)
                    
                    return {
                        "text": bs_text,
                        "title": self._extract_title_from_html(html) or newspaper_title,
                        "published_at": published_at or newspaper_published_at,
                        "language": "en",  # Would need language detection
                        "authors": newspaper_authors  # Use authors from newspaper3k if available
                    }
            
            # If BeautifulSoup didn't work well, fall back to newspaper3k if we have it
            if newspaper_text and self._is_valid_content(newspaper_text):
                return {
                    "text": newspaper_text,
                    "title": newspaper_title,
                    "published_at": newspaper_published_at,
                    "language": "en",
                    "authors": newspaper_authors
                }
            
            if bs_text and len(bs_text) > 500:
                extraction_attempts["beautifulsoup"]["validation_failed"] = True
        except Exception as e:
            extraction_attempts["beautifulsoup"] = {
                "success": False,
                "error": str(e)
            }
            pass
        
        # If we have newspaper3k content but BeautifulSoup failed, use it
        if newspaper_text and self._is_valid_content(newspaper_text):
            return {
                "text": newspaper_text,
                "title": newspaper_title,
                "published_at": newspaper_published_at,
                "language": "en",
                "authors": newspaper_authors
            }
        
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
                    if text is None:
                        text = ""
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
                    if text is None:
                        text = ""
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
        
        # First, try to extract from JSON-LD structured data (common in modern news sites)
        json_ld_text = self._extract_from_json_ld(soup)
        if json_ld_text and len(json_ld_text) > 500:
            return json_ld_text
        
        # Remove unwanted elements
        for unwanted in soup.select('script, style, nav, header, footer, .ads, .comments, .sidebar, .social-share, .related-posts'):
            unwanted.decompose()
        
        # Common article content selectors (expanded list)
        content_selectors = [
            'article', '.article', '.post', '.content', '.story', '.entry',
            '.article-content', '.article-body', '.entry-content', '.post-content',
            '.story-content', '.story-body', '.main-content', '#article-body',
            '.news-content', '.article__content', '.article__body',
            '.entry-content', '.post-body', '.content-area', '.main-article',
            '.article-text', '.story-text', '.news-text', '.content-body',
            'main article',
            '[role="main"]',
            '.news-article',
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
        if text is None:
            text = ""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return '\n\n'.join(lines)
    
    def _extract_from_json_ld(self, soup: BeautifulSoup) -> str:
        """Extract article content from JSON-LD structured data"""
        try:
            import json
            
            # Find all script tags with type="application/ld+json"
            json_ld_scripts = soup.find_all('script', type='application/ld+json')
            
            for script in json_ld_scripts:
                try:
                    data = json.loads(script.string)
                    
                    # Handle both single objects and arrays
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and item.get('@type') == 'NewsArticle':
                                article_body = item.get('articleBody', '')
                                if article_body and len(article_body) > 200:
                                    # Clean up the text - replace HTML entities and normalize
                                    article_body = article_body.replace('&quot;', '"').replace('&amp;', '&')
                                    article_body = article_body.replace('..', '.')  # Fix double periods
                                    return article_body
                    elif isinstance(data, dict) and data.get('@type') == 'NewsArticle':
                        article_body = data.get('articleBody', '')
                        if article_body and len(article_body) > 200:
                            # Clean up the text - replace HTML entities and normalize
                            article_body = article_body.replace('&quot;', '"').replace('&amp;', '&')
                            article_body = article_body.replace('..', '.')  # Fix double periods
                            return article_body
                            
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
                    
        except Exception as e:
            # If anything fails, return empty string
            pass
            
        return ""
    
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

    def _extract_published_date_from_html(self, html: str) -> Optional[str]:
        """Extract published date from HTML"""
        if not BEAUTIFULSOUP_AVAILABLE:
            return None
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Try different date selectors and attributes
        date_selectors = [
            'time[datetime]',
            'time[pubdate]',
            '.published',
            '.date',
            '.publish-date',
            '.article-date',
            '.post-date',
            '.entry-date',
            '[property="article:published_time"]',
            '[name="article:published_time"]',
            '[property="og:published_time"]',
            '[name="twitter:card"]',
            'meta[property="article:published_time"]',
            'meta[name="article:published_time"]',
            'meta[property="og:published_time"]'
        ]
        
        for selector in date_selectors:
            element = soup.select_one(selector)
            if element:
                # Try datetime attribute first
                date_str = element.get('datetime') or element.get('pubdate')
                if date_str:
                    try:
                        from datetime import datetime
                        # Try to parse the date
                        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        return dt.strftime('%Y-%m-%d')
                    except:
                        pass
                
                # Try text content
                date_text = element.get_text().strip()
                if date_text:
                    try:
                        from datetime import datetime
                        # Try common date formats
                        for fmt in ['%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%SZ', '%B %d, %Y', '%d %B %Y', '%Y-%m-%d %H:%M:%S']:
                            try:
                                dt = datetime.strptime(date_text, fmt)
                                return dt.strftime('%Y-%m-%d')
                            except:
                                continue
                    except:
                        pass
        
        # Try to find date patterns in the HTML content
        date_patterns = [
            r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
            r'(\d{4}/\d{2}/\d{2})',  # YYYY/MM/DD
            r'(\d{2}/\d{2}/\d{4})',  # MM/DD/YYYY
            r'(\d{1,2}\s+\w+\s+\d{4})',  # DD Month YYYY
            r'(\w+\s+\d{1,2},?\s+\d{4})',  # Month DD, YYYY
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, html)
            if matches:
                try:
                    from datetime import datetime
                    for match in matches:
                        # Try to parse the date
                        for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d %B %Y', '%B %d, %Y', '%B %d %Y']:
                            try:
                                dt = datetime.strptime(match, fmt)
                                return dt.strftime('%Y-%m-%d')
                            except:
                                continue
                except:
                    pass
        
        return None

    def _extract_published_date_from_json_ld(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract published date from JSON-LD structured data"""
        try:
            import json
            from datetime import datetime
            
            # Find all script tags with type="application/ld+json"
            json_ld_scripts = soup.find_all('script', type='application/ld+json')
            
            for script in json_ld_scripts:
                try:
                    data = json.loads(script.string)
                    
                    # Handle both single objects and arrays
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and item.get('@type') == 'NewsArticle':
                                date_published = item.get('datePublished')
                                if date_published:
                                    try:
                                        # Try to parse ISO format
                                        dt = datetime.fromisoformat(date_published.replace('Z', '+00:00'))
                                        return dt.strftime('%Y-%m-%d')
                                    except:
                                        pass
                    elif isinstance(data, dict) and data.get('@type') == 'NewsArticle':
                        date_published = data.get('datePublished')
                        if date_published:
                            try:
                                # Try to parse ISO format
                                dt = datetime.fromisoformat(date_published.replace('Z', '+00:00'))
                                return dt.strftime('%Y-%m-%d')
                            except:
                                pass
                            
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
                    
        except Exception:
            pass
            
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
        if not text or len(text) < 100:  # Reduced from 200
            return False
        
        # Check for error phrases
        text_lower = text.lower()
        for phrase in self.error_phrases:
            if phrase in text_lower:
                return False
        
        # Check for reasonable word count
        words = text.split()
        if len(words) < 10:  # Reduced from 20
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
