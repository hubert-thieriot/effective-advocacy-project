"""
CREA publications URL collector
"""

import logging
import requests
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from .base_collector import BaseURLCollector

logger = logging.getLogger(__name__)


class CREAPublicationsCollector(BaseURLCollector):
    """Collect URLs from CREA publications page"""
    
    def __init__(self, max_sources: Optional[int] = None, base_url: str = "https://energyandcleanair.org"):
        """
        Initialize CREA collector
        
        Args:
            max_sources: Maximum number of sources to collect
            base_url: Base URL for CREA website
        """
        super().__init__("crea_publications", max_sources)
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; EFI-Findings/1.0)'
        })
    
    def collect_urls(self) -> List[str]:
        """
        Collect publication URLs from CREA website
        
        Returns:
            List of publication URLs
        """
        logger.info("Collecting CREA publication URLs...")
        
        publications_urls = []
        
        try:
            # Start with the main publications page
            publications_page = f"{self.base_url}/publications/"
            response = self.session.get(publications_page, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find publication links
            publication_links = self._find_publication_links(soup, publications_page)
            publications_urls.extend(publication_links)
            
            # Look for pagination and additional pages
            additional_pages = self._find_additional_pages(soup, publications_page)
            for page_url in additional_pages:
                try:
                    page_response = self.session.get(page_url, timeout=30)
                    page_response.raise_for_status()
                    page_soup = BeautifulSoup(page_response.content, 'html.parser')
                    
                    page_links = self._find_publication_links(page_soup, page_url)
                    publications_urls.extend(page_links)
                    
                except Exception as e:
                    logger.warning(f"Error processing additional page {page_url}: {e}")
            
            # Check additional publication sections
            additional_sections = self._get_additional_publication_sections()
            for section_url in additional_sections:
                try:
                    section_response = self.session.get(section_url, timeout=30)
                    section_response.raise_for_status()
                    section_soup = BeautifulSoup(section_response.content, 'html.parser')
                    
                    section_links = self._find_publication_links(section_soup, section_url)
                    publications_urls.extend(section_links)
                    
                    # Handle pagination for weekly snapshot category (different pattern)
                    if 'weekly-snapshot' in section_url:
                        weekly_snapshot_pages = self._find_weekly_snapshot_pages(section_soup, section_url)
                        for weekly_page_url in weekly_snapshot_pages:
                            try:
                                weekly_page_response = self.session.get(weekly_page_url, timeout=30)
                                weekly_page_response.raise_for_status()
                                weekly_page_soup = BeautifulSoup(weekly_page_response.content, 'html.parser')
                                
                                weekly_page_links = self._find_publication_links(weekly_page_soup, weekly_page_url)
                                publications_urls.extend(weekly_page_links)
                                
                            except Exception as e:
                                logger.warning(f"Error processing weekly snapshot page {weekly_page_url}: {e}")
                    else:
                        # Use standard pagination for other sections
                        section_pages = self._find_additional_pages(section_soup, section_url)
                        for section_page_url in section_pages:
                            try:
                                section_page_response = self.session.get(section_page_url, timeout=30)
                                section_page_response.raise_for_status()
                                section_page_soup = BeautifulSoup(section_page_response.content, 'html.parser')
                                
                                section_page_links = self._find_publication_links(section_page_soup, section_page_url)
                                publications_urls.extend(section_page_links)
                                
                            except Exception as e:
                                logger.warning(f"Error processing section page {section_page_url}: {e}")
                    
                except Exception as e:
                    logger.warning(f"Error processing section {section_url}: {e}")
            
            # Remove duplicates while preserving order
            seen = set()
            unique_urls = []
            for url in publications_urls:
                if url not in seen:
                    seen.add(url)
                    unique_urls.append(url)
            
            logger.info(f"Collected {len(unique_urls)} unique publication URLs")
            return unique_urls
            
        except Exception as e:
            logger.error(f"Error collecting CREA publications: {e}")
            return []
    
    def _find_publication_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Find publication links in a page"""
        publication_links = []
        
        # Look for various types of publication links
        link_selectors = [
            'a[href*="/publication/"]',
            'a[href*="/report/"]',
            'a[href*="/analysis/"]',
            'a[href*="/study/"]',
            'a[href*="/research/"]',
            '.publication-link a',
            '.report-link a',
            'article a',
            '.entry-title a',
            '.gb-container-link',  # WordPress Grid Builder container links
            '.wpgb-card a',  # WordPress Grid Builder card links
            '.gb-query-loop-item a'  # Query loop item links
        ]
        
        for selector in link_selectors:
            links = soup.select(selector)
            for link in links:
                href = link.get('href')
                if href:
                    # Convert relative URLs to absolute
                    absolute_url = urljoin(base_url, href)
                    
                    # Check if it looks like a publication URL
                    if self._is_publication_url(absolute_url):
                        publication_links.append(absolute_url)
        
        # Also look for links in the WordPress Grid Builder structure
        # CREA uses specific classes for their publications
        grid_links = soup.select('.gb-query-loop-item .gb-container-link')
        for link in grid_links:
            href = link.get('href')
            if href:
                absolute_url = urljoin(base_url, href)
                if self._is_publication_url(absolute_url):
                    publication_links.append(absolute_url)
        
        return publication_links
    
    def _find_additional_pages(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Find additional pagination pages"""
        additional_pages = []
        
        # Look for pagination links
        pagination_selectors = [
            '.pagination a',
            '.page-numbers a',
            '.pager a',
            'nav a[href*="page"]',
            'a[href*="p="]',
            'a[href*="query-bf51d214-page"]',  # CREA custom pagination
            '.wpgb-pagination a',  # WordPress Grid Builder pagination
            '.wpgb-pagination-facet a'
        ]
        
        for selector in pagination_selectors:
            links = soup.select(selector)
            for link in links:
                href = link.get('href')
                if href and 'page' in href.lower():
                    absolute_url = urljoin(base_url, href)
                    if absolute_url not in additional_pages:
                        additional_pages.append(absolute_url)
        
        # Also try to generate pagination URLs based on CREA's pattern
        # The pattern is: /publications/?cst=&query-bf51d214-page=N
        crea_pagination_urls = self._generate_crea_pagination_urls(base_url)
        additional_pages.extend(crea_pagination_urls)
        
        return additional_pages
    
    def _generate_crea_pagination_urls(self, base_url: str) -> List[str]:
        """Generate CREA pagination URLs based on their custom pattern"""
        pagination_urls = []
        
        # Try to find the maximum number of pages by checking page 2, 3, etc.
        # CREA uses the pattern: /publications/?cst=&query-bf51d214-page=N
        for page_num in range(2, 21):  # Try up to 20 pages
            page_url = f"{base_url}/publications/?cst=&query-bf51d214-page={page_num}"
            pagination_urls.append(page_url)
        
        return pagination_urls
    
    def _get_additional_publication_sections(self) -> List[str]:
        """Get additional publication section URLs"""
        sections = [
            f"{self.base_url}/news/",
            f"{self.base_url}/news/?query-17bc7528-page=2",
            f"{self.base_url}/category/financing-putins-war/weekly-snapshot/"
        ]
        return sections
    
    def _find_weekly_snapshot_pages(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Find pagination pages for weekly snapshot category"""
        additional_pages = []
        
        # Look for pagination links in weekly snapshot category
        pagination_selectors = [
            'a[href*="page/"]',
            '.pagination a',
            'nav a[href*="page"]'
        ]
        
        for selector in pagination_selectors:
            links = soup.select(selector)
            for link in links:
                href = link.get('href')
                if href:
                    absolute_url = urljoin(base_url, href)
                    if 'weekly-snapshot' in absolute_url and 'page/' in absolute_url:
                        additional_pages.append(absolute_url)
        
        # Also generate pagination URLs based on the pattern we observed
        # Weekly snapshot uses /page/N/ pattern
        for page_num in range(2, 11):  # Try up to 10 pages
            page_url = f"{base_url}page/{page_num}/"
            additional_pages.append(page_url)
        
        return additional_pages
    
    def _is_publication_url(self, url: str) -> bool:
        """Check if a URL looks like a publication URL"""
        url_lower = url.lower()
        
        # Must be from the CREA domain
        if not url_lower.startswith(self.base_url.lower()):
            return False
        
        # Must contain publication-related paths
        publication_indicators = [
            '/publication/',
            '/report/',
            '/analysis/',
            '/study/',
            '/research/',
            '/weekly-snapshot'
        ]
        
        return any(indicator in url_lower for indicator in publication_indicators)
    
    def close(self):
        """Close the session"""
        self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
