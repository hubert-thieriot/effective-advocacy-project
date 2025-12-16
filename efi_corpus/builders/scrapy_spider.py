"""
Scrapy Spider for concurrent corpus building
"""

import scrapy
from scrapy.crawler import CrawlerProcess
from typing import List, Dict, Any, Tuple
import time
import random
from pathlib import Path
import hashlib
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

from ..fetcher import Fetcher
from ..text_extractor import TextExtractor
from ..corpus_handle import CorpusHandle
from ..types import DiscoveryItem, BuilderParams

# Import DateTimeEncoder from efi_core.utils
from efi_core.utils import DateTimeEncoder


class AntiDetectionMiddleware:
    """Custom middleware to handle anti-bot detection"""
    
    def __init__(self):
        self.user_agents = [
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
        ]
    
    def process_request(self, request, spider):
        """Add random user agent and additional headers"""
        request.headers['User-Agent'] = random.choice(self.user_agents)
        request.headers['Referer'] = 'https://www.google.com/'
        request.headers['DNT'] = '1'
        return None
    
    def process_response(self, request, response, spider):
        """Handle Access Denied responses"""
        if response.status in [403, 410]:
            # Check if it's an Access Denied page
            if b'Access Denied' in response.body or b'access denied' in response.body.lower():
                print(f"  🚫 Access Denied for {request.url} (Status: {response.status})")
                # Return a custom response that will be handled by the spider
                return response
        return response


class EFISpider(scrapy.Spider):
    """
    Scrapy spider for concurrent processing of URLs in corpus building
    """
    name = 'efi_spider'

    # Class variable to store results
    results = {}

    def __init__(self, urls: List[str], fetcher: Fetcher, text_extractor: TextExtractor,
                 corpus: CorpusHandle, params: BuilderParams, manifest: Dict[str, Any],
                 discovered_by_url: Dict[str, Any] = None, force_refresh: bool = False, 
                 result_key: str = None, url_timeout: int = 60, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.urls = urls
        self.fetcher = fetcher
        self.text_extractor = text_extractor
        self.corpus = corpus
        self.params = params
        self.manifest = manifest
        self.discovered_by_url = discovered_by_url or {}
        self.force_refresh = force_refresh
        self.result_key = result_key or 'default'
        self.url_timeout = url_timeout
        self.processed_count = 0
        self.added_count = 0
        self.failed_count = 0
        self.skipped_quality = 0
        self.skipped_text_extraction = 0
        self.failed_urls = []  # Track failed URLs with error details

        # Common tracking parameters to strip from URLs
        self.TRACKERS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "gclid", "fbclid"}

        # Initialize results for this spider instance
        EFISpider.results[self.result_key] = {
            'processed_count': 0,
            'added_count': 0,
            'failed_count': 0,
            'skipped_quality': 0,
            'skipped_text_extraction': 0,
            'failed_urls': []
        }

    def canonicalize(self, url: str) -> str:
        """Canonicalize a URL by removing tracking parameters and normalizing"""
        p = urlparse(url.strip())
        q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if k not in self.TRACKERS]
        q.sort()
        return urlunparse((
            p.scheme.lower(),
            (p.hostname or "").lower() or "",
            p.path or "/",
            "",
            urlencode(q, doseq=True),
            ""
        ))

    def start_requests(self):
        """Generate initial requests for all URLs"""
        print(f"🕷️  Starting to process {len(self.urls)} URLs...")
        for i, url in enumerate(self.urls):
            print(f"🕷️  Queuing URL {i+1}/{len(self.urls)}: {url}")
            yield scrapy.Request(
                url=url,
                callback=self.parse_article,
                meta={'original_url': url},
                dont_filter=True  # Allow duplicate URLs if needed
            )

    def parse_article(self, response):
        """Parse each article response"""
        url = response.meta['original_url']
        self.processed_count += 1

        try:
            print(f"Processing URL ({self.processed_count}/{len(self.urls)}): {url}")

            # Check for Access Denied responses
            if response.status in [403, 410]:
                if b'Access Denied' in response.body or b'access denied' in response.body.lower():
                    error_msg = f"Access Denied (Status: {response.status})"
                    print(f"  🚫 Access Denied for {url} (Status: {response.status})")
                    self.failed_count += 1
                    self.failed_urls.append({"url": url, "error": error_msg})
                    EFISpider.results[self.result_key]['failed_count'] += 1
                    EFISpider.results[self.result_key]['failed_urls'].append({"url": url, "error": error_msg})
                    return

            # Add timeout for individual URL processing
            import signal
            import time

            def timeout_handler(signum, frame):
                raise TimeoutError(f"URL processing timeout: {url}")

            # Set configurable timeout for individual URL processing
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.url_timeout)

            start_time = time.time()

            # Use existing fetcher to get content
            stable_id = hashlib.sha1(self.canonicalize(url).encode("utf-8")).hexdigest()

            # Get content using the existing fetcher (respect force_refresh setting)
            blob_id, blob_path, fetch_meta = self.fetcher.get(url, url, force_refresh=self.force_refresh)

            # Read the compressed blob
            with open(blob_path, 'rb') as f:
                import zstandard as zstd
                cbytes = f.read()

            # Decompress
            try:
                dctx = zstd.ZstdDecompressor()
                raw_bytes = dctx.decompress(cbytes)
            except zstd.ZstdError:
                raw_bytes = cbytes

            # Determine file extension based on content type
            mime_type = fetch_meta.get('mime', '')
            if 'html' in mime_type:
                raw_ext = 'html'
            elif 'pdf' in mime_type:
                raw_ext = 'pdf'
            else:
                raw_ext = 'bin'

            # Extract text using existing text extractor
            parsed = self.text_extractor.extract_text(raw_bytes, raw_ext, url)
            text = parsed.get("text") or ""

            # Track text extraction failures
            if not text:
                print(f"  ⚠️  Skipped: No text extracted")
                self.skipped_text_extraction += 1
                EFISpider.results[self.result_key]['skipped_text_extraction'] += 1
                return

            # Quality gate
            if len(text) < 200:  # Reduced from 400
                print(f"  ⚠️  Skipped: Text too short ({len(text)} chars)")
                self.skipped_quality += 1
                EFISpider.results[self.result_key]['skipped_quality'] += 1
                return

            # Get discovery item metadata (if available)
            discovered_item = self.discovered_by_url.get(url)

            # Merge extras: run-level extras and per-item extras
            merged_extra = self.params.extra or {}

            # Fallbacks from discovery when parser lacks metadata
            title = parsed.get("title") or (discovered_item.title if discovered_item else None)
            published_at = parsed.get("published_at") or (discovered_item.published_at if discovered_item else None)
            language = parsed.get("language") or (discovered_item.language if discovered_item else None)
            authors = parsed.get("authors", []) or ((discovered_item.authors or []) if discovered_item else [])

            meta = {
                "doc_id": stable_id,
                "uri": url,
                "title": title,
                "published_at": published_at,
                "language": language,
                "authors": authors,
                "source": self.manifest["source"],
                "keywords": self.params.keywords,
                "extra": merged_extra,
            }

            self.corpus.write_document(
                stable_id=stable_id,
                meta=meta,
                text=text,
                raw_bytes=raw_bytes,
                raw_ext=raw_ext,
                fetch_info=fetch_meta
            )

            self.corpus.append_index({
                "id": stable_id,
                "url": url,
                "published_at": meta["published_at"],
                "title": meta["title"],
                "language": meta["language"],
                "keywords": self.params.keywords,
                "collection_id": merged_extra.get("collection_id"),
                "collection": merged_extra.get("collection") or merged_extra.get("collection_name"),
            })

            self.added_count += 1
            EFISpider.results[self.result_key]['added_count'] += 1
            
            # Calculate processing time
            processing_time = time.time() - start_time
            print(f"  ✅ Added successfully ({self.added_count} total) in {processing_time:.1f}s")
            
            # Cancel timeout
            signal.alarm(0)

        except TimeoutError as e:
            error_msg = str(e)
            print(f"  ⏰ Timeout processing URL: {url}")
            print(f"     Error: {e}")
            self.failed_count += 1
            self.failed_urls.append({"url": url, "error": error_msg})
            EFISpider.results[self.result_key]['failed_count'] += 1
            EFISpider.results[self.result_key]['failed_urls'].append({"url": url, "error": error_msg})
            # Cancel timeout
            signal.alarm(0)
            return
            
        except Exception as e:
            error_msg = str(e)
            print(f"  ❌ Failed to process URL: {url}")
            print(f"     Error: {e}")
            import traceback
            print(f"     Traceback: {traceback.format_exc()}")
            self.failed_count += 1
            self.failed_urls.append({"url": url, "error": error_msg})
            EFISpider.results[self.result_key]['failed_count'] += 1
            EFISpider.results[self.result_key]['failed_urls'].append({"url": url, "error": error_msg})
            # Cancel timeout
            signal.alarm(0)
            # Continue processing other URLs
            return

    def closed(self, reason):
        """Called when spider finishes"""
        print(f"\n🕷️  Scrapy spider finished: {reason}")
        print(f"📊 Results: {self.added_count} added, {self.skipped_quality} skipped (quality), {self.skipped_text_extraction} skipped (text), {self.failed_count} failed")


def run_scrapy_spider(urls: List[str], fetcher: Fetcher, text_extractor: TextExtractor,
                     corpus: CorpusHandle, params: BuilderParams, manifest: Dict[str, Any],
                     discovered_by_url: Dict[str, Any] = None, concurrent_requests: int = 16, 
                     download_delay: float = 0.1, force_refresh: bool = False, url_timeout: int = 60) -> Dict[str, Any]:
    """
    Run Scrapy spider for concurrent URL processing

    Args:
        urls: List of URLs to process
        fetcher: The existing fetcher instance
        text_extractor: The existing text extractor
        corpus: The corpus handle
        params: Build parameters
        manifest: Corpus manifest
        concurrent_requests: Number of concurrent requests
        download_delay: Delay between requests
        force_refresh: If True, bypass cache

    Returns:
        Processing results summary
    """
    import random
    import warnings
    # Suppress Scrapy deprecation warnings
    warnings.filterwarnings('ignore', category=scrapy.exceptions.ScrapyDeprecationWarning)
    
    result_key = f"spider_{random.randint(100000, 999999)}"

    # Configure Scrapy settings
    settings = {
        'CONCURRENT_REQUESTS': concurrent_requests,
        'DOWNLOAD_DELAY': download_delay,
        'AUTOTHROTTLE_ENABLED': True,
        'AUTOTHROTTLE_START_DELAY': 1,
        'AUTOTHROTTLE_MAX_DELAY': 10,
        'ROBOTSTXT_OBEY': True,
        'USER_AGENT': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'LOG_LEVEL': 'WARNING',  # Reduced verbosity - only show warnings and errors
        'LOG_ENABLED': True,
        'LOG_STDOUT': False,  # Don't duplicate logs to stdout
        'DOWNLOAD_TIMEOUT': 30,  # 30 second timeout per request
        'RETRY_TIMES': 2,  # Retry failed requests up to 2 times
        'RETRY_HTTP_CODES': [500, 502, 503, 504, 408, 429, 403, 410],  # Retry on these HTTP codes including 403/410
        'DOWNLOADER_MIDDLEWARES': {
            'efi_corpus.builders.scrapy_spider.AntiDetectionMiddleware': 100,
            'scrapy.downloadermiddlewares.retry.RetryMiddleware': 90,
        },
        'DEFAULT_REQUEST_HEADERS': {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
        },
        'RETRY_PRIORITY_ADJUST': -1,
        'CLOSESPIDER_TIMEOUT': 1800,  # Stop spider after 30 minutes
        'CLOSESPIDER_PAGECOUNT': 1000,  # Stop spider after 1k pages
        'CLOSESPIDER_ITEMCOUNT': 100,  # Stop spider after 100 successful items
    }

    # Create and run the spider
    process = CrawlerProcess(settings)

    # Pass spider class and arguments separately
    process.crawl(
        EFISpider,
        urls=urls,
        fetcher=fetcher,
        text_extractor=text_extractor,
        corpus=corpus,
        params=params,
        manifest=manifest,
        discovered_by_url=discovered_by_url,
        force_refresh=force_refresh,
        result_key=result_key,
        url_timeout=url_timeout
    )
    process.start()

    # Get results from the class variable
    results = EFISpider.results.get(result_key, {
        'processed_count': len(urls),
        'added_count': 0,
        'failed_count': 0,
        'skipped_quality': 0,
        'skipped_text_extraction': 0,
        'failed_urls': []
    })

    # Clean up the results
    if result_key in EFISpider.results:
        del EFISpider.results[result_key]

    # Return summary
    return {
        "added_count": results["added_count"],
        "skipped_quality": results["skipped_quality"],
        "skipped_text_extraction": results["skipped_text_extraction"],
        "failed_count": results["failed_count"],
        "failed_details": results.get("failed_urls", []),
        "processed_count": results["processed_count"],
        "concurrent_requests": concurrent_requests,
        "download_delay": download_delay
    }
