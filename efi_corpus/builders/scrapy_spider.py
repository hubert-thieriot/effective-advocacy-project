"""
Scrapy Spider for concurrent corpus building
"""

import scrapy
from scrapy.crawler import CrawlerProcess
from typing import List, Dict, Any, Tuple
import time
from pathlib import Path
import hashlib
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

from ..fetcher import Fetcher
from ..text_extractor import TextExtractor
from ..corpus_handle import CorpusHandle
from ..types import DiscoveryItem, BuilderParams

# Import DateTimeEncoder from efi_core.utils
from efi_core.utils import DateTimeEncoder


class EFISpider(scrapy.Spider):
    """
    Scrapy spider for concurrent processing of URLs in corpus building
    """
    name = 'efi_spider'

    # Class variable to store results
    results = {}

    def __init__(self, urls: List[str], fetcher: Fetcher, text_extractor: TextExtractor,
                 corpus: CorpusHandle, params: BuilderParams, manifest: Dict[str, Any],
                 force_refresh: bool = False, result_key: str = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.urls = urls
        self.fetcher = fetcher
        self.text_extractor = text_extractor
        self.corpus = corpus
        self.params = params
        self.manifest = manifest
        self.force_refresh = force_refresh
        self.result_key = result_key or 'default'
        self.processed_count = 0
        self.added_count = 0
        self.failed_count = 0
        self.skipped_quality = 0
        self.skipped_text_extraction = 0

        # Common tracking parameters to strip from URLs
        self.TRACKERS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "gclid", "fbclid"}

        # Initialize results for this spider instance
        EFISpider.results[self.result_key] = {
            'processed_count': 0,
            'added_count': 0,
            'failed_count': 0,
            'skipped_quality': 0,
            'skipped_text_extraction': 0
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
        for url in self.urls:
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
            if len(text) < 400:
                print(f"  ⚠️  Skipped: Text too short ({len(text)} chars)")
                self.skipped_quality += 1
                EFISpider.results[self.result_key]['skipped_quality'] += 1
                return

            # Get discovery item metadata (if available)
            # For now, we'll create basic metadata
            discovered_item = None  # We don't have this in concurrent mode

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
            print(f"  ✅ Added successfully ({self.added_count} total)")

        except Exception as e:
            print(f"  ❌ Failed to process URL: {url}")
            print(f"     Error: {e}")
            self.failed_count += 1
            EFISpider.results[self.result_key]['failed_count'] += 1

    def closed(self, reason):
        """Called when spider finishes"""
        print(f"\n🕷️  Scrapy spider finished: {reason}")
        print(f"📊 Results: {self.added_count} added, {self.skipped_quality} skipped (quality), {self.skipped_text_extraction} skipped (text), {self.failed_count} failed")


def run_scrapy_spider(urls: List[str], fetcher: Fetcher, text_extractor: TextExtractor,
                     corpus: CorpusHandle, params: BuilderParams, manifest: Dict[str, Any],
                     concurrent_requests: int = 16, download_delay: float = 0.1,
                     force_refresh: bool = False) -> Dict[str, Any]:
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
    result_key = f"spider_{random.randint(100000, 999999)}"

    # Configure Scrapy settings
    settings = {
        'CONCURRENT_REQUESTS': concurrent_requests,
        'DOWNLOAD_DELAY': download_delay,
        'AUTOTHROTTLE_ENABLED': True,
        'AUTOTHROTTLE_START_DELAY': 1,
        'AUTOTHROTTLE_MAX_DELAY': 10,
        'ROBOTSTXT_OBEY': True,
        'USER_AGENT': 'EFI-CorpusBuilder/1.0 (https://efi.org)',
        'LOG_LEVEL': 'WARNING',  # Reduce Scrapy logging
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
        force_refresh=force_refresh,
        result_key=result_key
    )
    process.start()

    # Get results from the class variable
    results = EFISpider.results.get(result_key, {
        'processed_count': len(urls),
        'added_count': 0,
        'failed_count': 0,
        'skipped_quality': 0,
        'skipped_text_extraction': 0
    })

    # Clean up the results
    if result_key in EFISpider.results:
        del EFISpider.results[result_key]

    # Return summary
    return {
        "added": results["added_count"],
        "skipped_quality": results["skipped_quality"],
        "skipped_text_extraction": results["skipped_text_extraction"],
        "failed": results["failed_count"],
        "total_processed": results["processed_count"],
        "concurrent_requests": concurrent_requests,
        "download_delay": download_delay
    }
