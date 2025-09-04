"""
MediaCloudCorpusBuilder - Corpus builder for MediaCloud data
"""

from pathlib import Path
from typing import Iterable, Dict, Any
import time
import zstandard as zstd
from decouple import config
from mediacloud import api

from .base import BaseCorpusBuilder
from ..types import BuilderParams, DiscoveryItem
from ..fetcher import Fetcher
from ..rate_limiter import RateLimiter, RateLimitConfig
from ..text_extractor import TextExtractor
from ..utils import ensure_date


class MediaCloudCorpusBuilder(BaseCorpusBuilder):
    """Corpus builder that integrates with MediaCloud for content discovery"""
    
    def __init__(self, corpus_dir: Path, collection_id: int = None, collection_name: str = None, 
                 rate_limit_config: RateLimitConfig = None, fetcher: Fetcher = None, cache_root: Path = None):
        super().__init__(corpus_dir, fetcher, cache_root)
        self.collection_id = collection_id
        self.collection_name = collection_name
        
        # Initialize rate limiter
        if rate_limit_config is None:
            rate_limit_config = RateLimitConfig()
        self.rate_limiter = RateLimiter(rate_limit_config)
        
        # Initialize text extractor
        self.text_extractor = TextExtractor()
        
        # Initialize MediaCloud API (support alternate env var name)
        api_key = config('MEDIACLOUD_API_KEY', default=None) or config('MEDIACLOUD_KEY', default=None)
        if not api_key:
            # Keep message aligned with tests expectation
            raise ValueError("MEDIACLOUD_KEY environment variable is required")
        self.mc_api = api.SearchApi(api_key)

    def discover(self, params: BuilderParams) -> Iterable[DiscoveryItem]:
        """
        Discover articles using MediaCloud queries or test URLs
        """
        # Check if test URLs are provided (for testing)
        test_urls = (params.extra or {}).get('test_urls')
        if test_urls:
            print(f"Using test URLs instead of MediaCloud API: {len(test_urls)} URLs")
            for i, url in enumerate(test_urls):
                print(f"Test URL {i+1}: {url}")
                yield DiscoveryItem(
                    url=url,
                    canonical_url=url,
                    title=f"Test Article {i+1}",
                    published_at=params.date_from,  # Use date_from as fallback
                    extra={"source": "test"}
                )
            return
        
        # Prepare list of query strings based on params/extra
        queries = self._prepare_queries(params)
        
        # Use collection_id from constructor or from params.extra
        collection_id = self.collection_id or (params.extra or {}).get('collection_id')
        if not collection_id:
            raise ValueError("collection_id must be provided either in constructor or params.extra")
        
        collection_name = self.collection_name or (params.extra or {}).get('collection_name', str(collection_id))
        
        print(f"Querying MediaCloud in collection {collection_name} ({collection_id})")
        print(f"Date range: {params.date_from} to {params.date_to}")

        # Normalize parameters for API
        start_date = ensure_date(params.date_from, "date_from")
        end_date = ensure_date(params.date_to, "date_to")

        try:
            collection_id_int = int(collection_id)
        except (TypeError, ValueError):
            collection_id_int = collection_id
        
        # Query MediaCloud for stories (aggregate across all queries)
        all_stories = []
        max_stories = (params.extra or {}).get('max_stories')
        
        for q in queries:
            print(f"{collection_name}: Running query: {q}")
            pagination_token = None
            more_stories = True
            while more_stories:
                print(f"{collection_name}: {len(all_stories)} stories retrieved so far")
                
                # Check if we've reached the max stories limit
                if max_stories and len(all_stories) >= max_stories:
                    print(f"{collection_name}: Reached max stories limit ({max_stories}), stopping pagination")
                    more_stories = False
                    break
                
                try:
                    pages, pagination_token = self.mc_api.story_list(
                        q,
                        collection_ids=[collection_id_int],
                        start_date=start_date,
                        end_date=end_date,
                        pagination_token=pagination_token
                    )
                    
                    if not pages:
                        print(f"{collection_name}: No more stories returned, ending pagination for query")
                        more_stories = False
                    else:
                        # Add collection name to each story (create a copy to avoid modifying the original)
                        for story in pages:
                            story_copy = story.copy()
                            story_copy['collection'] = collection_name
                            story_copy['collection_id'] = collection_id
                            all_stories.append(story_copy)
                            
                            # Check if we've reached the max stories limit after adding this story
                            if max_stories and len(all_stories) >= max_stories:
                                print(f"{collection_name}: Reached max stories limit ({max_stories}) after adding story")
                                more_stories = False
                                break
                        
                        if more_stories:
                            more_stories = pagination_token is not None
                            if not more_stories:
                                print(f"{collection_name}: Reached the end of pagination for query")
                            
                except Exception as e:
                    error_str = str(e)
                    print(f"{collection_name}: Error during API call: {e}")
                    
                    # Handle specific error types with different strategies
                    if "403" in error_str:
                        print(f"{collection_name}: Received 403 error, waiting 60 seconds before retrying...")
                        self.rate_limiter.wait_for_retry(60)
                        continue
                    elif "timeout" in error_str.lower() or "read timed out" in error_str.lower():
                        print(f"{collection_name}: Timeout error, waiting 30 seconds before retrying...")
                        self.rate_limiter.wait_for_retry(30)
                        continue
                    elif "connection" in error_str.lower() or "network" in error_str.lower():
                        print(f"{collection_name}: Connection error, waiting 45 seconds before retrying...")
                        self.rate_limiter.wait_for_retry(45)
                        continue
                    else:
                        # For other errors, try one more time after a short delay
                        print(f"{collection_name}: Unknown error, waiting 15 seconds before retrying...")
                        self.rate_limiter.wait_for_retry(15)
                        continue
                
                # Be polite to the API - use rate limiter
                self.rate_limiter.wait_if_needed()
        
        print(f"Total stories discovered: {len(all_stories)}")
        
        # Convert MediaCloud stories to DiscoveryItem objects
        for story in all_stories:
            try:
                # Extract URL - MediaCloud stories can have multiple URLs
                url = story.get('url') or story.get('guid', '')
                if not url:
                    continue
                    
                # Extract published date
                published_at = story.get('publish_date')
                if published_at:
                    # Convert MediaCloud timestamp to ISO format
                    from datetime import datetime
                    try:
                        dt = datetime.fromtimestamp(published_at)
                        published_at = dt.strftime('%Y-%m-%d')
                    except (ValueError, TypeError):
                        published_at = None
                
                # Extract title
                title = story.get('title', '').strip()
                
                # Extract language
                language = story.get('language')
                
                # Extract authors (MediaCloud doesn't always have this)
                authors = []
                if story.get('author'):
                    authors = [story['author']]
                
                yield DiscoveryItem(
                    url=url,
                    canonical_url=url,  # We'll let the fetcher handle canonicalization
                    published_at=published_at,
                    title=title,
                    language=language,
                    authors=authors,
                    extra={
                        'story_id': story.get('stories_id'),
                        'collection': story.get('collection'),
                        'collection_id': story.get('collection_id'),
                        'source_language': story.get('language'),
                        'media_id': story.get('media_id'),
                        'media_name': story.get('media_name'),
                        'media_url': story.get('media_url'),
                    }
                )
            except Exception as e:
                print(f"⚠️  Failed to process story {story.get('stories_id', 'unknown')}: {e}")
                continue  # Skip this story and continue with the next one

    def _prepare_queries(self, params: BuilderParams) -> list[str]:
        """Prepare MediaCloud query strings from params/extra.
        Precedence:
        - params.extra['queries'] as list[str]
        - params.extra['keywords'] as dict[str, list[str]] (compile to a single OR-combined query)
        - params.keywords as list[str] (compile to OR across terms)
        - otherwise, empty list
        """
        extra = params.extra or {}
        # Raw queries list provided
        
        if isinstance(extra.get('queries'), list) and extra.get('queries'):
            return [str(q).strip() for q in extra['queries'] if str(q).strip()]
        # Keywords mapping provided
        keywords_map = extra.get('keywords')
        if isinstance(keywords_map, dict) and keywords_map:
            return [self._compile_keywords_map_to_query(keywords_map)]
        # Simple keywords list on params
        if isinstance(params.keywords, list) and params.keywords:
            terms = [self._quote_term(t) for t in params.keywords if str(t).strip()]
            if not terms:
                return []
            return [f"({' OR '.join(terms)})"]
        return []

    @staticmethod
    def _quote_term(term: str) -> str:
        t = str(term).strip()
        # If already looks like a structured query, trust caller
        if any(op in t.upper() for op in [" OR ", " AND ", "(", ")"]):
            return t
        # If already quoted, return as-is
        if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
            return t
        # Otherwise, quote and escape internal quotes
        t = t.replace('"', '\\"')
        return f'"{t}"'

    @staticmethod
    def _compile_keywords_map_to_query(keywords_by_lang: dict[str, list[str]]) -> str:
        groups: list[str] = []
        for _, words in keywords_by_lang.items():
            terms = [MediaCloudCorpusBuilder._quote_term(w) for w in words if str(w).strip()]
            if terms:
                groups.append(f"({' OR '.join(terms)})")
        if not groups:
            return ""
        # OR across language groups (can later support AND patterns if needed)
        return f"({' OR '.join(groups)})"

    def fetch_raw(self, url: str, stable_id: str, force_refresh: bool = False) -> tuple[bytes, Dict[str, Any], str]:
        """
        Fetch raw content using the global fetcher cache

        Args:
            url: URL to fetch
            stable_id: Stable ID for the URL
            force_refresh: If True, bypass cache and fetch fresh content
        """
        blob_id, blob_path, fetch_meta = self.fetcher.get(url, url, force_refresh=force_refresh)

        # Read blob as stored in cache. Fetcher stores zstd-compressed bytes.
        with open(blob_path, 'rb') as f:
            cbytes = f.read()
        # Some tests may simulate uncompressed content; try decompress, fall back to raw
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
        
        return raw_bytes, fetch_meta, raw_ext

    def parse_text(self, raw_bytes: bytes, raw_ext: str, url: str) -> Dict[str, Any]:
        """
        Parse raw content into structured text and metadata using the text extractor
        """
        return self.text_extractor.extract_text(raw_bytes, raw_ext, url)

    # ---------- utility methods ----------
    def _is_domain_blacklisted(self, url: str, blacklist: list[str]) -> bool:
        """Check if a URL's domain is in the blacklist"""
        from urllib.parse import urlparse
        try:
            domain = urlparse(url).netloc.lower()
            return any(blacklisted_domain.lower() in domain for blacklisted_domain in blacklist)
        except Exception:
            # If URL parsing fails, assume it's not blacklisted
            return False

    def _is_url_blacklisted(self, url: str, blacklist: list[str]) -> bool:
        """Check if a URL contains any blacklisted patterns"""
        try:
            url_lower = url.lower()
            return any(pattern.lower() in url_lower for pattern in blacklist)
        except Exception:
            # If URL processing fails, assume it's not blacklisted
            return False

    def _apply_filters(self, items: list[DiscoveryItem], params: BuilderParams) -> list[DiscoveryItem]:
        """Apply common filtering logic to discovered items"""
        domain_blacklist = (params.extra or {}).get('domain_blacklist', [])
        url_blacklist = (params.extra or {}).get('url_blacklist', [])

        if domain_blacklist:
            original_count = len(items)
            items = [item for item in items if not self._is_domain_blacklisted(item.url, domain_blacklist)]
            filtered_count = original_count - len(items)
            if filtered_count > 0:
                print(f"Filtered out {filtered_count} items from blacklisted domains")

        if url_blacklist:
            original_count = len(items)
            items = [item for item in items if not self._is_url_blacklisted(item.url, url_blacklist)]
            filtered_count = original_count - len(items)
            if filtered_count > 0:
                print(f"Filtered out {filtered_count} items with blacklisted URL patterns")

        return items

    # ---------- main run method ----------
    def run(self, *, params: BuilderParams | None = None, override: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Run the corpus builder

        If params is None, load from manifest. If override provided, overlay keys.
        Idempotent: only adds new docs.
        """
        from hashlib import sha1
        import time

        # Common tracking parameters to strip from URLs
        TRACKERS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "gclid", "fbclid"}

        def canonicalize(url: str) -> str:
            """Canonicalize a URL by removing tracking parameters and normalizing"""
            from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
            p = urlparse(url.strip())
            q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if k not in TRACKERS]
            q.sort()
            return urlunparse((
                p.scheme.lower(),
                (p.hostname or "").lower() or "",
                p.path or "/",
                "",
                urlencode(q, doseq=True),
                ""
            ))

        manifest = self.corpus.load_manifest()
        if params is None:
            if not manifest:
                raise ValueError("No manifest found; first run must provide params.")
            from ..types import BuilderParams
            params = BuilderParams(**manifest["params"])
        if override:
            # Override selected fields (keywords, date_from, date_to, extra)
            merged = {**params.__dict__, **override}
            params = BuilderParams(**merged)

        # Persist (current) params to manifest before run
        manifest.setdefault("name", self.corpus.corpus_path.name)
        manifest.setdefault("source", self.__class__.__name__.replace("Builder", "").lower())
        manifest["params"] = params.__dict__
        manifest.setdefault("history", [])

        # Discover items
        discovered = list(self.discover(params))

        # Apply filtering
        discovered = self._apply_filters(discovered, params)

        # Map URL to discovery item for later metadata enrichment
        discovered_by_url = {d.url: d for d in discovered}

        # Compute frontier (items not already in corpus)
        def sid(u: str) -> str:
            return sha1(canonicalize(u).encode("utf-8")).hexdigest()

        pairs = [(d.url, sid(d.url)) for d in discovered]
        frontier = [(u, s) for (u, s) in pairs if not self.corpus.has_doc(s)]

        # Choose processing strategy based on config (default to concurrent)
        use_concurrent = (params.extra or {}).get('use_concurrent_processing', True)

        # Check if we should force refresh (disable cache)
        force_refresh = (params.extra or {}).get('force_refresh_cache', False)

        if use_concurrent:
            return self._process_concurrent(frontier, discovered_by_url, params, manifest, force_refresh)
        else:
            return self._process_sequential(frontier, discovered_by_url, params, manifest, force_refresh)

    def _process_sequential(self, frontier, discovered_by_url, params, manifest, force_refresh: bool = False) -> Dict[str, Any]:
        """Process items sequentially (current approach)"""
        import time

        # Process frontier
        added = 0
        failed_urls = []
        skipped_quality = 0
        skipped_text_extraction = 0
        skipped_duplicate = len(discovered_by_url) - len(frontier)  # Already in corpus

        for url, stable_id in frontier:
            try:
                print(f"Processing URL: {url}")

                raw_bytes, fetch_meta, raw_ext = self.fetch_raw(url, stable_id, force_refresh=force_refresh)
                parsed = self.parse_text(raw_bytes, raw_ext, url)
                text = parsed.get("text") or ""

                # Track text extraction failures
                if not text:
                    print(f"  ⚠️  Skipped: No text extracted")
                    skipped_text_extraction += 1
                    continue

                # Quality gate
                if len(text) < 400:
                    print(f"  ⚠️  Skipped: Text too short ({len(text)} chars)")
                    skipped_quality += 1
                    continue

                # Merge extras: run-level extras and per-item extras
                discovered_item = discovered_by_url.get(url)
                per_item_extra = (discovered_item.extra if discovered_item else {}) or {}
                merged_extra = {**(params.extra or {}), **per_item_extra}

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
                    "source": manifest["source"],
                    "keywords": params.keywords,
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
                    "keywords": params.keywords,
                    "collection_id": merged_extra.get("collection_id"),
                    "collection": merged_extra.get("collection") or merged_extra.get("collection_name"),
                })
                added += 1
                print(f"  ✅ Added successfully")

            except Exception as e:
                print(f"  ❌ Failed to process URL: {url}")
                print(f"     Error: {e}")
                failed_urls.append({"url": url, "error": str(e)})
                continue  # Continue with next URL instead of crashing

        # Update manifest
        manifest["history"].append({
            "run_at": time.time(),
            "discovered": len(discovered_by_url),
            "added": added,
            "skipped_quality": skipped_quality,
            "skipped_text_extraction": skipped_text_extraction,
            "skipped_duplicate": skipped_duplicate,
            "failed": len(failed_urls),
            "date_from": params.date_from,
            "date_to": params.date_to,
            "keywords": params.keywords
        })
        manifest["doc_count"] = (manifest.get("doc_count", 0) + added)
        self.corpus.save_manifest(manifest)

        # Print comprehensive summary
        print(f"\n📊 Build Summary:")
        print(f"  Discovered: {len(discovered_by_url)}")
        print(f"  Added: {added}")
        print(f"  Skipped (quality): {skipped_quality}")
        print(f"  Skipped (text extraction): {skipped_text_extraction}")
        print(f"  Skipped (duplicate): {skipped_duplicate}")
        print(f"  Failed: {len(failed_urls)}")
        print(f"  Total docs in corpus: {self.corpus.get_document_count()}")

        # Print detailed failure summary
        if failed_urls:
            print(f"\n⚠️  {len(failed_urls)} URLs failed to process:")
            for failed in failed_urls[:5]:  # Show first 5 failures
                print(f"  - {failed['url']}: {failed['error']}")
            if len(failed_urls) > 5:
                print(f"  ... and {len(failed_urls) - 5} more")

        return {
            "discovered": len(discovered_by_url),
            "added": added,
            "skipped_quality": skipped_quality,
            "skipped_text_extraction": skipped_text_extraction,
            "skipped_duplicate": skipped_duplicate,
            "failed": len(failed_urls),
            "total_docs": self.corpus.get_document_count(),
            "failed_details": failed_urls
        }

    def _process_concurrent(self, frontier, discovered_by_url, params, manifest, force_refresh: bool = False) -> Dict[str, Any]:
        """Process items concurrently using Scrapy"""
        print("🕷️  Starting concurrent processing with Scrapy...")

        # Extract URLs from frontier
        urls = [url for url, _ in frontier]

        # Get configuration for concurrent processing
        concurrent_requests = (params.extra or {}).get('concurrent_requests', 16)
        download_delay = (params.extra or {}).get('download_delay', 0.1)

        print(f"Concurrent settings: {concurrent_requests} requests, {download_delay}s delay")

        try:
            # Import the Scrapy spider
            from .scrapy_spider import run_scrapy_spider

            # Run the Scrapy spider
            result = run_scrapy_spider(
                urls=urls,
                fetcher=self.fetcher,
                text_extractor=self.text_extractor,
                corpus=self.corpus,
                params=params,
                manifest=manifest,
                concurrent_requests=concurrent_requests,
                download_delay=download_delay,
                force_refresh=force_refresh
            )

            # Update manifest with results
            manifest["history"].append({
                "run_at": time.time(),
                "discovered": len(discovered_by_url),
                "added": result["added"],
                "skipped_quality": result["skipped_quality"],
                "skipped_text_extraction": result["skipped_text_extraction"],
                "skipped_duplicate": len(discovered_by_url) - len(frontier),
                "failed": result["failed"],
                "date_from": params.date_from,
                "date_to": params.date_to,
                "keywords": params.keywords,
                "processing_mode": "concurrent",
                "concurrent_requests": concurrent_requests,
                "download_delay": download_delay
            })
            manifest["doc_count"] = (manifest.get("doc_count", 0) + result["added"])
            self.corpus.save_manifest(manifest)

            return result

        except Exception as e:
            import traceback
            print(f"❌ Scrapy processing failed: {e}")
            print(f"❌ Traceback: {traceback.format_exc()}")
            print("Falling back to sequential processing...")
            return self._process_sequential(frontier, discovered_by_url, params, manifest, force_refresh)
