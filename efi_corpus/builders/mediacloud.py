"""
MediaCloudCorpusBuilder - Corpus builder for MediaCloud data
"""

from pathlib import Path
from typing import Iterable, Dict, Any
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
        Discover articles using MediaCloud queries
        """
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

    def fetch_raw(self, url: str, stable_id: str) -> tuple[bytes, Dict[str, Any], str]:
        """
        Fetch raw content using the global fetcher cache
        """
        blob_id, blob_path, fetch_meta = self.fetcher.get(url, url)

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
