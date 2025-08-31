"""
BaseCorpusBuilder - Abstract base class for all corpus builders
"""

from abc import ABC, abstractmethod
from hashlib import sha1
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
from pathlib import Path
from typing import Iterable, Dict, Any
import time

from ..types import BuilderParams, DiscoveryItem
from ..corpus_handle import CorpusHandle
from ..fetcher import Fetcher

# Common tracking parameters to strip from URLs
TRACKERS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "gclid", "fbclid"}


def canonicalize(url: str) -> str:
    """Canonicalize a URL by removing tracking parameters and normalizing"""
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


class BaseCorpusBuilder(ABC):
    """Abstract parent for all corpus builders"""
    
    def __init__(self, corpus_dir: Path, fetcher: Fetcher = None, cache_root: Path = None):
        self.corpus = CorpusHandle(corpus_dir, read_only=False)
        
        # Use provided fetcher or create default one
        if fetcher is not None:
            self.fetcher = fetcher
        else:
            if cache_root is None:
                cache_root = Path("cache")
            self.fetcher = Fetcher(cache_root)

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

    # ---------- abstract hooks ----------
    @abstractmethod
    def discover(self, params: BuilderParams) -> Iterable[DiscoveryItem]:
        """Discover items to be added to the corpus"""
        ...

    @abstractmethod
    def fetch_raw(self, url: str, stable_id: str) -> tuple[bytes, Dict[str, Any], str]:
        """
        Fetch raw content for a URL
        
        Returns:
            (raw_bytes, fetch_meta, raw_ext)
            raw_ext: e.g. "html.zst", "pdf"
        """
        ...

    @abstractmethod
    def parse_text(self, raw_bytes: bytes, raw_ext: str, url: str) -> Dict[str, Any]:
        """
        Parse raw content into structured data
        
        Returns:
            {"text": str, "title": str|None, "published_at": str|None,
             "language": str|None, "authors": list[str]|None}
        """
        ...

    # ---------- orchestration ----------
    def run(self, *, params: BuilderParams | None = None, override: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Run the corpus builder
        
        If params is None, load from manifest. If override provided, overlay keys.
        Idempotent: only adds new docs.
        """
        manifest = self.corpus.load_manifest()
        if params is None:
            if not manifest:
                raise ValueError("No manifest found; first run must provide params.")
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
        
        # Apply domain blacklist filtering if configured
        domain_blacklist = (params.extra or {}).get('domain_blacklist', [])
        print(f"DEBUG: Domain blacklist config: {domain_blacklist}")
        if domain_blacklist:
            original_count = len(discovered)
            discovered = [item for item in discovered if not self._is_domain_blacklisted(item.url, domain_blacklist)]
            filtered_count = original_count - len(discovered)
            if filtered_count > 0:
                print(f"Filtered out {filtered_count} items from blacklisted domains: {domain_blacklist}")
            else:
                print(f"DEBUG: No items were filtered out by domain blacklist")
        else:
            print(f"DEBUG: No domain blacklist configured")
        
        # Apply URL blacklist filtering if configured
        url_blacklist = (params.extra or {}).get('url_blacklist', [])
        print(f"DEBUG: URL blacklist config: {url_blacklist}")
        if url_blacklist:
            original_count = len(discovered)
            discovered = [item for item in discovered if not self._is_url_blacklisted(item.url, url_blacklist)]
            filtered_count = original_count - len(discovered)
            if filtered_count > 0:
                print(f"Filtered out {filtered_count} items with blacklisted URL patterns: {url_blacklist}")
            else:
                print(f"DEBUG: No items were filtered out by URL blacklist")
        else:
            print(f"DEBUG: No URL blacklist configured")
        
        # Map URL to discovery item for later metadata enrichment
        discovered_by_url = {d.url: d for d in discovered}
        
        # Compute frontier (items not already in corpus)
        def sid(u: str) -> str: 
            return sha1(canonicalize(u).encode("utf-8")).hexdigest()
        
        pairs = [(d.url, sid(d.url)) for d in discovered]
        frontier = [(u, s) for (u, s) in pairs if not self.corpus.has_doc(s)]

        # Process frontier
        added = 0
        failed_urls = []
        skipped_quality = 0
        skipped_text_extraction = 0
        skipped_duplicate = len(discovered) - len(frontier)  # Already in corpus
        
        for url, stable_id in frontier:
            try:
                print(f"Processing URL: {url}")
                
                raw_bytes, fetch_meta, raw_ext = self.fetch_raw(url, stable_id)
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
        
        # Print comprehensive summary
        print(f"\n📊 Build Summary:")
        print(f"  Discovered: {len(discovered)}")
        print(f"  Added: {added}")
        print(f"  Skipped (quality): {skipped_quality}")
        print(f"  Skipped (text extraction): {skipped_text_extraction}")
        print(f"  Skipped (duplicate): {skipped_duplicate}")
        print(f"  Failed: {len(failed_urls)}")
        print(f"  Total docs in corpus: {self.corpus.get_document_count()}")
        
        # Calculate total processed (should equal discovered - duplicates)
        total_processed = added + skipped_quality + skipped_text_extraction + len(failed_urls)
        print(f"  Total processed: {total_processed}")
        
        # Verify numbers add up
        expected_total = len(discovered) - skipped_duplicate
        if total_processed != expected_total:
            print(f"  ⚠️  Number mismatch: processed {total_processed} vs expected {expected_total}")
        
        # Update manifest
        manifest["history"].append({
            "run_at": time.time(),
            "discovered": len(discovered),
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
        
        # Print detailed failure summary
        if failed_urls:
            print(f"\n⚠️  {len(failed_urls)} URLs failed to process:")
            for failed in failed_urls[:5]:  # Show first 5 failures
                print(f"  - {failed['url']}: {failed['error']}")
            if len(failed_urls) > 5:
                print(f"  ... and {len(failed_urls) - 5} more")
        
        return {
            "discovered": len(discovered),
            "added": added,
            "skipped_quality": skipped_quality,
            "skipped_text_extraction": skipped_text_extraction,
            "skipped_duplicate": skipped_duplicate,
            "failed": len(failed_urls),
            "total_docs": self.corpus.get_document_count(),
            "failed_details": failed_urls
        }
    
    def run_from_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the corpus builder using a config dictionary
        
        Args:
            config: Configuration dictionary loaded from YAML/JSON
            
        Returns:
            Summary of the build process
        """
        # Extract parameters from config
        params_cfg = config.get("parameters", {})
        
        # Create BuilderParams
        params = BuilderParams(
            keywords=params_cfg.get("keywords", []),
            date_from=params_cfg.get("date_from", ""),
            date_to=params_cfg.get("date_to", ""),
            extra=params_cfg
        )
        
        # Run the builder
        return self.run(params=params)
