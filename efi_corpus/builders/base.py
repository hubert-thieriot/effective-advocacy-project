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
        self.corpus = CorpusHandle(corpus_dir)
        
        # Use provided fetcher or create default one
        if fetcher is not None:
            self.fetcher = fetcher
        else:
            if cache_root is None:
                cache_root = Path("cache")
            self.fetcher = Fetcher(cache_root)

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
        manifest.setdefault("name", self.corpus.dir.name)
        manifest.setdefault("source", self.__class__.__name__.replace("Builder", "").lower())
        manifest["params"] = params.__dict__
        manifest.setdefault("history", [])

        # Discover items
        discovered = list(self.discover(params))
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
        
        for url, stable_id in frontier:
            try:
                print(f"Processing URL: {url}")
                
                raw_bytes, fetch_meta, raw_ext = self.fetch_raw(url, stable_id)
                parsed = self.parse_text(raw_bytes, raw_ext, url)
                text = parsed.get("text") or ""
                
                # Quality gate
                if len(text) < 400:
                    print(f"  ⚠️  Skipped: Text too short ({len(text)} chars)")
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
            "discovered": len(discovered),
            "added": added,
            "failed": len(failed_urls),
            "date_from": params.date_from,
            "date_to": params.date_to,
            "keywords": params.keywords
        })
        manifest["doc_count"] = (manifest.get("doc_count", 0) + added)
        self.corpus.save_manifest(manifest)

        # Print summary
        if failed_urls:
            print(f"\n⚠️  {len(failed_urls)} URLs failed to process:")
            for failed in failed_urls[:5]:  # Show first 5 failures
                print(f"  - {failed['url']}: {failed['error']}")
            if len(failed_urls) > 5:
                print(f"  ... and {len(failed_urls) - 5} more")
        
        print(f"\n📊 Build Summary:")
        print(f"  Discovered: {len(discovered)}")
        print(f"  Added: {added}")
        print(f"  Failed: {len(failed_urls)}")
        print(f"  Total docs in corpus: {manifest['doc_count']}")

        return {
            "discovered": len(discovered), 
            "added": added, 
            "failed": len(failed_urls),
            "doc_count": manifest["doc_count"]
        }
