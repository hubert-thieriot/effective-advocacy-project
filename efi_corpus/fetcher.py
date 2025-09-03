"""
Fetcher - Manages content-addressed cache of fetched URLs
"""

import hashlib
import json
import time
import requests
import zstandard as zstd
from pathlib import Path
from typing import Tuple, Dict, Any, Union
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class Fetcher:
    """Manages content-addressed cache of fetched URLs"""
    
    def __init__(self, cache_root: Union[Path, str], ua: str = "EFI-CorpusFetcher/1.0", timeout: int = 30):
        # Convert cache_root to Path if it's a string
        if isinstance(cache_root, str):
            cache_root = Path(cache_root)
            
        self.cache = cache_root / "http"
        for sub in ("blobs", "meta", "map"):
            (self.cache / sub).mkdir(parents=True, exist_ok=True)
        self.ua = ua
        self.timeout = timeout
        
        # Create a session with retry logic
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,  # Total number of retries
            backoff_factor=1,  # Wait 1, 2, 4 seconds between retries
            status_forcelist=[429, 500, 502, 503, 504],  # Retry on these status codes
            allowed_methods=["GET", "HEAD"]  # Only retry safe methods
        )
        
        # Create adapter with retry strategy
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers
        self.session.headers.update({
            "User-Agent": self.ua,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        })

    def _blob_paths(self, blob_id: str) -> Tuple[Path, Path]:
        """Get the blob and meta file paths for a given blob_id"""
        return (
            self.cache / "blobs" / f"{blob_id}.bin.zst",
            self.cache / "meta" / f"{blob_id}.json"
        )

    def get(self, url: str, canon_url: str, force_refresh: bool = False) -> Tuple[str, Path, Dict[str, Any]]:
        """
        Fetch a URL and return (blob_id, blob_path, fetch_meta)
        
        - Checks cache/map for stable_id
        - Uses ETag/Last-Modified for conditional GET
        - Stores raw in blobs/<blob_id>.bin.zst (zstd)
        - Stores HTTP meta in meta/<blob_id>.json
        - Updates map/<stable_id>.json
        """
        # Check if we've seen this URL before
        sid = hashlib.sha1(canon_url.encode("utf-8")).hexdigest()
        map_path = self.cache / "map" / f"{sid}.json"
        etag = last_mod = None
        
        if map_path.exists():
            prev = json.loads(map_path.read_text())
            blob_id = prev.get("blob_id")
            if blob_id:
                _, meta_path = self._blob_paths(blob_id)
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    etag = meta.get("etag")
                    last_mod = meta.get("last_modified")
                    if not force_refresh:
                        # Return existing blob immediately
                        blob_path, _ = self._blob_paths(blob_id)
                        return blob_id, blob_path, meta

        # Prepare headers for conditional GET
        headers = {"User-Agent": self.ua}
        if etag:
            headers["If-None-Match"] = etag
        if last_mod:
            headers["If-Modified-Since"] = last_mod

        # Fetch the URL
        try:
            r = self.session.get(url, timeout=self.timeout)
        except requests.exceptions.Timeout:
            print(f"Timeout fetching {url} (timeout: {self.timeout}s)")
            raise
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error fetching {url}: {e}")
            raise
        except requests.exceptions.RequestException as e:
            print(f"Request error fetching {url}: {e}")
            raise
        
        # Handle 304 Not Modified
        if r.status_code == 304 and map_path.exists():
            blob_id = json.loads(map_path.read_text())["blob_id"]
            blob_path, meta_path = self._blob_paths(blob_id)
            return blob_id, blob_path, json.loads(meta_path.read_text())

        # Process new content
        content = r.content
        blob_id = hashlib.sha256(content).hexdigest()
        blob_path, meta_path = self._blob_paths(blob_id)

        # Store blob if it doesn't exist
        if not blob_path.exists():
            cctx = zstd.ZstdCompressor(level=10, write_content_size=True)
            blob_path.write_bytes(cctx.compress(content))
            
            meta = {
                "url": url,
                "canonical_url": canon_url,
                "status": r.status_code,
                "etag": r.headers.get("ETag"),
                "last_modified": r.headers.get("Last-Modified"),
                "mime": r.headers.get("Content-Type"),
                "fetched_at": time.time(),
                "size": len(content)
            }
            from efi_core.utils import DateTimeEncoder
            meta_path.write_text(json.dumps(meta, indent=2, cls=DateTimeEncoder), encoding="utf-8")
        else:
            meta = json.loads(meta_path.read_text())

        # Update URL→blob map
        map_path.write_text(
            json.dumps({"canonical_url": canon_url, "blob_id": blob_id}, indent=2, cls=DateTimeEncoder),
            encoding="utf-8"
        )
        
        return blob_id, blob_path, meta

    def fetch_raw(self, url: str, stable_id: str) -> Tuple[bytes, Dict[str, Any], str]:
        """
        Fetch raw content from a URL and return (raw_bytes, fetch_meta, raw_ext)
        
        This is a lower-level method used by corpus builders.
        """
        # Get the content using the high-level get method
        blob_id, blob_path, fetch_meta = self.get(url, url)
        
        # Read the compressed blob
        with open(blob_path, 'rb') as f:
            cbytes = f.read()
        
        # Decompress
        dctx = zstd.ZstdDecompressor()
        raw_bytes = dctx.decompress(cbytes)
        
        # Determine file extension from content type
        mime = fetch_meta.get("mime", "")
        if "html" in mime.lower():
            raw_ext = "html"
        elif "pdf" in mime.lower():
            raw_ext = "pdf"
        elif "text" in mime.lower():
            raw_ext = "txt"
        else:
            raw_ext = "bin"
        
        return raw_bytes, fetch_meta, raw_ext
