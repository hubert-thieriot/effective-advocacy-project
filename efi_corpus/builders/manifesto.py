"""
ManifestoCorpusBuilder - Corpus builder for Manifesto Project data

Uses the Manifesto Project API to download political party manifestos.
API documentation: https://manifestoproject.wzb.eu/information/documents/api
"""

from pathlib import Path
from typing import Iterable, Dict, Any, List, Optional
from hashlib import sha1
import time
import requests
from decouple import config

from .base import BaseCorpusBuilder
from ..types import BuilderParams, DiscoveryItem
from ..fetcher import Fetcher


class ManifestoCorpusBuilder(BaseCorpusBuilder):
    """
    Corpus builder that integrates with the Manifesto Project API.
    
    Downloads political party manifestos and stores them with metadata.
    One document = one manifesto (all text segments concatenated).
    """
    
    API_BASE_URL = "https://manifesto-project.wzb.eu/api/v1"
    
    def __init__(self, corpus_dir: Path, countries: List[str] = None, fetcher: Fetcher = None, cache_root: Path = None):
        """
        Initialize the Manifesto corpus builder.
        
        Args:
            corpus_dir: Path to the corpus directory
            countries: List of country codes to filter manifestos (ISO 3166-1 numeric codes)
            fetcher: Optional fetcher instance (not used for API calls, but kept for consistency)
            cache_root: Optional cache root path
        """
        super().__init__(corpus_dir, fetcher, cache_root)
        self.countries = countries or []
        
        # Initialize Manifesto Project API key
        api_key = config('MANIFESTO_KEY')
        if not api_key:
            raise ValueError("MANIFESTO_KEY environment variable is required.")
        self.api_key = api_key
        
        # Cache for core dataset metadata
        self._core_dataset_cache: Optional[List[Dict[str, Any]]] = None
    
    def _api_request(self, endpoint: str, params: Dict[str, Any] = None) -> Any:
        """
        Make a request to the Manifesto Project API.
        
        Args:
            endpoint: API endpoint (e.g., 'metadata', 'texts_and_annotations')
            params: Query parameters
            
        Returns:
            JSON response from the API
        """
        url = f"{self.API_BASE_URL}/{endpoint}"
        request_params = {"api_key": self.api_key}
        if params:
            request_params.update(params)
        
        response = requests.get(url, params=request_params, timeout=60)
        response.raise_for_status()
        return response.json()
    
    def _get_core_dataset(self, version: str = "MPDS2025a") -> List[Dict[str, Any]]:
        """
        Get the core dataset (metadata for all manifestos).
        
        The core dataset is returned as a list where first row is headers.
        We convert it to a list of dictionaries.
        
        Args:
            version: Dataset version (e.g., "MPDS2024a")
            
        Returns:
            List of manifesto metadata entries as dictionaries
        """
        if self._core_dataset_cache is not None:
            return self._core_dataset_cache
        
        print(f"Fetching Manifesto Project core dataset (version: {version})...")
        response = self._api_request("get_core", {"key": version})
        
        # The response is a list where first row is headers
        if not isinstance(response, list) or len(response) < 2:
            print("Warning: Core dataset is empty or invalid")
            self._core_dataset_cache = []
            return self._core_dataset_cache
        
        headers = response[0]
        rows = response[1:]
        
        # Convert to list of dicts
        self._core_dataset_cache = []
        for row in rows:
            entry = dict(zip(headers, row))
            # Construct manifesto_id from party and date
            party = entry.get("party", "")
            date = entry.get("date", "")
            if party and date:
                entry["manifesto_id"] = f"{party}_{date}"
            self._core_dataset_cache.append(entry)
        
        print(f"Fetched {len(self._core_dataset_cache)} manifesto entries from core dataset")
        return self._core_dataset_cache
    
    def _get_manifesto_text(self, manifesto_id: str, metadata_version: str = "2025-1") -> Optional[str]:
        """
        Get the full text of a manifesto by concatenating all text segments.
        
        Args:
            manifesto_id: The manifesto key (e.g., "41320_202109")
            metadata_version: Metadata/texts version (e.g., "2025-1")
            
        Returns:
            Concatenated text of the manifesto, or None if not available
        """
        try:
            response = self._api_request("texts_and_annotations", {
                "keys[]": manifesto_id,
                "version": metadata_version
            })
            
            # The response contains items with text segments
            items = response.get("items", [])
            if not items:
                return None
            
            # Get the first (and usually only) manifesto in the response
            manifesto_data = items[0]
            text_items = manifesto_data.get("items", [])
            
            # Concatenate all text segments
            texts = []
            for item in text_items:
                text = item.get("text", "").strip()
                if text:
                    texts.append(text)
            
            return " ".join(texts) if texts else None
            
        except Exception as e:
            print(f"⚠️  Failed to get text for manifesto {manifesto_id}: {e}")
            return None
    
    def _get_manifesto_metadata(self, manifesto_id: str, metadata_version: str = "2025-1") -> Optional[Dict[str, Any]]:
        """
        Get detailed metadata for a manifesto including language, title, etc.
        
        Args:
            manifesto_id: The manifesto key (e.g., "41320_202109")
            metadata_version: Metadata version (e.g., "2025-1")
            
        Returns:
            Metadata dict or None if not available
        """
        try:
            response = self._api_request("metadata", {
                "keys[]": manifesto_id,
                "version": metadata_version
            })
            
            items = response.get("items", [])
            if items:
                return items[0]
            return None
            
        except Exception as e:
            print(f"⚠️  Failed to get metadata for manifesto {manifesto_id}: {e}")
            return None
    
    def discover(self, params: BuilderParams) -> Iterable[DiscoveryItem]:
        """
        Discover manifestos based on filter criteria.
        
        Filters:
        - countries: List of country codes (from constructor or params.extra)
        - date_from: Minimum election date (YYYY-MM-DD or YYYY format)
        """
        # Get filter parameters
        countries = self.countries or (params.extra or {}).get('countries', [])
        date_from = params.date_from
        date_to = params.date_to
        core_version = (params.extra or {}).get('core_version', 'MPDS2025a')
        
        # Get core dataset
        core_dataset = self._get_core_dataset(core_version)
        
        # Filter manifestos
        for entry in core_dataset:
            # Extract key fields
            manifesto_id = entry.get("manifesto_id")
            if not manifesto_id:
                continue
            
            # Country filter (using countryname or country code)
            country_code = str(entry.get("country", ""))
            country_name = entry.get("countryname", "")
            
            if countries:
                # Check if country matches any in the filter list
                country_match = False
                for c in countries:
                    c_str = str(c).lower()
                    if c_str == country_code.lower() or c_str == country_name.lower():
                        country_match = True
                        break
                if not country_match:
                    continue
            
            # Date filter (using edate - election date or date field)
            election_date = entry.get("edate", "") or entry.get("date", "")
            if election_date:
                # Election date is in format DD/MM/YYYY or YYYYMM
                # Extract year for filtering
                if "/" in str(election_date):
                    # Format: DD/MM/YYYY
                    election_year = str(election_date).split("/")[-1]
                else:
                    # Format: YYYYMM
                    election_year = str(election_date)[:4]
                
                # Filter by date_from
                if date_from:
                    from_year = str(date_from)[:4]
                    if election_year < from_year:
                        continue
                
                # Filter by date_to
                if date_to:
                    to_year = str(date_to)[:4]
                    if election_year > to_year:
                        continue
            
            # Extract metadata
            party_id = entry.get("party", "")
            party_name = entry.get("partyname", "")
            party_abbrev = entry.get("partyabbrev", "")
            
            # Create a URL-like identifier for the manifesto
            url = f"manifesto://{manifesto_id}"
            
            yield DiscoveryItem(
                url=url,
                canonical_url=url,
                title=f"{party_name} ({party_abbrev}) - {election_date}" if party_name else manifesto_id,
                published_at=str(election_date) if election_date else None,
                extra={
                    "manifesto_id": manifesto_id,
                    "party_id": party_id,
                    "party_name": party_name,
                    "party_abbrev": party_abbrev,
                    "country": country_code,
                    "country_name": country_name,
                    "election_date": election_date,
                    # Store additional metadata from core dataset
                    "pervote": entry.get("pervote"),  # Vote share
                    "absseat": entry.get("absseat"),  # Absolute seats
                    "totseats": entry.get("totseats"),  # Total seats
                    "progtype": entry.get("progtype"),  # Program type
                }
            )
    
    def run(self, *, params: BuilderParams | None = None, override: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Run the corpus builder.
        
        Args:
            params: Build parameters (if None, load from manifest)
            override: Override specific parameters
            
        Returns:
            Summary of the build process
        """
        manifest = self.corpus.load_manifest()
        
        if params is None:
            if not manifest:
                raise ValueError("No manifest found; first run must provide params.")
            params = BuilderParams(**manifest["params"])
        
        if override:
            merged = {**params.__dict__, **override}
            params = BuilderParams(**merged)
        
        # Persist params to manifest before run
        manifest.setdefault("name", self.corpus.corpus_path.name)
        manifest.setdefault("source", "manifestocorpus")
        manifest["params"] = params.__dict__
        manifest.setdefault("history", [])
        
        # Get versions from params
        metadata_version = (params.extra or {}).get('metadata_version', '2025-1')
        
        # Discover manifestos
        print("🔍 Discovering manifestos...")
        discovered = list(self.discover(params))
        print(f"Found {len(discovered)} manifestos matching criteria")
        
        # Map manifesto_id to discovery item for metadata enrichment
        discovered_by_url = {d.url: d for d in discovered}
        
        # Compute frontier (manifestos not already in corpus)
        def stable_id(url: str) -> str:
            return sha1(url.encode("utf-8")).hexdigest()
        
        pairs = [(d.url, stable_id(d.url)) for d in discovered]
        frontier = [(u, s) for (u, s) in pairs if not self.corpus.has_doc(s)]
        
        print(f"📥 {len(frontier)} new manifestos to download (skipping {len(pairs) - len(frontier)} already in corpus)")
        
        # Process frontier
        added = 0
        failed = []
        skipped_empty = 0
        
        for i, (url, sid) in enumerate(frontier, 1):
            try:
                discovered_item = discovered_by_url.get(url)
                manifesto_id = discovered_item.extra.get("manifesto_id") if discovered_item else None
                
                if not manifesto_id:
                    print(f"  ⚠️  Skipping: No manifesto_id for {url}")
                    continue
                
                print(f"[{i}/{len(frontier)}] Downloading manifesto: {manifesto_id}")
                
                # Get detailed metadata first (to check if text is available)
                detailed_meta = self._get_manifesto_metadata(manifesto_id, metadata_version)
                
                if not detailed_meta:
                    print(f"  ⚠️  Skipped: No metadata available (text may not be digitized)")
                    skipped_empty += 1
                    continue
                
                # Get manifesto text
                text = self._get_manifesto_text(manifesto_id, metadata_version)
                
                if not text:
                    print(f"  ⚠️  Skipped: No text available")
                    skipped_empty += 1
                    continue
                
                # Build metadata
                meta = {
                    "doc_id": sid,
                    "uri": url,
                    "manifesto_id": manifesto_id,
                    "title": detailed_meta.get("title") or (discovered_item.title if discovered_item else manifesto_id),
                    "published_at": discovered_item.published_at if discovered_item else None,
                    "language": detailed_meta.get("language"),
                    "source": "manifestocorpus",
                    "extra": {
                        **(discovered_item.extra if discovered_item else {}),
                        "annotations": detailed_meta.get("annotations"),
                        "handbook": detailed_meta.get("handbook"),
                        "url_original": detailed_meta.get("url_original"),
                    },
                }
                
                # Write document to corpus
                self.corpus.write_document(
                    stable_id=sid,
                    meta=meta,
                    text=text,
                    raw_bytes=text.encode("utf-8"),
                    raw_ext="txt",
                    fetch_info={"fetched_at": time.time(), "source": "manifesto_api", "version": metadata_version}
                )
                
                # Append to index
                self.corpus.append_index({
                    "id": sid,
                    "url": url,
                    "manifesto_id": manifesto_id,
                    "published_at": meta["published_at"],
                    "title": meta["title"],
                    "language": meta["language"],
                    "party_name": discovered_item.extra.get("party_name") if discovered_item else None,
                    "party_abbrev": discovered_item.extra.get("party_abbrev") if discovered_item else None,
                    "country": discovered_item.extra.get("country") if discovered_item else None,
                    "country_name": discovered_item.extra.get("country_name") if discovered_item else None,
                })
                
                added += 1
                print(f"  ✅ Added ({len(text)} chars)")
                
                # Small delay to be nice to the API
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                failed.append({"url": url, "error": str(e)})
                continue
        
        # Update manifest
        manifest["history"].append({
            "run_at": time.time(),
            "discovered": len(discovered),
            "added": added,
            "skipped_empty": skipped_empty,
            "skipped_duplicate": len(pairs) - len(frontier),
            "failed": len(failed),
            "failed_details": failed,
            "date_from": params.date_from,
            "date_to": params.date_to,
        })
        manifest["doc_count"] = manifest.get("doc_count", 0) + added
        self.corpus.save_manifest(manifest)
        
        # Print summary
        print(f"\n📊 Build Summary:")
        print(f"  Discovered: {len(discovered)}")
        print(f"  Added: {added}")
        print(f"  Skipped (empty/no text): {skipped_empty}")
        print(f"  Skipped (duplicate): {len(pairs) - len(frontier)}")
        print(f"  Failed: {len(failed)}")
        print(f"  Total docs in corpus: {self.corpus.get_document_count()}")
        
        if failed:
            print(f"\n⚠️  {len(failed)} manifestos failed:")
            for f in failed[:5]:
                print(f"  - {f['url']}: {f['error']}")
            if len(failed) > 5:
                print(f"  ... and {len(failed) - 5} more")
        
        return {
            "discovered": len(discovered),
            "added": added,
            "skipped_empty": skipped_empty,
            "skipped_duplicate": len(pairs) - len(frontier),
            "failed": len(failed),
            "total_docs": self.corpus.get_document_count(),
            "failed_details": failed
        }
