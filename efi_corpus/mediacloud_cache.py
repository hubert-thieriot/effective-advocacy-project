"""
MediaCloud Search Cache - Caches MediaCloud API search results
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta, date


class DateAwareJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles date objects"""
    def default(self, obj):
        if isinstance(obj, date):
            return obj.isoformat()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


class MediaCloudSearchCache:
    """Caches MediaCloud API search results to avoid repeated API calls"""
    
    def __init__(self, cache_root: Path):
        self.cache_root = cache_root / "mediacloud_searches"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        
        # Cache metadata file
        self.metadata_file = self.cache_root / "cache_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata"""
        if self.metadata_file.exists():
            try:
                return json.loads(self.metadata_file.read_text())
            except (json.JSONDecodeError, FileNotFoundError):
                return {"searches": {}, "last_cleanup": None}
        return {"searches": {}, "last_cleanup": None}
    
    def _save_metadata(self):
        """Save cache metadata"""
        self.metadata_file.write_text(json.dumps(self.metadata, indent=2, cls=DateAwareJSONEncoder))
    
    def _generate_cache_key(self, query: str, collection_id: int = None, media_ids: List[int] = None, start_date = None, end_date = None) -> str:
        """Generate a unique cache key for a search"""
        # Convert dates to strings if they're date objects
        start_date_str = str(start_date) if hasattr(start_date, 'isoformat') else str(start_date) if start_date else ""
        end_date_str = str(end_date) if hasattr(end_date, 'isoformat') else str(end_date) if end_date else ""
        
        # Create a deterministic key based on search parameters
        key_data = {
            "query": query,
            "start_date": start_date_str,
            "end_date": end_date_str
        }
        if collection_id is not None:
            key_data["collection_id"] = collection_id
        if media_ids is not None:
            key_data["media_ids"] = sorted(media_ids)  # Sort for consistency
        key_string = json.dumps(key_data, sort_keys=True, cls=DateAwareJSONEncoder)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]
    
    def _get_cache_file_path(self, cache_key: str) -> Path:
        """Get the file path for a cache entry"""
        return self.cache_root / f"search_{cache_key}.json"
    
    def get_search_results(self, query: str, collection_id: int = None, media_ids: List[int] = None, 
                          start_date = None, end_date = None, max_age_hours: int = 24) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached search results if they exist and are not too old
        
        Args:
            query: Search query
            collection_id: MediaCloud collection ID (optional, use with media_ids)
            media_ids: List of MediaCloud media IDs (optional, use with collection_id)
            start_date: Start date for search
            end_date: End date for search
            max_age_hours: Maximum age of cached results in hours
            
        Returns:
            Cached search results or None if not found/too old
        """
        cache_key = self._generate_cache_key(query, collection_id, media_ids, start_date, end_date)
        cache_file = self._get_cache_file_path(cache_key)
        
        if not cache_file.exists():
            return None
        
        # Check if cache entry exists in metadata
        if cache_key not in self.metadata["searches"]:
            return None
        
        cache_entry = self.metadata["searches"][cache_key]
        cache_time = datetime.fromisoformat(cache_entry["cached_at"])
        
        # Check if cache is too old
        if datetime.now() - cache_time > timedelta(hours=max_age_hours):
            print(f"🗑️  Cache expired for query: {query[:50]}...")
            self._remove_cache_entry(cache_key)
            return None
        
        # Load and return cached results
        try:
            cached_data = json.loads(cache_file.read_text())
            print(f"📋 Using cached results for query: {query[:50]}... ({len(cached_data['stories'])} stories)")
            return cached_data["stories"]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️  Error reading cache file: {e}")
            self._remove_cache_entry(cache_key)
            return None
    
    def cache_search_results(self, query: str, collection_id: int = None, media_ids: List[int] = None,
                           start_date = None, end_date = None, stories: List[Dict[str, Any]] = None):
        """
        Cache search results
        
        Args:
            query: Search query
            collection_id: MediaCloud collection ID (optional)
            media_ids: List of MediaCloud media IDs (optional)
            start_date: Start date for search
            end_date: End date for search
            stories: List of story dictionaries to cache
        """
        cache_key = self._generate_cache_key(query, collection_id, media_ids, start_date, end_date)
        cache_file = self._get_cache_file_path(cache_key)
        
        # Convert dates to strings if they're date objects
        start_date_str = str(start_date) if hasattr(start_date, 'isoformat') else str(start_date) if start_date else ""
        end_date_str = str(end_date) if hasattr(end_date, 'isoformat') else str(end_date) if end_date else ""
        
        # Prepare cache data
        cache_data = {
            "query": query,
            "start_date": start_date_str,
            "end_date": end_date_str,
            "stories": stories or [],
            "cached_at": datetime.now().isoformat(),
            "story_count": len(stories) if stories else 0
        }
        if collection_id is not None:
            cache_data["collection_id"] = collection_id
        if media_ids is not None:
            cache_data["media_ids"] = media_ids
        
        # Save to file
        cache_file.write_text(json.dumps(cache_data, indent=2, cls=DateAwareJSONEncoder))
        
        # Update metadata
        metadata_entry = {
            "query": query,
            "start_date": start_date_str,
            "end_date": end_date_str,
            "story_count": len(stories) if stories else 0,
            "cached_at": datetime.now().isoformat(),
            "cache_file": str(cache_file)
        }
        if collection_id is not None:
            metadata_entry["collection_id"] = collection_id
        if media_ids is not None:
            metadata_entry["media_ids"] = media_ids
        self.metadata["searches"][cache_key] = metadata_entry
        self._save_metadata()
        
        print(f"💾 Cached {len(stories)} stories for query: {query[:50]}...")
    
    def _remove_cache_entry(self, cache_key: str):
        """Remove a cache entry and its file"""
        if cache_key in self.metadata["searches"]:
            cache_file = self._get_cache_file_path(cache_key)
            if cache_file.exists():
                cache_file.unlink()
            del self.metadata["searches"][cache_key]
            self._save_metadata()
    
    def clear_cache(self, older_than_hours: Optional[int] = None):
        """
        Clear cache entries
        
        Args:
            older_than_hours: If provided, only clear entries older than this many hours
        """
        if older_than_hours is None:
            # Clear all entries
            for cache_key in list(self.metadata["searches"].keys()):
                self._remove_cache_entry(cache_key)
            print(f"🗑️  Cleared all MediaCloud search cache entries")
        else:
            # Clear only old entries
            cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
            old_entries = []
            
            for cache_key, entry in self.metadata["searches"].items():
                cache_time = datetime.fromisoformat(entry["cached_at"])
                if cache_time < cutoff_time:
                    old_entries.append(cache_key)
            
            for cache_key in old_entries:
                self._remove_cache_entry(cache_key)
            
            print(f"🗑️  Cleared {len(old_entries)} MediaCloud search cache entries older than {older_than_hours} hours")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_entries = len(self.metadata["searches"])
        total_stories = sum(entry["story_count"] for entry in self.metadata["searches"].values())
        
        # Calculate cache size
        cache_size = 0
        for cache_file in self.cache_root.glob("search_*.json"):
            if cache_file.is_file():
                cache_size += cache_file.stat().st_size
        
        return {
            "total_entries": total_entries,
            "total_stories": total_stories,
            "cache_size_bytes": cache_size,
            "cache_size_mb": round(cache_size / (1024 * 1024), 2),
            "last_cleanup": self.metadata.get("last_cleanup")
        }
    
    def list_cached_searches(self) -> List[Dict[str, Any]]:
        """List all cached searches"""
        searches = []
        for cache_key, entry in self.metadata["searches"].items():
            searches.append({
                "cache_key": cache_key,
                "query": entry["query"],
                "collection_id": entry["collection_id"],
                "date_range": f"{entry['start_date']} to {entry['end_date']}",
                "story_count": entry["story_count"],
                "cached_at": entry["cached_at"]
            })
        return searches
