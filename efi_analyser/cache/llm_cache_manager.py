"""
Centralized LLM cache manager for all LLM queries.

Provides consistent caching across different LLM interfaces (Ollama, OpenAI, etc.)
with support for cache invalidation, management, and debugging.
"""

from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a single cache entry."""
    model: str
    messages: List[Dict[str, str]]
    parameters: Dict[str, Any]
    response: str
    timestamp: float
    inference_time: float
    cache_key: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary."""
        return cls(**data)


class LLMCacheManager:
    """Centralized cache manager for all LLM queries."""
    
    def __init__(self, cache_dir: Path = Path("cache/llm"), max_cache_size_mb: Optional[int] = None):
        """Initialize cache manager.
        
        Args:
            cache_dir: Directory to store cache files
            max_cache_size_mb: Maximum cache size in MB (None for unlimited)
        """
        self.cache_dir = Path(cache_dir)
        self.max_cache_size_mb = max_cache_size_mb
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different models
        self.model_dirs = {}
        
    def _get_model_dir(self, model: str) -> Path:
        """Get cache directory for a specific model."""
        if model not in self.model_dirs:
            # Clean model name for directory name
            clean_model = model.replace(":", "_").replace("/", "_").replace(" ", "_")
            model_dir = self.cache_dir / clean_model
            model_dir.mkdir(parents=True, exist_ok=True)
            self.model_dirs[model] = model_dir
        return self.model_dirs[model]
    
    def _generate_cache_key(self, model: str, messages: List[Dict[str, str]], parameters: Dict[str, Any]) -> str:
        """Generate a unique cache key for the query."""
        # Create a deterministic key based on model, messages, and parameters
        key_data = {
            "model": model,
            "messages": messages,
            "parameters": parameters
        }
        
        # Sort keys for consistent hashing
        key_json = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(key_json.encode('utf-8')).hexdigest()
    
    def _get_cache_file_path(self, model: str, cache_key: str) -> Path:
        """Get the file path for a cache entry."""
        model_dir = self._get_model_dir(model)
        # Use first 2 chars of hash for subdirectory organization
        subdir = model_dir / cache_key[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{cache_key}.json"
    
    def get(self, model: str, messages: List[Dict[str, str]], parameters: Dict[str, Any]) -> Optional[CacheEntry]:
        """Get cached response if available.
        
        Args:
            model: Model name
            messages: Input messages
            parameters: Model parameters (temperature, etc.)
            
        Returns:
            CacheEntry if found, None otherwise
        """
        cache_key = self._generate_cache_key(model, messages, parameters)
        cache_file = self._get_cache_file_path(model, cache_key)
        
        if not cache_file.exists():
            return None
            
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                entry = CacheEntry.from_dict(data)
                logger.debug(f"Cache hit for {model}: {cache_key[:8]}...")
                return entry
        except Exception as e:
            logger.warning(f"Failed to read cache file {cache_file}: {e}")
            return None
    
    def put(self, model: str, messages: List[Dict[str, str]], parameters: Dict[str, Any], 
            response: str, inference_time: float) -> str:
        """Store response in cache.
        
        Args:
            model: Model name
            messages: Input messages
            parameters: Model parameters
            response: Model response
            inference_time: Time taken for inference
            
        Returns:
            Cache key for the stored entry
        """
        cache_key = self._generate_cache_key(model, messages, parameters)
        cache_file = self._get_cache_file_path(model, cache_key)
        
        entry = CacheEntry(
            model=model,
            messages=messages,
            parameters=parameters,
            response=response,
            timestamp=time.time(),
            inference_time=inference_time,
            cache_key=cache_key
        )
        
        try:
            # Write to temporary file first, then rename (atomic operation)
            temp_file = cache_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(entry.to_dict(), f, indent=2, ensure_ascii=False)
            temp_file.rename(cache_file)
            logger.debug(f"Cached response for {model}: {cache_key[:8]}...")
        except Exception as e:
            logger.warning(f"Failed to write cache file {cache_file}: {e}")
        
        return cache_key
    
    def invalidate_model(self, model: str) -> int:
        """Invalidate all cache entries for a specific model.
        
        Args:
            model: Model name to invalidate
            
        Returns:
            Number of cache files removed
        """
        model_dir = self._get_model_dir(model)
        removed_count = 0
        
        if model_dir.exists():
            for cache_file in model_dir.rglob("*.json"):
                try:
                    cache_file.unlink()
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {cache_file}: {e}")
        
        logger.info(f"Invalidated {removed_count} cache entries for model {model}")
        return removed_count
    
    def invalidate_all(self) -> int:
        """Invalidate all cache entries.
        
        Returns:
            Number of cache files removed
        """
        removed_count = 0
        
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.rglob("*.json"):
                try:
                    cache_file.unlink()
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {cache_file}: {e}")
        
        logger.info(f"Invalidated {removed_count} total cache entries")
        return removed_count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = {
            "total_entries": 0,
            "total_size_mb": 0.0,
            "models": {},
            "oldest_entry": None,
            "newest_entry": None
        }
        
        if not self.cache_dir.exists():
            return stats
        
        oldest_time = float('inf')
        newest_time = 0.0
        
        for cache_file in self.cache_dir.rglob("*.json"):
            try:
                file_size = cache_file.stat().st_size
                stats["total_size_mb"] += file_size / (1024 * 1024)
                stats["total_entries"] += 1
                
                # Extract model from path
                model_name = cache_file.parent.parent.name
                if model_name not in stats["models"]:
                    stats["models"][model_name] = {"entries": 0, "size_mb": 0.0}
                stats["models"][model_name]["entries"] += 1
                stats["models"][model_name]["size_mb"] += file_size / (1024 * 1024)
                
                # Check timestamps
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    entry_time = data.get('timestamp', 0)
                    if entry_time < oldest_time:
                        oldest_time = entry_time
                        stats["oldest_entry"] = time.ctime(entry_time)
                    if entry_time > newest_time:
                        newest_time = entry_time
                        stats["newest_entry"] = time.ctime(entry_time)
                        
            except Exception as e:
                logger.warning(f"Failed to process cache file {cache_file}: {e}")
        
        return stats
    
    def cleanup_old_entries(self, max_age_days: int = 30) -> int:
        """Remove cache entries older than specified days.
        
        Args:
            max_age_days: Maximum age in days
            
        Returns:
            Number of entries removed
        """
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        removed_count = 0
        
        if not self.cache_dir.exists():
            return removed_count
        
        for cache_file in self.cache_dir.rglob("*.json"):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    entry_time = data.get('timestamp', 0)
                    
                    if entry_time < cutoff_time:
                        cache_file.unlink()
                        removed_count += 1
                        
            except Exception as e:
                logger.warning(f"Failed to process cache file {cache_file}: {e}")
        
        logger.info(f"Cleaned up {removed_count} old cache entries (>{max_age_days} days)")
        return removed_count


# Global cache manager instance
_global_cache_manager: Optional[LLMCacheManager] = None


def get_cache_manager() -> LLMCacheManager:
    """Get the global cache manager instance."""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = LLMCacheManager()
    return _global_cache_manager


def set_cache_manager(cache_manager: LLMCacheManager) -> None:
    """Set the global cache manager instance."""
    global _global_cache_manager
    _global_cache_manager = cache_manager
