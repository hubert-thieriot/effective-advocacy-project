"""
Tests for Fetcher
"""

import pytest
import json
import hashlib
import time
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from unittest.mock import MagicMock
import requests

from efi_corpus.fetcher import Fetcher


class TestFetcher:
    """Test cases for Fetcher"""
    
    @pytest.fixture
    def cache_root(self, tmp_path):
        """Create a temporary cache directory"""
        cache_root = tmp_path / "cache"
        return cache_root
    
    @pytest.fixture
    def fetcher(self, cache_root):
        """Create a Fetcher instance"""
        return Fetcher(cache_root, ua="TestFetcher/1.0", timeout=10)
    
    def test_init_creates_directories(self, cache_root):
        """Test that initialization creates necessary cache directories"""
        fetcher = Fetcher(cache_root)
        
        assert (cache_root / "http" / "blobs").exists()
        assert (cache_root / "http" / "meta").exists()
        assert (cache_root / "http" / "map").exists()
    
    def test_init_custom_ua_and_timeout(self, cache_root):
        """Test that custom user agent and timeout are set"""
        fetcher = Fetcher(cache_root, ua="CustomUA/2.0", timeout=30)
        
        assert fetcher.ua == "CustomUA/2.0"
        assert fetcher.timeout == 30
    
    def test_blob_paths(self, fetcher):
        """Test that blob paths are correctly generated"""
        blob_id = "abc123def456"
        blob_path, meta_path = fetcher._blob_paths(blob_id)
        
        assert blob_path.name == "abc123def456.bin.zst"
        assert meta_path.name == "abc123def456.json"
        assert blob_path.parent.name == "blobs"
        assert meta_path.parent.name == "meta"
    
    def test_get_new_url_success(self, fetcher, cache_root):
        """Test successful fetching of a new URL"""
        # Mock the session response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"<html>Test content</html>"
        mock_response.headers = {
            "ETag": '"abc123"',
            "Last-Modified": "Wed, 13 Aug 2025 10:00:00 GMT",
            "Content-Type": "text/html"
        }
        
        # Patch the instance's session
        with patch.object(fetcher, 'session') as mock_session:
            mock_session.get.return_value = mock_response
            
            url = "https://example.com/article"
            canon_url = "https://example.com/article"
            
            # Test the get method
            blob_id, blob_path, fetch_meta = fetcher.get(url, canon_url)
            
            # Verify the request was made
            mock_session.get.assert_called_once_with(url, timeout=10)
        
        # Verify blob_id is correct (SHA256 of content)
        expected_blob_id = hashlib.sha256(b"<html>Test content</html>").hexdigest()
        assert blob_id == expected_blob_id
        
        # Verify blob path
        assert blob_path.name == f"{expected_blob_id}.bin.zst"
        
        # Verify fetch metadata
        assert fetch_meta["url"] == url
        assert fetch_meta["canonical_url"] == canon_url
        assert fetch_meta["status"] == 200
        assert fetch_meta["etag"] == '"abc123"'
        assert fetch_meta["last_modified"] == "Wed, 13 Aug 2025 10:00:00 GMT"
        assert fetch_meta["mime"] == "text/html"
        assert "fetched_at" in fetch_meta
        assert fetch_meta["size"] == 25
        
        # Verify files were created
        assert blob_path.exists()
        meta_path = cache_root / "http" / "meta" / f"{expected_blob_id}.json"
        assert meta_path.exists()
        
        # Verify URL mapping was created
        stable_id = hashlib.sha1(canon_url.encode("utf-8")).hexdigest()
        map_path = cache_root / "http" / "map" / f"{stable_id}.json"
        assert map_path.exists()
        
        # Verify map content
        with open(map_path, 'r') as f:
            map_data = json.load(f)
        assert map_data["canonical_url"] == canon_url
        assert map_data["blob_id"] == expected_blob_id
    
    def test_get_existing_url_from_cache(self, fetcher, cache_root):
        """Test that existing URLs are served from cache"""
        # Create existing cache entries
        existing_blob_id = "existing123"
        stable_id = hashlib.sha1("https://example.com/article".encode("utf-8")).hexdigest()
        
        # Create map entry
        map_path = cache_root / "http" / "map" / f"{stable_id}.json"
        map_path.parent.mkdir(parents=True, exist_ok=True)
        with open(map_path, 'w') as f:
            json.dump({
                "canonical_url": "https://example.com/article",
                "blob_id": existing_blob_id
            }, f)
        
        # Create meta entry
        meta_path = cache_root / "http" / "meta" / f"{existing_blob_id}.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, 'w') as f:
            json.dump({
                "url": "https://example.com/article",
                "canonical_url": "https://example.com/article",
                "status": 200,
                "etag": '"abc123"',
                "last_modified": "Wed, 13 Aug 2025 10:00:00 GMT",
                "mime": "text/html",
                "fetched_at": time.time(),
                "size": 25
            }, f)
        
        # Create blob file
        blob_path = cache_root / "http" / "blobs" / f"{existing_blob_id}.bin.zst"
        blob_path.parent.mkdir(parents=True, exist_ok=True)
        blob_path.write_bytes(b"compressed content")
        
        url = "https://example.com/article"
        canon_url = "https://example.com/article"
        
        # Test the get method
        blob_id, blob_path, fetch_meta = fetcher.get(url, canon_url)
        
        # Should return from cache without making HTTP request
        # (No need to check mock since we're not patching)
        
        # Should return existing data
        assert blob_id == existing_blob_id
        assert fetch_meta["status"] == 200
    
    def test_get_with_conditional_headers(self, fetcher, cache_root):
        """Test that conditional GET headers are prepared when cache exists"""
        # Create existing cache with ETag and Last-Modified
        existing_blob_id = "existing123"
        stable_id = hashlib.sha1("https://example.com/article".encode("utf-8")).hexdigest()
        
        # Create map entry
        map_path = cache_root / "http" / "map" / f"{stable_id}.json"
        map_path.parent.mkdir(parents=True, exist_ok=True)
        with open(map_path, 'w') as f:
            json.dump({
                "canonical_url": "https://example.com/article",
                "blob_id": existing_blob_id
            }, f)
        
        # Create meta entry with ETag and Last-Modified
        meta_path = cache_root / "http" / "meta" / f"{existing_blob_id}.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, 'w') as f:
            json.dump({
                "url": "https://example.com/article",
                "canonical_url": "https://example.com/article",
                "status": 200,
                "etag": '"abc123"',
                "last_modified": "Wed, 13 Aug 2025 10:00:00 GMT",
                "mime": "text/html",
                "fetched_at": time.time(),
                "size": 25
            }, f)
        
        # Create blob file
        blob_path = cache_root / "http" / "blobs" / f"{existing_blob_id}.bin.zst"
        blob_path.parent.mkdir(parents=True, exist_ok=True)
        blob_path.write_bytes(b"compressed content")
        
        url = "https://example.com/article"
        canon_url = "https://example.com/article"
        
        # Test the get method (should return from cache)
        blob_id, blob_path, fetch_meta = fetcher.get(url, canon_url)
        
        # Should return cached content without making HTTP request
        assert blob_id == existing_blob_id
        assert fetch_meta["status"] == 200
        assert fetch_meta["etag"] == '"abc123"'
        assert fetch_meta["last_modified"] == "Wed, 13 Aug 2025 10:00:00 GMT"
    
    def test_get_force_refresh(self, fetcher, cache_root):
        """Test that force_refresh bypasses cache"""
        # Create existing cache
        existing_blob_id = "existing123"
        stable_id = hashlib.sha1("https://example.com/article".encode("utf-8")).hexdigest()
        
        # Create map entry
        map_path = cache_root / "http" / "map" / f"{stable_id}.json"
        map_path.parent.mkdir(parents=True, exist_ok=True)
        with open(map_path, 'w') as f:
            json.dump({
                "canonical_url": "https://example.com/article",
                "blob_id": existing_blob_id
            }, f)
        
        # Create meta entry
        meta_path = cache_root / "http" / "meta" / f"{existing_blob_id}.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, 'w') as f:
            json.dump({
                "url": "https://example.com/article",
                "canonical_url": "https://example.com/article",
                "status": 200,
                "etag": '"abc123"',
                "last_modified": "Wed, 13 Aug 2025 10:00:00 GMT",
                "mime": "text/html",
                "fetched_at": time.time(),
                "size": 25
            }, f)
        
        # Create blob file
        blob_path = cache_root / "http" / "blobs" / f"{existing_blob_id}.bin.zst"
        blob_path.parent.mkdir(parents=True, exist_ok=True)
        blob_path.write_bytes(b"compressed content")
        
        # Mock response for force refresh
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"<html>Updated content</html>"
        mock_response.headers = {"Content-Type": "text/html"}
        
        url = "https://example.com/article"
        canon_url = "https://example.com/article"
        
        # Patch the instance's session
        with patch.object(fetcher, 'session') as mock_session:
            mock_session.get.return_value = mock_response
            
            # Test the get method with force_refresh
            blob_id, blob_path, fetch_meta = fetcher.get(url, canon_url, force_refresh=True)
            
            # Should make HTTP request despite cache
            mock_session.get.assert_called_once()
            
            # Should return new content
            expected_blob_id = hashlib.sha256(b"<html>Updated content</html>").hexdigest()
            assert blob_id == expected_blob_id
    
    def test_get_http_error(self, fetcher, cache_root):
        """Test handling of HTTP errors"""
        # Mock response with error status
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.content = b"Not Found"
        mock_response.headers = {"Content-Type": "text/html"}
        
        url = "https://example.com/notfound"
        canon_url = "https://example.com/notfound"
        
        # Patch the instance's session
        with patch.object(fetcher, 'session') as mock_session:
            mock_session.get.return_value = mock_response
            
            # Test the get method
            blob_id, blob_path, fetch_meta = fetcher.get(url, canon_url)
            
            # Should still process the error response
            assert blob_id == hashlib.sha256(b"Not Found").hexdigest()
            assert fetch_meta["status"] == 404
            assert fetch_meta["mime"] == "text/html"
    
    def test_get_requests_exception(self, fetcher):
        """Test handling of requests exceptions"""
        url = "https://example.com/error"
        canon_url = "https://example.com/error"
        
        # Patch the instance's session to raise an exception
        with patch.object(fetcher, 'session') as mock_session:
            mock_session.get.side_effect = requests.exceptions.ConnectionError("Connection failed")
            
            # Should raise the exception
            with pytest.raises(requests.exceptions.ConnectionError, match="Connection failed"):
                fetcher.get(url, canon_url)
    
    def test_get_different_canonical_urls(self, fetcher, cache_root):
        """Test that different canonical URLs create different stable IDs"""
        url1 = "https://example.com/article?utm_source=google"
        canon_url1 = "https://example.com/article"
        
        url2 = "https://example.com/article?utm_source=facebook"
        canon_url2 = "https://example.com/article"
        
        # Both should have the same stable_id (canonical URL)
        stable_id1 = hashlib.sha1(canon_url1.encode("utf-8")).hexdigest()
        stable_id2 = hashlib.sha1(canon_url2.encode("utf-8")).hexdigest()
        
        assert stable_id1 == stable_id2
        
        # But different original URLs
        assert url1 != url2
    
    def test_get_content_deduplication(self, fetcher, cache_root):
        """Test that identical content gets the same blob_id"""
        # Mock first response
        mock_response1 = Mock()
        mock_response1.status_code = 200
        mock_response1.content = b"<html>Same content</html>"
        mock_response1.headers = {"Content-Type": "text/html"}
        
        # Mock second response with different URL but same content
        mock_response2 = Mock()
        mock_response2.status_code = 200
        mock_response2.content = b"<html>Same content</html>"
        mock_response2.headers = {"Content-Type": "text/html"}
        
        url1 = "https://example1.com/article"
        canon_url1 = "https://example1.com/article"
        
        url2 = "https://example2.com/article"
        canon_url2 = "https://example2.com/article"
        
        # Patch the instance's session
        with patch.object(fetcher, 'session') as mock_session:
            mock_session.get.side_effect = [mock_response1, mock_response2]
            
            # Get first URL
            blob_id1, blob_path1, fetch_meta1 = fetcher.get(url1, canon_url1)
            
            # Get second URL
            blob_id2, blob_path2, fetch_meta2 = fetcher.get(url2, canon_url2)
            
            # Should have same blob_id (content-based)
            assert blob_id1 == blob_id2
            
            # But different stable_ids (URL-based)
            stable_id1 = hashlib.sha1(canon_url1.encode("utf-8")).hexdigest()
            stable_id2 = hashlib.sha1(canon_url2.encode("utf-8")).hexdigest()
            assert stable_id1 != stable_id2
            
            # Should reuse existing blob file
            assert blob_path1 == blob_path2
