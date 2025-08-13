"""
Tests for MediaCloudCorpusBuilder
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import date, timedelta

from efi_corpus.builders.mediacloud import MediaCloudCorpusBuilder
from efi_corpus.types import BuilderParams, DiscoveryItem
from efi_corpus.fetcher import Fetcher
from efi_corpus.rate_limiter import RateLimitConfig


class TestMediaCloudCorpusBuilder:
    """Test cases for MediaCloudCorpusBuilder"""
    
    @pytest.fixture
    def mock_fetcher(self):
        """Create a mock fetcher"""
        fetcher = Mock(spec=Fetcher)
        fetcher.get.return_value = ("mock_blob_id", Path("/mock/blob/path"), {"status": 200, "mime": "text/html"})
        return fetcher
    
    @pytest.fixture
    def rate_limit_config(self):
        """Create a rate limit config with rate limiting disabled for testing"""
        return RateLimitConfig(enabled=False)
    
    @pytest.fixture
    def mock_corpus_dir(self, tmp_path):
        """Create a temporary corpus directory"""
        corpus_dir = tmp_path / "test_corpus"
        corpus_dir.mkdir()
        return corpus_dir
    
    @pytest.fixture
    def sample_stories(self):
        """Sample MediaCloud stories for testing"""
        return [
            {
                "id": "story1",
                "title": "Climate Change Impact on Cities",
                "url": "https://example.com/climate-article",
                "guid": "https://example.com/climate-article",
                "publish_date": 1735689600,  # 2025-01-01
                "language": "en",
                "author": "John Doe",
                "stories_id": 12345,
                "media_id": 678,
                "media_name": "Example News",
                "media_url": "example.com"
            },
            {
                "id": "story2", 
                "title": "Renewable Energy Solutions",
                "url": "https://example.com/energy-article",
                "guid": "https://example.com/energy-article",
                "publish_date": 1735776000,  # 2025-01-02
                "language": "en",
                "author": "Jane Smith",
                "stories_id": 12346,
                "media_id": 679,
                "media_name": "Tech News",
                "media_url": "technews.com"
            }
        ]
    
    @pytest.fixture
    def builder_params(self):
        """Sample builder parameters"""
        return BuilderParams(
            keywords=["climate change", "renewable energy"],
            date_from="2025-01-01",
            date_to="2025-01-31",
            extra={"collection_id": 1, "collection_name": "Test Collection"}
        )
    
    @patch('efi_corpus.builders.mediacloud.config')
    @patch('efi_corpus.builders.mediacloud.api')
    def test_init_with_api_key(self, mock_api, mock_config, mock_fetcher, mock_corpus_dir):
        """Test initialization with valid API key"""
        mock_config.return_value = "test_api_key"
        mock_search_api = Mock()
        mock_api.SearchApi.return_value = mock_search_api
        
        # Disable rate limiting for testing
        rate_limit_config = RateLimitConfig(enabled=False)
        
        builder = MediaCloudCorpusBuilder(
            mock_corpus_dir, 
            collection_id=1, 
            collection_name="Test Collection",
            rate_limit_config=rate_limit_config,
            fetcher=mock_fetcher
        )
        
        assert builder.collection_id == 1
        assert builder.collection_name == "Test Collection"
        assert builder.mc_api == mock_search_api
        assert not builder.rate_limiter.config.enabled  # Rate limiting should be disabled
        mock_api.SearchApi.assert_called_once_with("test_api_key")
    
    @patch('efi_corpus.builders.mediacloud.config')
    def test_init_missing_api_key(self, mock_config, mock_fetcher, mock_corpus_dir, rate_limit_config):
        """Test initialization fails without API key"""
        mock_config.return_value = None
        
        with pytest.raises(ValueError, match="MEDIACLOUD_KEY environment variable is required"):
            MediaCloudCorpusBuilder(mock_corpus_dir, rate_limit_config=rate_limit_config, fetcher=mock_fetcher)
    
    @patch('efi_corpus.builders.mediacloud.config')
    @patch('efi_corpus.builders.mediacloud.api')
    def test_discover_success(self, mock_api, mock_config, mock_fetcher, mock_corpus_dir, 
                             sample_stories, builder_params, rate_limit_config):
        """Test successful discovery of stories"""
        mock_config.return_value = "test_api_key"
        mock_search_api = Mock()
        mock_api.SearchApi.return_value = mock_search_api
        
        # Mock the story_list method to return our sample stories
        mock_search_api.story_list.return_value = (sample_stories, None)
        
        builder = MediaCloudCorpusBuilder(mock_corpus_dir, rate_limit_config=rate_limit_config, fetcher=mock_fetcher)
        
        # Test discovery
        discovered = list(builder.discover(builder_params))
        
        assert len(discovered) == 2
        assert all(isinstance(item, DiscoveryItem) for item in discovered)
        
        # Check first story
        first = discovered[0]
        assert first.url == "https://example.com/climate-article"
        assert first.title == "Climate Change Impact on Cities"
        assert first.published_at == "2025-01-01"
        assert first.language == "en"
        assert first.authors == ["John Doe"]
        assert first.extra["story_id"] == 12345
        assert first.extra["collection"] == "Test Collection"
        
        # Check second story
        second = discovered[1]
        assert second.url == "https://example.com/energy-article"
        assert second.title == "Renewable Energy Solutions"
        assert second.published_at == "2025-01-02"
    
    @patch('efi_corpus.builders.mediacloud.config')
    @patch('efi_corpus.builders.mediacloud.api')
    def test_discover_with_pagination(self, mock_api, mock_config, mock_fetcher, mock_corpus_dir,
                                     builder_params, sample_stories, rate_limit_config):
        """Test discovery with pagination"""
        mock_config.return_value = "test_api_key"
        mock_search_api = Mock()
        mock_api.SearchApi.return_value = mock_search_api
        
        # Mock pagination: first call returns stories with token, second returns empty
        mock_search_api.story_list.side_effect = [
            (sample_stories, "token123"),
            ([], None)
        ]
        
        builder = MediaCloudCorpusBuilder(mock_corpus_dir, rate_limit_config=rate_limit_config, fetcher=mock_fetcher)
        
        discovered = list(builder.discover(builder_params))
        
        assert len(discovered) == 2
        assert mock_search_api.story_list.call_count == 2
        
        # Check that pagination token was used
        calls = mock_search_api.story_list.call_args_list
        assert calls[0][1]['pagination_token'] is None
        assert calls[1][1]['pagination_token'] == "token123"
    
    @patch('efi_corpus.builders.mediacloud.config')
    @patch('efi_corpus.builders.mediacloud.api')
    def test_discover_api_error_retry(self, mock_api, mock_config, mock_fetcher, mock_corpus_dir,
                                     builder_params, sample_stories, rate_limit_config):
        """Test discovery retries on 403 errors"""
        mock_config.return_value = "test_api_key"
        mock_search_api = Mock()
        mock_api.SearchApi.return_value = mock_search_api
        
        # Mock 403 error first, then success
        mock_search_api.story_list.side_effect = [
            Exception("403 Forbidden"),
            (sample_stories[:1], None)
        ]
        
        builder = MediaCloudCorpusBuilder(mock_corpus_dir, rate_limit_config=rate_limit_config, fetcher=mock_fetcher)
        
        discovered = list(builder.discover(builder_params))
        
        assert len(discovered) == 1
        assert mock_search_api.story_list.call_count == 2
    
    @patch('efi_corpus.builders.mediacloud.config')
    @patch('efi_corpus.builders.mediacloud.api')
    def test_discover_missing_collection_id(self, mock_api, mock_config, mock_fetcher, mock_corpus_dir, rate_limit_config):
        """Test discovery fails without collection_id"""
        mock_config.return_value = "test_api_key"
        mock_api.SearchApi.return_value = Mock()
        
        builder = MediaCloudCorpusBuilder(mock_corpus_dir, rate_limit_config=rate_limit_config, fetcher=mock_fetcher)
        params = BuilderParams(
            keywords=["test"],
            date_from="2025-01-01",
            date_to="2025-01-31"
        )
        
        with pytest.raises(ValueError, match="collection_id must be provided"):
            list(builder.discover(params))
    
    @patch('efi_corpus.builders.mediacloud.config')
    @patch('efi_corpus.builders.mediacloud.api')
    def test_discover_story_without_url(self, mock_api, mock_config, mock_fetcher, mock_corpus_dir,
                                       builder_params, rate_limit_config):
        """Test discovery skips stories without URLs"""
        mock_config.return_value = "test_api_key"
        mock_search_api = Mock()
        mock_api.SearchApi.return_value = mock_search_api
        
        # Story without URL
        stories_without_url = [{"id": "story1", "title": "No URL Story"}]
        mock_search_api.story_list.return_value = (stories_without_url, None)
        
        builder = MediaCloudCorpusBuilder(mock_corpus_dir, rate_limit_config=rate_limit_config, fetcher=mock_fetcher)
        
        discovered = list(builder.discover(builder_params))
        
        assert len(discovered) == 0
    
    def test_fetch_raw(self, mock_fetcher, mock_corpus_dir, rate_limit_config):
        """Test fetching raw content"""
        with patch('efi_corpus.builders.mediacloud.config') as mock_config, \
             patch('efi_corpus.builders.mediacloud.api') as mock_api:
            
            mock_config.return_value = "test_api_key"
            mock_api.SearchApi.return_value = Mock()
            
            builder = MediaCloudCorpusBuilder(mock_corpus_dir, rate_limit_config=rate_limit_config, fetcher=mock_fetcher)
            
            # Mock file read with proper context manager
            mock_file = Mock()
            mock_file.read.return_value = b"compressed_content"
            mock_file.__enter__ = Mock(return_value=mock_file)
            mock_file.__exit__ = Mock(return_value=None)
            
            with patch('builtins.open', return_value=mock_file):
                raw_bytes, fetch_meta, raw_ext = builder.fetch_raw("https://example.com", "test_id")
            
            assert raw_bytes == b"compressed_content"
            assert fetch_meta == {"status": 200, "mime": "text/html"}
            assert raw_ext == "html"  # Based on mime type
    
    def test_parse_text_html(self, mock_fetcher, mock_corpus_dir, rate_limit_config):
        """Test parsing HTML content"""
        with patch('efi_corpus.builders.mediacloud.config') as mock_config, \
             patch('efi_corpus.builders.mediacloud.api') as mock_api:
            
            mock_config.return_value = "test_api_key"
            mock_api.SearchApi.return_value = Mock()
            
            builder = MediaCloudCorpusBuilder(mock_corpus_dir, rate_limit_config=rate_limit_config, fetcher=mock_fetcher)
            
            # Test with HTML content (longer to pass validation)
            html_content = b"<html><head><title>Test Title</title></head><body><p>This is a much longer test content that should be sufficient to pass the minimum length requirements for text extraction. It needs to be at least 200 characters long and contain enough words to be considered valid content for testing purposes.</p></body></html>"
            result = builder.parse_text(html_content, "html", "https://example.com")
            
            assert "test content" in result["text"].lower()
            assert result["title"] == "Test Title"  # Now extracted from HTML
            assert result["language"] == "en"
            assert result["authors"] == []
    
    def test_parse_text_pdf(self, mock_fetcher, mock_corpus_dir, rate_limit_config):
        """Test parsing PDF content"""
        with patch('efi_corpus.builders.mediacloud.config') as mock_config, \
             patch('efi_corpus.builders.mediacloud.api') as mock_api:
            
            mock_config.return_value = "test_api_key"
            mock_api.SearchApi.return_value = Mock()
            
            builder = MediaCloudCorpusBuilder(mock_corpus_dir, rate_limit_config=rate_limit_config, fetcher=mock_fetcher)
            
            # Test with PDF content (simulated, longer to pass validation)
            # Note: PDF extraction is not implemented yet, so expect empty result
            pdf_content = b"%PDF-1.4\nThis is a much longer test PDF content that should be sufficient to pass the minimum length requirements for text extraction. It needs to be at least 200 characters long and contain enough words to be considered valid content for testing purposes."
            result = builder.parse_text(pdf_content, "pdf", "https://example.com")
            
            # PDF extraction not implemented yet, so text should be empty
            assert result["text"] == ""
            assert result["title"] is None
    
    def test_parse_text_uncompressed(self, mock_fetcher, mock_corpus_dir, rate_limit_config):
        """Test parsing uncompressed content"""
        with patch('efi_corpus.builders.mediacloud.config') as mock_config, \
             patch('efi_corpus.builders.mediacloud.api') as mock_api:
            
            mock_config.return_value = "test_api_key"
            mock_api.SearchApi.return_value = Mock()
            
            builder = MediaCloudCorpusBuilder(mock_corpus_dir, rate_limit_config=rate_limit_config, fetcher=mock_fetcher)
            
            # Test with uncompressed content (longer to pass validation)
            uncompressed_content = b"This is a much longer plain text content that should be sufficient to pass the minimum length requirements for text extraction. It needs to be at least 200 characters long and contain enough words to be considered valid content for testing purposes."
            result = builder.parse_text(uncompressed_content, "txt", "https://example.com")
            
            assert "plain text content" in result["text"].lower()


# Helper function for mocking file operations
def mock_open(mock=None, read_data=None):
    """Mock open function for testing"""
    if mock is None:
        mock = MagicMock()
    
    if read_data is not None:
        mock.return_value.read.return_value = read_data
    
    return mock
