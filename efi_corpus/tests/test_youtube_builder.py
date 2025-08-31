"""
Tests for YouTubeCorpusBuilder
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from efi_corpus.builders.youtube import YouTubeCorpusBuilder
from efi_corpus.types import BuilderParams, DiscoveryItem


class TestYouTubeCorpusBuilder:
    """Test cases for YouTubeCorpusBuilder"""
    
    @pytest.fixture
    def temp_corpus_dir(self, tmp_path):
        """Create a temporary corpus directory"""
        return tmp_path / "test_youtube_corpus"
    
    @pytest.fixture
    def mock_fetcher(self):
        """Create a mock fetcher"""
        return Mock()
    
    @pytest.fixture
    def builder(self, temp_corpus_dir, mock_fetcher):
        """Create a YouTubeCorpusBuilder instance for testing"""
        with patch.dict('os.environ', {'YOUTUBE_API_KEY': 'test_key'}):
            return YouTubeCorpusBuilder(
                corpus_dir=temp_corpus_dir,
                fetcher=mock_fetcher,
                max_videos=5
            )
    
    def test_init_without_api_key(self, temp_corpus_dir):
        """Test that builder fails without API key"""
        with patch('efi_corpus.builders.youtube.config', return_value=None):
            with pytest.raises(ValueError, match="YOUTUBE_API_KEY environment variable is required"):
                YouTubeCorpusBuilder(corpus_dir=temp_corpus_dir)
    
    def test_extract_channel_id_from_url_username(self, builder):
        """Test extracting channel ID from @username format"""
        with patch.object(builder, '_get_channel_id_by_username', return_value='UC123456'):
            channel_id = builder._extract_channel_id_from_url('https://www.youtube.com/@NDTV')
            assert channel_id == 'UC123456'
    
    def test_extract_channel_id_from_url_channel_format(self, builder):
        """Test extracting channel ID from /channel/ID format"""
        channel_id = builder._extract_channel_id_from_url('https://www.youtube.com/channel/UC123456')
        assert channel_id == 'UC123456'
    
    def test_extract_channel_id_from_url_invalid(self, builder):
        """Test extracting channel ID from invalid URL"""
        with pytest.raises(ValueError, match="Could not extract channel ID from URL"):
            builder._extract_channel_id_from_url('https://invalid.com')
    
    def test_get_channel_id_by_username(self, builder):
        """Test getting channel ID by username via API"""
        mock_response = Mock()
        mock_response.json.return_value = {
            'items': [{'snippet': {'channelId': 'UC123456'}}]
        }
        mock_response.raise_for_status.return_value = None
        
        with patch('requests.get', return_value=mock_response):
            channel_id = builder._get_channel_id_by_username('NDTV')
            assert channel_id == 'UC123456'
    
    def test_get_channel_id_by_username_not_found(self, builder):
        """Test getting channel ID when username not found"""
        mock_response = Mock()
        mock_response.json.return_value = {'items': []}
        mock_response.raise_for_status.return_value = None
        
        with patch('requests.get', return_value=mock_response):
            with pytest.raises(ValueError, match="Could not find channel ID for username"):
                builder._get_channel_id_by_username('NonexistentChannel')
    
    def test_discover_without_channel_info(self, builder):
        """Test discover fails without channel information"""
        params = BuilderParams(
            keywords=['test'],
            date_from='2024-01-01',
            date_to='2024-12-31'
        )
        
        with pytest.raises(ValueError, match="Either channel_id or channel_username must be provided"):
            list(builder.discover(params))
    
    def test_discover_with_channel_id(self, builder):
        """Test discover with channel ID"""
        builder.channel_id = 'UC123456'
        
        # Mock the search results
        mock_videos = [
            {
                'id': {'videoId': 'video1'},
                'snippet': {
                    'title': 'Test Video 1',
                    'publishedAt': '2024-01-01T00:00:00Z',
                    'channelTitle': 'Test Channel',
                    'defaultLanguage': 'en',
                    'description': 'Test description',
                    'thumbnails': {},
                    'tags': [],
                    'categoryId': '22'
                }
            }
        ]
        
        # Mock the transcript availability check to return True
        with patch.object(builder, '_search_channel_videos', return_value=mock_videos), \
             patch.object(builder, '_has_transcript_available', return_value=True):
            params = BuilderParams(
                keywords=['test'],
                date_from='2024-01-01',
                date_to='2024-12-31'
            )
            
            items = list(builder.discover(params))
            assert len(items) == 1
            assert items[0].url == 'https://www.youtube.com/watch?v=video1'
            assert items[0].title == 'Test Video 1'
            assert items[0].authors == ['Test Channel']
    
    def test_fetch_raw_success(self, builder):
        """Test successful transcript fetching"""
        url = 'https://www.youtube.com/watch?v=test123'
        
        mock_transcript = [
            {'text': 'Hello world', 'start': 0, 'duration': 2},
            {'text': 'This is a test', 'start': 2, 'duration': 3}
        ]
        
        # Mock the new API structure
        mock_transcript = Mock()
        
        # Create mock transcript segments with proper structure
        mock_segment1 = Mock()
        mock_segment1.text = 'Hello world'
        mock_segment1.duration = 2
        
        mock_segment2 = Mock()
        mock_segment2.text = 'This is a test'
        mock_segment2.duration = 3
        
        mock_transcript.fetch.return_value = [mock_segment1, mock_segment2]
        mock_transcript.language_code = 'en'
        mock_transcript.is_generated = False
        
        # Mock the list method to return our mock transcript
        with patch('youtube_transcript_api.YouTubeTranscriptApi.list', return_value=[mock_transcript]):
            raw_bytes, fetch_meta, raw_ext = builder.fetch_raw(url, 'test123')
            
            assert raw_ext == 'txt'
            assert b'Hello world' in raw_bytes
            assert b'This is a test' in raw_bytes
            assert fetch_meta['video_id'] == 'test123'
            assert fetch_meta['transcript_segments'] == 2
    
    def test_fetch_raw_failure(self, builder):
        """Test transcript fetching failure"""
        url = 'https://www.youtube.com/watch?v=test123'
        
        # Mock the new API structure with an exception
        with patch('youtube_transcript_api.YouTubeTranscriptApi.list', 
                  side_effect=Exception('Transcript not available')):
            raw_bytes, fetch_meta, raw_ext = builder.fetch_raw(url, 'test123')
            
            assert raw_bytes == b''
            assert fetch_meta['error'] == 'Transcript not available'
            assert raw_ext == 'txt'
    
    def test_parse_text(self, builder):
        """Test transcript text parsing"""
        url = 'https://www.youtube.com/watch?v=test123'
        raw_bytes = b'Hello world. This is a test transcript.'
        
        result = builder.parse_text(raw_bytes, 'txt', url)
        
        assert result['text'] == 'Hello world. This is a test transcript.'
        assert result['language'] == 'en'
        assert result['extra']['video_id'] == 'test123'
        assert result['extra']['transcript_length'] == 39
        assert result['extra']['word_count'] == 7
