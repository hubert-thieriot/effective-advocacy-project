"""
YouTube Corpus Builder Configuration Loader
"""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from dataclasses import dataclass

from .youtube import YouTubeCorpusBuilder
from ..types import BuilderParams
from ..rate_limiter import RateLimitConfig


@dataclass
class YouTubeConfig:
    """Configuration for YouTube corpus building"""
    
    # Basic info
    name: str = "YouTube Corpus"
    description: str = ""
    source: str = "youtube"
    
    # Channel/playlist configuration
    channel_url: Optional[str] = None
    playlist_url: Optional[str] = None
    channel_id: Optional[str] = None
    
    # Search parameters
    keywords: list[str] = None
    date_start: str = ""
    date_end: str = ""
    
    # All other fields with defaults
    max_videos: int = 50
    max_videos_per_keyword: int = 25
    requests_per_minute: int = 30
    min_interval: float = 2.0
    search_order: str = "date"
    include_auto_generated: bool = False
    min_duration: int = 60
    max_duration: int = 7200
    min_text_length: int = 400
    min_word_count: int = 50
    corpus_dir: str = "youtube_corpus"
    include_metadata: bool = True
    include_thumbnails: bool = False
    include_tags: bool = True
    incremental_enabled: bool = True
    force_full_rescan: bool = False
    skip_failed_transcripts: bool = True
    max_retries: int = 3
    retry_delay: int = 60
    log_level: str = "INFO"
    save_search_logs: bool = True
    log_skipped_videos: bool = True
    
    def __post_init__(self):
        """Post-initialization validation"""
        if self.keywords is None:
            self.keywords = []
        if not self.channel_url and not self.playlist_url:
            raise ValueError("Either channel URL or playlist URL is required")
        if not self.date_start or not self.date_end:
            raise ValueError("Both start and end dates are required")


class YouTubeConfigLoader:
    """Load and validate YouTube corpus builder configurations"""
    
    @staticmethod
    def load_config(config_path: Path | str) -> YouTubeConfig:
        """Load configuration from YAML file"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        return YouTubeConfigLoader._parse_config(config_data)
    
    @staticmethod
    def _parse_config(data: Dict[str, Any]) -> YouTubeConfig:
        """Parse configuration data into YouTubeConfig object"""
        
        # Extract channel/playlist info
        channel = data.get('channel', {})
        channel_url = channel.get('url')
        
        playlist = data.get('playlist', {})
        playlist_url = playlist.get('url')
        
        if not channel_url and not playlist_url:
            raise ValueError("Either channel URL or playlist URL is required in configuration")
        
        # Extract keywords (optional for playlists)
        keywords = data.get('keywords', [])
        if not keywords and not playlist_url:
            raise ValueError("At least one keyword is required for channel searches")
        
        # Extract dates
        dates = data.get('dates', {})
        date_start = dates.get('start')
        date_end = dates.get('end')
        if not date_start or not date_end:
            raise ValueError("Both start and end dates are required")
        
        # Extract limits
        limits = data.get('limits', {})
        
        # Extract rate limiting
        rate_limiting = data.get('rate_limiting', {})
        
        # Extract search options
        search = data.get('search', {})
        
        # Extract quality filters
        quality = data.get('quality', {})
        
        # Extract output options
        output = data.get('output', {})
        
        # Extract incremental options
        incremental = data.get('incremental', {})
        
        # Extract error handling
        error_handling = data.get('error_handling', {})
        
        # Extract logging
        logging = data.get('logging', {})
        
        return YouTubeConfig(
            name=data.get('name', 'YouTube Corpus'),
            description=data.get('description', ''),
            source=data.get('source', 'youtube'),
            channel_url=channel_url,
            playlist_url=playlist_url,
            channel_id=channel.get('channel_id'),
            keywords=keywords,
            date_start=date_start,
            date_end=date_end,
            max_videos=limits.get('max_videos', 50),
            max_videos_per_keyword=limits.get('max_videos_per_keyword', 25),
            requests_per_minute=rate_limiting.get('requests_per_minute', 30),
            min_interval=rate_limiting.get('min_interval', 2.0),
            search_order=search.get('order', 'date'),
            include_auto_generated=search.get('include_auto_generated', False),
            min_duration=search.get('min_duration', 60),
            max_duration=search.get('max_duration', 7200),
            min_text_length=quality.get('min_text_length', 400),
            min_word_count=quality.get('min_word_count', 50),
            corpus_dir=output.get('corpus_dir', 'youtube_corpus'),
            include_metadata=output.get('include_metadata', True),
            include_thumbnails=output.get('include_thumbnails', False),
            include_tags=output.get('include_tags', True),
            incremental_enabled=incremental.get('enabled', True),
            force_full_rescan=incremental.get('force_full_rescan', False),
            skip_failed_transcripts=error_handling.get('skip_failed_transcripts', True),
            max_retries=error_handling.get('max_retries', 3),
            retry_delay=error_handling.get('retry_delay', 60),
            log_level=logging.get('level', 'INFO'),
            save_search_logs=logging.get('save_search_logs', True),
            log_skipped_videos=logging.get('log_skipped_videos', True)
        )
    
    @staticmethod
    def create_builder(config: YouTubeConfig, corpus_dir: Optional[Path] = None) -> YouTubeCorpusBuilder:
        """Create a YouTubeCorpusBuilder instance from configuration"""
        
        # Determine corpus directory
        if corpus_dir is None:
            corpus_dir = Path(config.corpus_dir)
        
        # Create rate limit config
        rate_limit_config = RateLimitConfig(
            requests_per_minute=config.requests_per_minute,
            min_interval=config.min_interval
        )
        
        # Create builder
        builder = YouTubeCorpusBuilder(
            corpus_dir=corpus_dir,
            max_videos=config.max_videos,
            rate_limit_config=rate_limit_config
        )
        
        # Set channel or playlist info
        if config.channel_id:
            builder.channel_id = config.channel_id
        elif config.channel_url:
            # Extract from URL
            builder.channel_username = config.channel_url.split('@')[-1].split('/')[0]
        elif config.playlist_url:
            # Extract playlist ID from URL
            builder.playlist_id = builder._extract_playlist_id_from_url(config.playlist_url)
        
        return builder
    
    @staticmethod
    def create_params(config: YouTubeConfig) -> BuilderParams:
        """Create BuilderParams from configuration"""
        
        extra = {
            'force_full_rescan': config.force_full_rescan,
            'max_videos_per_keyword': config.max_videos_per_keyword,
            'search_order': config.search_order,
            'include_auto_generated': config.include_auto_generated,
            'min_duration': config.min_duration,
            'max_duration': config.max_duration,
            'min_text_length': config.min_text_length,
            'min_word_count': config.min_word_count,
            'include_thumbnails': config.include_thumbnails,
            'include_tags': config.include_tags,
            'skip_failed_transcripts': config.skip_failed_transcripts,
            'max_retries': config.max_retries,
            'retry_delay': config.retry_delay,
            'log_level': config.log_level,
            'save_search_logs': config.save_search_logs,
            'log_skipped_videos': config.log_skipped_videos
        }
        
        return BuilderParams(
            keywords=config.keywords,
            date_from=config.date_start,
            date_to=config.date_end,
            extra=extra
        )

