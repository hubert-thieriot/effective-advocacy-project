"""
YouTubeCorpusBuilder - Corpus builder for YouTube transcripts
"""

from pathlib import Path
from typing import Iterable, Dict, Any, List
import json

import re
from urllib.parse import urlparse, parse_qs
from decouple import config
import requests
from youtube_transcript_api import YouTubeTranscriptApi

from .base import BaseCorpusBuilder
from ..types import BuilderParams, DiscoveryItem
from ..fetcher import Fetcher

from ..utils import ensure_date


class SmartTranscriptFormatter:
    """Custom formatter that reconstructs proper sentences from YouTube transcript segments"""
    
    def __init__(self):
        # Common sentence endings that should be followed by a period
        self.sentence_endings = [
            r'\b(?:thank you|thanks|goodbye|bye|see you|cheers|regards|sincerely|yours|best|kind regards)\b',
            r'\b(?:yes|no|okay|ok|sure|absolutely|definitely|exactly|indeed|right|correct)\b',
            r'\b(?:uh|um|ah|oh|hmm|well|so|now|then|therefore|thus|hence)\b',
            r'\b(?:and|but|or|nor|yet|however|nevertheless|nonetheless|meanwhile|furthermore|moreover)\b'
        ]
        
        # Patterns that indicate sentence boundaries
        self.sentence_boundaries = [
            r'\.\s+[A-Z]',  # Period followed by capital letter
            r'\?\s+[A-Z]',  # Question mark followed by capital letter
            r'!\s+[A-Z]',   # Exclamation mark followed by capital letter
            r'\.\s+[0-9]',  # Period followed by number
            r'\n\s*[A-Z]',  # Newline followed by capital letter
        ]
    
    def format_transcript(self, transcript_segments: List) -> str:
        """
        Format transcript segments into readable text with proper sentence structure
        
        Args:
            transcript_segments: List of transcript segment objects (FetchedTranscriptSnippet)
            
        Returns:
            Formatted text with proper sentences and paragraphs
        """
        if not transcript_segments:
            return ""
        
        # Extract text from segments (handle both dict and object formats)
        texts = []
        for segment in transcript_segments:
            if hasattr(segment, 'text'):
                # New API: FetchedTranscriptSnippet object
                texts.append(segment.text.strip())
            elif isinstance(segment, dict) and 'text' in segment:
                # Old API: dictionary format
                texts.append(segment['text'].strip())
            else:
                # Fallback
                texts.append(str(segment).strip())
        
        # Join segments with intelligent spacing
        raw_text = self._join_segments(texts)
        
        # Clean up the text
        cleaned_text = self._clean_text(raw_text)
        
        # Reconstruct sentences
        reconstructed_text = self._reconstruct_sentences(cleaned_text)
        
        # Final cleanup and paragraph formation
        final_text = self._format_paragraphs(reconstructed_text)
        
        return final_text
    
    def _join_segments(self, texts: List[str]) -> str:
        """Join transcript segments with intelligent spacing"""
        if not texts:
            return ""
        
        joined = texts[0]
        
        for i, text in enumerate(texts[1:], 1):
            prev_text = texts[i-1]
            
            # Always add a space between segments for now
            # The sentence reconstruction will handle proper formatting
            joined += " " + text
        
        return joined
    
    def _clean_text(self, text: str) -> str:
        """Clean up common transcript artifacts"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common transcript artifacts
        text = re.sub(r'\[Music\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[Applause\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[Laughter\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[Crowd noise\]', '', text, flags=re.IGNORECASE)
        
        # Clean up common speech patterns
        text = re.sub(r'\b(?:uh|um|ah|oh)\b', '', text, flags=re.IGNORECASE)
        
        # Remove excessive punctuation
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'\?{2,}', '?', text)
        text = re.sub(r'!{2,}', '!', text)
        
        return text.strip()
    
    def _reconstruct_sentences(self, text: str) -> str:
        """Reconstruct proper sentences from transcript text"""
        # Split into potential sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        reconstructed = []
        current_sentence = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if this looks like a complete sentence
            if self._is_complete_sentence(sentence):
                if current_sentence:
                    reconstructed.append(current_sentence + " " + sentence)
                    current_sentence = ""
                else:
                    reconstructed.append(sentence)
            else:
                # This might be a sentence fragment, accumulate it
                if current_sentence:
                    current_sentence += " " + sentence
                else:
                    current_sentence = sentence
        
        # Add any remaining sentence fragment
        if current_sentence:
            reconstructed.append(current_sentence)
        
        return ". ".join(reconstructed)
    
    def _is_complete_sentence(self, text: str) -> bool:
        """Check if text looks like a complete sentence"""
        if not text:
            return False
        
        # Ends with sentence-ending punctuation
        if text.rstrip().endswith(('.', '!', '?')):
            return True
        
        # Check if it's a complete thought (ends with common endings)
        for pattern in self.sentence_endings:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        # For transcript text, be more lenient - assume most segments are complete
        # since they come from timed transcript segments
        return True
    
    def _format_paragraphs(self, text: str) -> str:
        """Format text into readable paragraphs"""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        paragraphs = []
        current_paragraph = []
        sentence_count = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            current_paragraph.append(sentence)
            sentence_count += 1
            
            # Start new paragraph after 2-3 sentences or on topic change
            if sentence_count >= 2 or self._is_topic_change(sentence):
                if current_paragraph:
                    paragraphs.append(" ".join(current_paragraph))
                    current_paragraph = []
                    sentence_count = 0
        
        # Add remaining sentences
        if current_paragraph:
            paragraphs.append(" ".join(current_paragraph))
        
        return "\n\n".join(paragraphs)
    
    def _is_topic_change(self, sentence: str) -> bool:
        """Detect potential topic changes (simplified heuristic)"""
        # Look for transition words that often indicate topic changes
        transition_words = [
            'however', 'nevertheless', 'meanwhile', 'furthermore', 'moreover',
            'on the other hand', 'in contrast', 'additionally', 'further',
            'now', 'then', 'next', 'finally', 'in conclusion'
        ]
        
        sentence_lower = sentence.lower()
        return any(word in sentence_lower for word in transition_words)


class YouTubeCorpusBuilder(BaseCorpusBuilder):
    """Corpus builder that integrates with YouTube Data API v3 for transcript extraction"""
    
    def __init__(self, corpus_dir: Path, channel_id: str = None, channel_username: str = None,
                 playlist_id: str = None, fetcher: Fetcher = None, cache_root: Path = None, max_videos: int = None):
        super().__init__(corpus_dir, fetcher, cache_root)
        self.channel_id = channel_id
        self.channel_username = channel_username
        self.playlist_id = playlist_id
        self.max_videos = max_videos  # None means no limit
        
        # No rate limiting needed - YouTube API allows 180,000 queries per minute
        
        # Initialize YouTube API key
        api_key = config('YOUTUBE_API_KEY', default=None)
        if not api_key:
            raise ValueError("YOUTUBE_API_KEY environment variable is required")
        self.api_key = api_key
        self.base_url = "https://www.googleapis.com/youtube/v3"
        
        # Initialize transcript formatter
        self.formatter = SmartTranscriptFormatter()

    def _extract_channel_id_from_url(self, url: str) -> str:
        """Extract channel ID from various YouTube URL formats"""
        parsed = urlparse(url)
        
        if parsed.netloc == "www.youtube.com":
            path_parts = parsed.path.split('/')
            
            # Handle @username format
            if len(path_parts) >= 2 and path_parts[1].startswith('@'):
                username = path_parts[1][1:]  # Remove @
                return self._get_channel_id_by_username(username)
            
            # Handle /channel/ID format
            elif len(path_parts) >= 3 and path_parts[1] == 'channel':
                return path_parts[2]
            
            # Handle /c/username format
            elif len(path_parts) >= 3 and path_parts[1] == 'c':
                username = path_parts[2]
                return self._get_channel_id_by_username(username)
        
        raise ValueError(f"Could not extract channel ID from URL: {url}")
    
    def _extract_playlist_id_from_url(self, url: str) -> str:
        """Extract playlist ID from YouTube playlist URL"""
        parsed = urlparse(url)
        
        if parsed.netloc == "www.youtube.com":
            # Handle playlist URLs
            if "playlist" in parsed.path:
                path_parts = parsed.path.split('/')
                for i, part in enumerate(path_parts):
                    if part == "playlist" and i + 1 < len(path_parts):
                        return path_parts[i + 1]
            
            # Handle playlist URLs with query parameters
            query_params = parse_qs(parsed.query)
            if 'list' in query_params:
                return query_params['list'][0]
        
        raise ValueError(f"Could not extract playlist ID from URL: {url}")

    def _get_channel_id_by_username(self, username: str) -> str:
        """Get channel ID from username using YouTube API"""
        url = f"{self.base_url}/search"
        params = {
            'part': 'snippet',
            'q': username,
            'type': 'channel',
            'key': self.api_key,
            'maxResults': 1
        }
        

        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"  Rate limit hit during channel lookup, retrying immediately...")
                # Retry once
                response = requests.get(url, params=params)
                response.raise_for_status()
            else:
                raise  # Re-raise other HTTP errors
        
        data = response.json()
        items = data.get('items', [])
        
        if not items:
            raise ValueError(f"Could not find channel ID for username: {username}")
        
        # Handle both possible response formats
        if 'id' in items[0] and 'channelId' in items[0]['id']:
            return items[0]['id']['channelId']
        elif 'snippet' in items[0] and 'channelId' in items[0]['snippet']:
            return items[0]['snippet']['channelId']
        else:
            raise ValueError(f"Unexpected response format for username: {username}")

    def discover(self, params: BuilderParams) -> Iterable[DiscoveryItem]:
        """Discover YouTube videos from the specified channel or playlist"""
        print(f"Starting corpus building process...")
        
        # Handle playlist discovery
        if self.playlist_id or (params.extra and 'playlist_url' in params.extra):
            if not self.playlist_id:
                playlist_url = params.extra['playlist_url']
                self.playlist_id = self._extract_playlist_id_from_url(playlist_url)
            
            print(f"Discovering videos from playlist: {self.playlist_id}")
            
            # Get limits from params
            max_videos = params.extra.get('max_videos', self.max_videos) if params.extra else self.max_videos
            limit_text = f"Max videos: {max_videos}" if max_videos else "No video limit"
            print(f"\n{limit_text}")
            
            # Get date range from params
            date_from = params.extra.get('date_start') if params.extra else None
            date_to = params.extra.get('date_end') if params.extra else None
            
            # Check for incremental building
            if params.extra and params.extra.get('incremental_enabled', True):
                latest_date = self._get_latest_video_date()
                if latest_date:
                    print(f"  Incremental building: continuing from {latest_date}")
                    date_from = latest_date
            
            # Fetch videos from playlist
            videos = self._fetch_playlist_videos(date_from, date_to, max_videos)
            
            all_videos = []
            for video in videos:
                video_id = video['id']['videoId']
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                
                # Skip if already in corpus
                if self.corpus.has_doc_by_url(video_url):
                    print(f"Skipping duplicate video: {video_id}")
                    continue
                
                # Check if transcript is available
                if not self._has_transcript_available(video_id):
                    print(f"Skipping video without transcript: {video_id}")
                    continue
                
                # Create discovery item
                discovery_item = DiscoveryItem(
                    url=video_url,
                    canonical_url=video_url,
                    title=video['snippet']['title'],
                    published_at=video['snippet']['publishedAt'],
                    extra={
                        'playlist_id': self.playlist_id,
                        'description': video['snippet']['description'],
                        'thumbnails': video['snippet'].get('thumbnails', {}),
                        'channel_title': video['snippet']['channelTitle']
                    }
                )
                
                all_videos.append(discovery_item)
                
                # Only check limit if max_videos is specified
                if max_videos and len(all_videos) >= max_videos:
                    print(f"Reached max videos limit ({max_videos})")
                    break
            
            print(f"Total unique videos found: {len(all_videos)}")
            return all_videos
        
        # Handle channel discovery (existing logic)
        if not self.channel_id:
            if params.extra and 'channel_url' in params.extra:
                self.channel_id = self._extract_channel_id_from_url(params.extra['channel_url'])
            else:
                raise ValueError("Either channel_id or channel_username must be provided")
        
        print(f"Discovering videos from channel: {self.channel_id}")
        
        # Get keywords from params
        keywords = params.extra.get('keywords', ['air pollution']) if params.extra else ['air pollution']
        print(f"Keywords: {keywords}")
        
        # Get limits from params
        max_videos = params.extra.get('max_videos', self.max_videos) if params.extra else self.max_videos
        limit_text = f"Max videos: {max_videos}" if max_videos else "No video limit"
        print(f"\n{limit_text}")
        
        # Get date range from params
        date_from = params.extra.get('date_start') if params.extra else None
        date_to = params.extra.get('date_end') if params.extra else None
        
        # Check for incremental building
        if params.extra and params.extra.get('incremental_enabled', True):
            latest_date = self._get_latest_video_date()
            if latest_date:
                print(f"  Incremental building: continuing from {latest_date}")
                date_from = latest_date
        
        all_videos = []
        
        # Search for each keyword
        for keyword in keywords:
            # Only check limit if max_videos is specified
            if max_videos and len(all_videos) >= max_videos:
                break
                
            print(f"Searching for keyword: {keyword}")
            remaining_limit = max_videos - len(all_videos) if max_videos else None
            videos = self._search_channel_videos(keyword, date_from, date_to, remaining_limit)
            
            for video in videos:
                video_id = video['id']['videoId']
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                
                # Skip if already in corpus
                if self.corpus.has_doc_by_url(video_url):
                    print(f"Skipping duplicate video: {video_id}")
                    continue
                
                # Check if transcript is available
                if not self._has_transcript_available(video_id):
                    print(f"Skipping video without transcript: {video_id}")
                    continue
                
                # Create discovery item
                discovery_item = DiscoveryItem(
                    url=video_url,
                    canonical_url=video_url,
                    title=video['snippet']['title'],
                    published_at=video['snippet']['publishedAt'],
                    authors=[video['snippet']['channelTitle']],  # Set authors from channel title
                    extra={
                        'channel_id': self.channel_id,
                        'keyword': keyword,
                        'description': video['snippet']['description'],
                        'thumbnails': video['snippet'].get('thumbnails', {}),
                        'channel_title': video['snippet']['channelTitle']
                    }
                )
                
                all_videos.append(discovery_item)
                
                # Only check limit if max_videos is specified
                if max_videos and len(all_videos) >= max_videos:
                    print(f"Reached max videos limit ({max_videos})")
                    break
        
        print(f"Total unique videos found: {len(all_videos)}")
        return all_videos

    def _search_channel_videos(self, keyword: str, date_from: str = None, date_to: str = None, max_videos: int = None) -> List[Dict]:
        """Search for videos in a channel with the given keyword"""
        url = f"{self.base_url}/search"
        params = {
            'part': 'snippet',
            'channelId': self.channel_id,
            'q': keyword,
            'type': 'video',
            'key': self.api_key,
            'maxResults': 50,  # YouTube API max per request
            'order': 'date'  # Changed from 'relevance' to 'date' to go backward in time
        }
        
        # Add date filtering if provided
        if date_from:
            params['publishedAfter'] = date_from + 'T00:00:00Z'
        if date_to:
            params['publishedBefore'] = date_to + 'T23:59:59Z'
        
        all_videos = []
        next_page_token = None
        
        # Continue fetching until no more pages or limit reached
        while True:
            if next_page_token:
                params['pageToken'] = next_page_token
            
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    print(f"  Rate limit hit during search, retrying immediately...")
                    continue  # Retry this request
                else:
                    raise  # Re-raise other HTTP errors
            
            data = response.json()
            videos = data.get('items', [])
            all_videos.extend(videos)
            
            # Check if there are more pages
            next_page_token = data.get('nextPageToken')
            if not next_page_token:
                break
            
            # Check limit only if max_videos is specified
            if max_videos and len(all_videos) >= max_videos:
                all_videos = all_videos[:max_videos]
                break
        
        return all_videos
    
    def _fetch_playlist_videos(self, date_from: str = None, date_to: str = None, max_videos: int = None) -> List[Dict]:
        """Fetch videos from a playlist with date filtering"""
        if not self.playlist_id:
            raise ValueError("Playlist ID is required to fetch playlist videos")
        
        url = f"{self.base_url}/playlistItems"
        params = {
            'part': 'snippet',
            'playlistId': self.playlist_id,
            'key': self.api_key,
            'maxResults': 50,  # YouTube API max per request
        }
        
        all_videos = []
        next_page_token = None
        
        print(f"  Fetching playlist videos with date filter: {date_from} to {date_to}")
        
        # Continue fetching until no more pages or limit reached
        while True:
            if next_page_token:
                params['pageToken'] = next_page_token
            
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    print(f"  Rate limit hit during playlist fetch, retrying immediately...")
                    continue  # Retry this request
                else:
                    raise  # Re-raise other HTTP errors
            
            data = response.json()
            items = data.get('items', [])
            
            # Convert playlist items to video format and apply date filtering
            for item in items:
                published_at = item['snippet'].get('publishedAt')
                
                # Apply date filtering if dates are specified
                if date_from and published_at:
                    if published_at < date_from:
                        print(f"  Skipping video {item['snippet']['resourceId']['videoId']} - too old: {published_at}")
                        continue
                
                if date_to and published_at:
                    if published_at > date_to:
                        print(f"  Skipping video {item['snippet']['resourceId']['videoId']} - too new: {published_at}")
                        continue
                
                video_data = {
                    'id': {'videoId': item['snippet']['resourceId']['videoId']},
                    'snippet': item['snippet']
                }
                all_videos.append(video_data)
                
                # Check limit only if max_videos is specified
                if max_videos and len(all_videos) >= max_videos:
                    print(f"  Reached max videos limit ({max_videos})")
                    break
            
            # If we've reached the limit, stop fetching more pages
            if max_videos and len(all_videos) >= max_videos:
                break
            
            # Check if there are more pages
            next_page_token = data.get('nextPageToken')
            if not next_page_token:
                break
        
        print(f"  Found {len(all_videos)} videos after date filtering")
        return all_videos

    def _has_transcript_available(self, video_id: str) -> bool:
        """Check if a video has transcripts available"""
        try:
            transcript_list = YouTubeTranscriptApi().list(video_id)
            # Count by iterating since TranscriptList doesn't have len()
            for _ in transcript_list:
                return True  # Found at least one transcript
            return False
        except Exception as e:
            print(f"  No transcripts available for {video_id}: {e}")
            return False
        
    def _fetch_transcript(self, video_id: str) -> str:
        """Fetch transcript for a video"""
        try:
            transcript_list = YouTubeTranscriptApi().list(video_id)
            # Get the first available transcript
            for transcript in transcript_list:
                return transcript
        except Exception:
            pass
        return None

    def _get_latest_video_date(self) -> str:
        """Get the latest video date from the existing corpus"""
        try:
            if not self.corpus.layout.index_path.exists():
                return None
            
            latest_date = None
            with open(self.corpus.layout.index_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            doc_data = json.loads(line)
                            published_at = doc_data.get('published_at')
                            if published_at and (latest_date is None or published_at > latest_date):
                                latest_date = published_at
                        except json.JSONDecodeError:
                            continue
            
            return latest_date
        except Exception:
            return None

    def fetch_raw(self, url: str, stable_id: str) -> tuple[bytes, Dict[str, Any], str]:
        """
        Fetch raw transcript data for a YouTube video
        """
        # Extract video ID from URL
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)
        video_id = query_params.get('v', [None])[0]
        
        if not video_id:
            raise ValueError(f"Could not extract video ID from URL: {url}")
        
        print(f"Fetching transcript for video: {video_id}")
        
        # No rate limiting needed - YouTube API allows 180,000 queries per minute
        
        try:
            # First check if transcripts are available
            transcript_list = YouTubeTranscriptApi().list(video_id)
            
            # Look for English transcripts first, then any available
            transcript = None
            try:
                transcript = transcript_list.find_transcript(['en'])
            except:
                # Try to get any available transcript
                available_transcripts = list(transcript_list)
                if available_transcripts:
                    transcript = available_transcripts[0]
                    print(f"  Using transcript in language: {transcript.language_code}")
                else:
                    print(f"  No transcripts available for {video_id}")
                    return b"", {'error': 'No transcripts available', 'video_id': video_id}, "txt"
            
            # Get the transcript content
            transcript_data = transcript.fetch()
            
            # Format transcript as text
            transcript_text = self.formatter.format_transcript(transcript_data)
            
            # Create metadata
            fetch_meta = {
                'video_id': video_id,
                'transcript_segments': len(transcript_data),
                'total_duration': sum(segment.duration for segment in transcript_data),
                'languages': [transcript.language_code],
                'transcript_type': 'manual' if not transcript.is_generated else 'auto-generated'
            }
            
            print(f"  Successfully fetched transcript: {len(transcript_text)} characters")
            # Return as bytes with metadata
            return transcript_text.encode('utf-8'), fetch_meta, "txt"
            
        except Exception as e:
            error_str = str(e)
            print(f"Error fetching transcript for {video_id}: {e}")
            
            # Handle specific error types with different strategies (like MediaCloud)
            if "429" in error_str or "Too Many Requests" in error_str:
                print(f"  Rate limit hit, retrying immediately...")
                # Try one more time immediately
                try:
                    transcript_list = YouTubeTranscriptApi().list(video_id)
                    # Get the first available transcript
                    for transcript in transcript_list:
                        transcript_data = transcript.fetch()
                        transcript_text = self.formatter.format_transcript(transcript_data)
                        
                        fetch_meta = {
                            'video_id': video_id,
                            'transcript_segments': len(transcript_data),
                            'total_duration': sum(segment.duration for segment in transcript_data),
                            'languages': [transcript.language_code],
                            'transcript_type': 'manual' if not transcript.is_generated else 'auto-generated'
                        }
                        
                        print(f"  Successfully fetched transcript on retry: {len(transcript_text)} characters")
                        return transcript_text.encode('utf-8'), fetch_meta, "txt"
                    print(f"  No transcripts available on retry")
                except Exception as retry_e:
                    print(f"  Retry failed: {retry_e}")
            
            # Return empty transcript if there's an error
            return b"", {'error': str(e), 'video_id': video_id}, "txt"

    def parse_text(self, raw_bytes: bytes, raw_ext: str, url: str) -> Dict[str, Any]:
        """
        Parse raw transcript data into structured format
        """
        # Extract video ID from URL for metadata
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)
        video_id = query_params.get('v', [None])[0]
        
        # Get video metadata from the discovery phase
        # This would ideally be stored and retrieved, but for now we'll parse what we have
        transcript_text = raw_bytes.decode('utf-8') if raw_bytes else ""
        
        return {
            "text": transcript_text,
            "title": None,  # Will be populated from discovery metadata
            "published_at": None,  # Will be populated from discovery metadata
            "language": "en",  # Default, could be detected from transcript
            "authors": None,  # Will be populated from discovery metadata
            "extra": {
                'video_id': video_id,
                'transcript_length': len(transcript_text),
                'word_count': len(transcript_text.split()) if transcript_text else 0
            }
        }

    # ---------- utility methods ----------
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

    def _apply_filters(self, items: list[DiscoveryItem], params: BuilderParams) -> list[DiscoveryItem]:
        """Apply common filtering logic to discovered items"""
        domain_blacklist = (params.extra or {}).get('domain_blacklist', [])
        url_blacklist = (params.extra or {}).get('url_blacklist', [])

        if domain_blacklist:
            original_count = len(items)
            items = [item for item in items if not self._is_domain_blacklisted(item.url, domain_blacklist)]
            filtered_count = original_count - len(items)
            if filtered_count > 0:
                print(f"Filtered out {filtered_count} items from blacklisted domains")

        if url_blacklist:
            original_count = len(items)
            items = [item for item in items if not self._is_url_blacklisted(item.url, url_blacklist)]
            filtered_count = original_count - len(items)
            if filtered_count > 0:
                print(f"Filtered out {filtered_count} items with blacklisted URL patterns")

        return items

    # ---------- main run method ----------
    def run(self, *, params: BuilderParams | None = None, override: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Run the corpus builder

        If params is None, load from manifest. If override provided, overlay keys.
        Idempotent: only adds new docs.
        """
        from hashlib import sha1
        import time

        # Common tracking parameters to strip from URLs
        TRACKERS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "gclid", "fbclid"}

        def canonicalize(url: str) -> str:
            """Canonicalize a URL by removing tracking parameters and normalizing"""
            from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
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

        manifest = self.corpus.load_manifest()
        if params is None:
            if not manifest:
                raise ValueError("No manifest found; first run must provide params.")
            from ..types import BuilderParams
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

        # Apply filtering
        discovered = self._apply_filters(discovered, params)

        # Map URL to discovery item for later metadata enrichment
        discovered_by_url = {d.url: d for d in discovered}

        # Compute frontier (items not already in corpus)
        def sid(u: str) -> str:
            return sha1(canonicalize(u).encode("utf-8")).hexdigest()

        pairs = [(d.url, sid(d.url)) for d in discovered]
        frontier = [(u, s) for (u, s) in pairs if not self.corpus.has_doc(s)]

        # YouTube builder currently only supports sequential processing
        return self._process_sequential(frontier, discovered_by_url, params, manifest)

    def _process_sequential(self, frontier, discovered_by_url, params, manifest) -> Dict[str, Any]:
        """Process items sequentially (YouTube's current approach)"""
        import time

        # Process frontier
        added = 0
        failed_urls = []
        skipped_quality = 0
        skipped_text_extraction = 0
        skipped_duplicate = len(discovered_by_url) - len(frontier)  # Already in corpus

        for i, (url, stable_id) in enumerate(frontier, 1):
            try:
                print(f"Processing URL ({i}/{len(frontier)}): {url}")

                raw_bytes, fetch_meta, raw_ext = self.fetch_raw(url, stable_id)
                parsed = self.parse_text(raw_bytes, raw_ext, url)
                text = parsed.get("text") or ""

                # Track text extraction failures
                if not text:
                    print(f"  ⚠️  Skipped: No text extracted")
                    skipped_text_extraction += 1
                    continue

                # Quality gate (YouTube transcripts might be shorter, so lower threshold)
                if len(text) < 200:
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

        # Update manifest
        manifest["history"].append({
            "run_at": time.time(),
            "discovered": len(discovered_by_url),
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

        # Print comprehensive summary
        print(f"\n📊 Build Summary:")
        print(f"  Discovered: {len(discovered_by_url)}")
        print(f"  Added: {added}")
        print(f"  Skipped (quality): {skipped_quality}")
        print(f"  Skipped (text extraction): {skipped_text_extraction}")
        print(f"  Skipped (duplicate): {skipped_duplicate}")
        print(f"  Failed: {len(failed_urls)}")
        print(f"  Total docs in corpus: {self.corpus.get_document_count()}")

        # Print detailed failure summary
        if failed_urls:
            print(f"\n⚠️  {len(failed_urls)} URLs failed to process:")
            for failed in failed_urls[:5]:  # Show first 5 failures
                print(f"  - {failed['url']}: {failed['error']}")
            if len(failed_urls) > 5:
                print(f"  ... and {len(failed_urls) - 5} more")

        return {
            "discovered": len(discovered_by_url),
            "added": added,
            "skipped_quality": skipped_quality,
            "skipped_text_extraction": skipped_text_extraction,
            "skipped_duplicate": skipped_duplicate,
            "failed": len(failed_urls),
            "total_docs": self.corpus.get_document_count(),
            "failed_details": failed_urls
        }
