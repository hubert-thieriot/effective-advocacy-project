"""
Types for the efi_findings module
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pathlib import Path


# Type alias for date fields to make intent explicit
DateField = Union[datetime, str, None]


@dataclass
class Finding:
    """A single key finding 
    extracted from a document"""
    text: str
    confidence: Optional[float] = None
    category: Optional[str] = None
    keywords: Optional[List[str]] = None


@dataclass
class DocumentFindings:
    """Findings extracted from a single document"""
    url: str
    title: Optional[str] = None
    published_at: Optional[datetime] = None  # Always stored as datetime
    language: Optional[str] = None
    extraction_date: datetime = field(default_factory=datetime.now)
    findings: List[Finding] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Ensure published_at is always a datetime object or None
        from .utils import normalize_date
        self.published_at = normalize_date(self.published_at)





@dataclass
class ExtractionConfig:
    """Configuration for LLM-based extraction"""
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 1000
    temperature: float = 0.1
    system_prompt: Optional[str] = None
    extraction_prompt: Optional[str] = None


@dataclass
class StorageConfig:
    """Configuration for findings storage"""
    storage_path: Path = Path("findings")
    index_filename: str = "findings_index.jsonl"
    findings_dir: str = "findings"
    backup_enabled: bool = True
