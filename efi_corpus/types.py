"""
Type definitions for the corpus system
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

# Legacy builder-related types stay here


@dataclass
class BuilderParams:
    """Parameters for corpus building"""
    keywords: List[str]
    date_from: str
    date_to: str
    extra: Optional[Dict[str, Any]] = None
    source_id: Optional[str] = None


@dataclass
class DiscoveryItem:
    """Item discovered during corpus building"""
    url: str
    canonical_url: str
    published_at: Optional[str] = None
    title: Optional[str] = None
    language: Optional[str] = None
    authors: Optional[List[str]] = None
    extra: Optional[Dict[str, Any]] = None


# Re-export the canonical Document from efi_core for compatibility
from efi_core.types import Document  # noqa: E402
