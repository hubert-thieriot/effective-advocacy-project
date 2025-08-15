"""
Type definitions for the corpus system
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Protocol
from pathlib import Path
import json
import hashlib


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


@dataclass
class Document:
    """Represents a document from a corpus"""
    doc_id: str
    url: str
    title: Optional[str]
    text: str
    published_at: Optional[str]
    language: Optional[str]
    meta: Dict[str, Any]


# =============================
# New types for chunking and embedding architecture
# =============================

@dataclass(frozen=True)
class ChunkerSpec:
    """Specification for a text chunking strategy"""
    name: str
    params: Dict[str, Any]

    def key(self) -> str:
        """Generate a stable hash key for this chunker configuration"""
        s = json.dumps({"name": self.name, "params": self.params}, sort_keys=True)
        return hashlib.sha1(s.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class EmbedderSpec:
    """Specification for an embedding model"""
    model_name: str
    dim: int
    revision: Optional[str] = None  # e.g., HF commit hash

    def key(self) -> str:
        """Generate a stable hash key for this embedder configuration"""
        s = json.dumps({"model": self.model_name, "dim": self.dim, "rev": self.revision}, sort_keys=True)
        return hashlib.sha1(s.encode()).hexdigest()[:12]


@dataclass
class DocState:
    """Tracks the processing state of a document for incremental rebuilds"""
    doc_id: str
    fingerprint: str
    last_built_ts: float
    chunker_key: str
    embedder_key: str


# =============================
# Protocol definitions
# =============================

class Chunker(Protocol):
    """Protocol for text chunking implementations"""
    def chunk(self, text: str) -> List[str]: ...


class Embedder(Protocol):
    """Protocol for text embedding implementations"""
    def embed(self, texts: List[str]) -> List[List[float]]: ...
    @property
    def spec(self) -> EmbedderSpec: ...


class AnnIndex(Protocol):
    """Protocol for approximate nearest neighbor index implementations"""
    def add(self, doc_id: str, chunk_ids: List[int], vectors: List[List[float]]) -> None: ...
    def query(self, q: List[float], top_k: int) -> List[tuple[str, int, float]]: ...  # (doc_id, chunk_id, score)
    def persist(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...
