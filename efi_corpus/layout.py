"""
File layout management for corpus processing pipeline
"""

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional


@dataclass(frozen=True)
class ChunkerSpec:
    name: str
    params: Dict[str, Any]

    def key(self) -> str:
        s = json.dumps({"name": self.name, "params": self.params}, sort_keys=True)
        return hashlib.sha1(s.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class EmbedderSpec:
    model_name: str
    dim: int
    revision: Optional[str] = None  # e.g., HF commit hash

    def key(self) -> str:
        s = json.dumps({"model": self.model_name, "dim": self.dim, "rev": self.revision}, sort_keys=True)
        return hashlib.sha1(s.encode()).hexdigest()[:12]


@dataclass
class LocalFilesystemLayout:
    """
    Manages file layout for corpus processing pipeline.
    
    Uses a simple flat structure:
    - corpus_root: contains documents with simple docid organization
    - workspace_root: contains cached chunks, embeddings, and indexes (can be None)
    """
    corpus_root: Path
    workspace_root: Optional[Path]

    @property
    def index_path(self) -> Path:
        """Get path to index.jsonl file"""
        return self.corpus_root / "index.jsonl"
    
    @property
    def manifest_path(self) -> Path:
        """Get path to manifest.json file"""
        return self.corpus_root / "manifest.json"
    
    @property
    def docs_dir(self) -> Path:
        """Get path to documents directory"""
        return self.corpus_root / "documents"

    def doc_dir(self, doc_id: str) -> Path:
        """Get document directory path"""
        return self.docs_dir / doc_id

    def text_path(self, doc_id: str) -> Path:
        """Get path to document text file"""
        return self.doc_dir(doc_id) / "text.txt"

    def meta_path(self, doc_id: str) -> Path:
        """Get path to document metadata file"""
        return self.doc_dir(doc_id) / "meta.json"

    def raw_path(self, doc_id: str) -> Optional[Path]:
        """Get path to raw document file (e.g., raw.html, raw.pdf)"""
        raw_dir = self.doc_dir(doc_id)
        raw_files = list(raw_dir.glob("raw.*"))
        return raw_files[0] if raw_files else None

    def fetch_path(self, doc_id: str) -> Path:
        """Get path to document fetch info file"""
        return self.doc_dir(doc_id) / "fetch.json"

    def chunks_path(self, doc_id: str, chunker: ChunkerSpec) -> Path:
        """Get path to cached chunks for a document"""
        if self.workspace_root is None:
            raise ValueError("Workspace not configured for chunking operations")
        return self.workspace_root / "chunks" / chunker.key() / f"{doc_id}.chunks.jsonl"

    def emb_path(self, doc_id: str, chunker: ChunkerSpec, embedder: EmbedderSpec) -> Path:
        """Get path to cached embeddings for a document"""
        if self.workspace_root is None:
            raise ValueError("Workspace not configured for embedding operations")
        return self.workspace_root / "embeddings" / chunker.key() / embedder.key() / f"{doc_id}.npy"

    def index_dir(self, chunker: ChunkerSpec, embedder: EmbedderSpec) -> Path:
        """Get directory for ANN index files"""
        if self.workspace_root is None:
            raise ValueError("Workspace not configured for indexing operations")
        return self.workspace_root / "indexes" / chunker.key() / embedder.key()

    def doc_state_path(self, doc_id: str) -> Path:
        """Get path to document state tracking file"""
        if self.workspace_root is None:
            raise ValueError("Workspace not configured for state tracking operations")
        return self.workspace_root / "doc_meta" / f"{doc_id}.meta.json"

    def ensure_workspace_dirs(self) -> None:
        """Create all necessary workspace directories"""
        if self.workspace_root is None:
            return  # No workspace configured
        
        dirs = [
            self.workspace_root / "chunks",
            self.workspace_root / "embeddings", 
            self.workspace_root / "indexes",
            self.workspace_root / "doc_meta"
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
