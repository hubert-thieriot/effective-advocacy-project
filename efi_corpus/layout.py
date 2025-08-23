"""
File layout management for EFI data structures

Provides four layout classes:
- CorpusLayout: Basic document storage layout
- LibraryLayout: Findings storage layout  
- EmbeddedCorpusLayout: Corpus with embedded vectors/chunks
- EmbeddedLibraryLayout: Library with embedded findings

Workspace structure:
workspace/
├── corpora/
│   └── {corpus_name}/
│       ├── chunks/
│       ├── embeddings/
│       └── indexes/
└── libraries/
    └── {library_name}/
        ├── chunks/
        ├── embeddings/
        └── indexes/
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from efi_core.types import ChunkerSpec, EmbedderSpec


class BaseLayout(ABC):
    """Base class for all layout implementations"""
    
    @abstractmethod
    def ensure_dirs(self) -> None:
        """Ensure all necessary directories exist"""
        pass


@dataclass
class CorpusLayout(BaseLayout):
    """
    Basic document storage layout for corpora.
    
    Structure:
    - corpus_root/
      - index.jsonl          # Document metadata index
      - manifest.json        # Corpus metadata
      - documents/           # Document storage
        - doc_id/
          - text.txt         # Document text
          - meta.json        # Document metadata
          - raw.*            # Raw document file
          - fetch.json       # Fetch information
    """
    corpus_root: Path
    
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

    def ensure_dirs(self) -> None:
        """Ensure all necessary directories exist"""
        self.docs_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class LibraryLayout(BaseLayout):
    """
    Findings storage layout for libraries.
    
    Structure:
    - library_root/
      - findings.json        # All findings data
      - metadata.json        # Library metadata
      - sources/             # Source documents (optional)
        - source_id/
          - text.txt         # Source text
          - meta.json        # Source metadata
    """
    library_root: Path
    
    @property
    def findings_path(self) -> Path:
        """Get path to findings.json file"""
        return self.library_root / "findings.json"
    
    @property
    def metadata_path(self) -> Path:
        """Get path to metadata.json file"""
        return self.library_root / "metadata.json"
    
    @property
    def sources_dir(self) -> Path:
        """Get path to sources directory (optional)"""
        return self.library_root / "sources"

    def source_dir(self, source_id: str) -> Path:
        """Get source directory path"""
        return self.sources_dir / source_id

    def source_text_path(self, source_id: str) -> Path:
        """Get path to source text file"""
        return self.source_dir(source_id) / "text.txt"

    def source_meta_path(self, source_id: str) -> Path:
        """Get path to source metadata file"""
        return self.source_dir(source_id) / "meta.json"

    def ensure_dirs(self) -> None:
        """Ensure all necessary directories exist"""
        self.sources_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class WorkspaceLayout(BaseLayout):
    """
    Base class for workspace-based layouts.
    
    Workspace structure:
    workspace_root/
    ├── corpora/
    │   └── {corpus_name}/
    │       ├── chunks/
    │       ├── embeddings/
    │       └── indexes/
    └── libraries/
        └── {library_name}/
            ├── chunks/
            ├── embeddings/
            └── indexes/
    """
    workspace_root: Path
    
    def ensure_dirs(self) -> None:
        """Ensure all necessary workspace directories exist"""
        # Create base workspace directories
        dirs = [
            self.workspace_root / "corpora",
            self.workspace_root / "libraries"
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def get_corpus_workspace_dir(self, corpus_name: str) -> Path:
        """Get workspace directory for a specific corpus"""
        return self.workspace_root / "corpora" / corpus_name
    
    def get_library_workspace_dir(self, library_name: str) -> Path:
        """Get workspace directory for a specific library"""
        return self.workspace_root / "libraries" / library_name


class EmbeddedCorpusLayout(CorpusLayout, WorkspaceLayout):
    """
    Corpus layout with embedded vectors and chunks.
    
    Extends CorpusLayout with workspace for:
    - chunks: Document chunks
    - embeddings: Vector embeddings
    - indexes: Search indexes
    """
    
    def __init__(self, corpus_path: Path, workspace_root: Path):
        CorpusLayout.__init__(self, corpus_path)
        WorkspaceLayout.__init__(self, workspace_root)
    
    def chunks_path(self, doc_id: str, chunker: ChunkerSpec) -> Path:
        """Get path to cached chunks for a document"""
        corpus_name = self.corpus_root.name
        workspace_dir = self.get_corpus_workspace_dir(corpus_name)
        return workspace_dir / "chunks" / chunker.key() / f"{doc_id}.chunks.jsonl"

    def emb_path(self, doc_id: str, chunker: ChunkerSpec, embedder: EmbedderSpec) -> Path:
        """Get path to cached embeddings for a document"""
        corpus_name = self.corpus_root.name
        workspace_dir = self.get_corpus_workspace_dir(corpus_name)
        return workspace_dir / "embeddings" / chunker.key() / embedder.key() / f"{doc_id}.npy"

    def index_dir(self, chunker: ChunkerSpec, embedder: EmbedderSpec) -> Path:
        """Get directory for search index files"""
        corpus_name = self.corpus_root.name
        workspace_dir = self.get_corpus_workspace_dir(corpus_name)
        return workspace_dir / "indexes" / chunker.key() / embedder.key()

    def ensure_dirs(self) -> None:
        """Ensure all necessary directories exist"""
        CorpusLayout.ensure_dirs(self)  # Call CorpusLayout.ensure_dirs()
        WorkspaceLayout.ensure_dirs(self)  # Call WorkspaceLayout.ensure_dirs()
        
        # Ensure corpus-specific workspace directories
        corpus_name = self.corpus_root.name
        corpus_workspace = self.get_corpus_workspace_dir(corpus_name)
        
        dirs = [
            corpus_workspace / "chunks",
            corpus_workspace / "embeddings", 
            corpus_workspace / "indexes"
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


class EmbeddedLibraryLayout(LibraryLayout, WorkspaceLayout):
    """
    Library layout with embedded findings.
    
    Extends LibraryLayout with workspace for:
    - chunks: Finding chunks
    - embeddings: Finding vector embeddings
    - indexes: Search indexes
    """
    
    def __init__(self, library_root: Path, workspace_root: Path):
        LibraryLayout.__init__(self, library_root)
        WorkspaceLayout.__init__(self, workspace_root)
    
    def chunks_path(self, finding_id: str, chunker: ChunkerSpec) -> Path:
        """Get path to cached chunks for a finding"""
        library_name = self.library_root.name
        workspace_dir = self.get_library_workspace_dir(library_name)
        return workspace_dir / "chunks" / chunker.key() / f"{finding_id}.chunks.jsonl"

    def emb_path(self, finding_id: str, chunker: ChunkerSpec, embedder: EmbedderSpec) -> Path:
        """Get path to cached embeddings for a finding"""
        library_name = self.library_root.name
        workspace_dir = self.get_library_workspace_dir(library_name)
        return workspace_dir / "embeddings" / chunker.key() / embedder.key() / f"{finding_id}.npy"

    def index_dir(self, chunker: ChunkerSpec, embedder: EmbedderSpec) -> Path:
        """Get directory for search index files"""
        library_name = self.library_root.name
        workspace_dir = self.get_library_workspace_dir(library_name)
        return workspace_dir / "indexes" / chunker.key() / embedder.key()

    def ensure_dirs(self) -> None:
        """Ensure all necessary directories exist"""
        LibraryLayout.ensure_dirs(self)  # Call LibraryLayout.ensure_dirs()
        WorkspaceLayout.ensure_dirs(self)  # Call WorkspaceLayout.ensure_dirs()
        
        # Ensure library-specific workspace directories
        library_name = self.library_root.name
        library_workspace = self.get_library_workspace_dir(library_name)
        
        dirs = [
            library_workspace / "chunks",
            library_workspace / "embeddings", 
            library_workspace / "indexes"
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


# Legacy alias for backward compatibility
LocalFilesystemLayout = CorpusLayout
