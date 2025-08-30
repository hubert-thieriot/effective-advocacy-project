"""
Unified corpus handle that handles both reading and writing operations
"""

import json
import hashlib
import zstandard as zstd
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional
from functools import wraps

from .types import Document
from efi_core.layout import CorpusLayout
from efi_core.protocols import Corpus


def write_only(func):
    """Decorator to ensure method is only called when not in read-only mode"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.read_only:
            raise RuntimeError(f"Cannot call {func.__name__} in read-only mode")
        return func(self, *args, **kwargs)
    return wrapper


class CorpusHandle(Corpus):
    """
    Unified corpus handle that handles both reading and writing operations.
    
    When read_only=True (default), only reading operations are available.
    When read_only=False, both reading and writing operations are available.
    """
    
    def __init__(self, corpus_path: Path, read_only: bool = True):
        """
        Initialize corpus handle
        
        Args:
            corpus_path: Path to corpus directory
            read_only: If True, only reading operations are available
        """
        self.corpus_path = Path(corpus_path)
        self.read_only = read_only
        
        # Initialize layout (no workspace needed for basic corpus operations)
        self.layout = CorpusLayout(self.corpus_path)
        
        # Create corpus directory if it doesn't exist
        if not self.corpus_path.exists():
            if read_only:
                raise ValueError(f"Corpus path does not exist: {corpus_path}")
            else:
                self.corpus_path.mkdir(parents=True, exist_ok=True)
        
        # Create initial corpus structure if it doesn't exist and we're not read-only
        if not read_only:
            self.layout.docs_dir.mkdir(parents=True, exist_ok=True)
            
            # Create empty index.jsonl if it doesn't exist
            if not self.layout.index_path.exists():
                self.layout.index_path.touch()
                print(f"Created new index file: {self.layout.index_path}")
            
            # Create empty manifest.json if it doesn't exist
            if not self.layout.manifest_path.exists():
                self.layout.manifest_path.touch()
                print(f"Created new manifest file: {self.layout.manifest_path}")
        else:
            # In read-only mode, validate that required files exist
            if not self.layout.index_path.exists():
                raise ValueError(f"Index file not found: {self.layout.index_path}")
        
        # Ensure workspace directories exist
        self.layout.ensure_dirs()

    # ============================================================================
    # READING OPERATIONS (always available)
    # ============================================================================
    
    def list_ids(self) -> List[str]:
        """Get list of all document IDs in the corpus"""
        doc_ids = []
        with open(self.layout.index_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        index_entry = json.loads(line)
                        doc_id = index_entry.get('id')
                        if doc_id:
                            doc_ids.append(doc_id)
                    except json.JSONDecodeError:
                        continue
        return doc_ids

    def get_text(self, doc_id: str) -> str:
        """Get document text by ID"""
        text_path = self.layout.text_path(doc_id)
        if not text_path.exists():
            raise FileNotFoundError(f"Text file not found for document {doc_id}: {text_path}")
        
        return text_path.read_text(encoding='utf-8')

    def get_metadata(self, doc_id: str) -> Dict[str, Any]:
        """Get document metadata by ID"""
        meta_path = self.layout.meta_path(doc_id)
        if not meta_path.exists():
            return {}
        
        try:
            return json.loads(meta_path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def get_fetch_info(self, doc_id: str) -> Dict[str, Any]:
        """Get document fetch information by ID"""
        fetch_path = self.layout.fetch_path(doc_id)
        if not fetch_path.exists():
            return {}
        
        try:
            return json.loads(fetch_path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a complete document by ID"""
        try:
            # Get metadata from index
            doc_meta = None
            with open(self.layout.index_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            index_entry = json.loads(line)
                            if index_entry.get('id') == doc_id:
                                doc_meta = index_entry
                                break
                        except json.JSONDecodeError:
                            continue
            
            if not doc_meta:
                return None
            
            # Get text and additional metadata
            text = self.get_text(doc_id)
            meta = self.get_metadata(doc_id)
            fetch_info = self.get_fetch_info(doc_id)
            
            # Merge metadata
            full_meta = {**meta, **fetch_info}
            
            return Document(
                doc_id=doc_id,
                url=doc_meta.get('url', ''),
                title=doc_meta.get('title'),
                text=text,
                published_at=doc_meta.get('published_at'),
                language=doc_meta.get('language'),
                meta=full_meta
            )
            
        except Exception as e:
            print(f"Error reading document {doc_id}: {e}")
            return None


    
    def iter_documents(self) -> Iterator[Document]:
        """
        Iterate over all documents in the corpus.
        
        Returns:
            Iterator over Document objects
        """
        for doc_id in self.list_ids():
            doc = self.get_document(doc_id)
            if doc:
                yield doc

    def get_document_count(self) -> int:
        """Get total number of documents in corpus"""
        return len(self.list_ids())

    def get_corpus_info(self) -> Dict[str, Any]:
        """Get basic information about the corpus"""
        manifest = {}
        if self.layout.manifest_path.exists():
            try:
                manifest = json.loads(self.layout.manifest_path.read_text(encoding='utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        
        return {
            "corpus_path": str(self.corpus_path),
            "workspace_path": str(self.layout.workspace_root) if self.layout.workspace_root else None,
            "document_count": self.get_document_count(),
            "read_only": self.read_only,
            "manifest": manifest
        }

    def fingerprint(self, doc_id: str) -> str:
        """Generate content fingerprint for a document"""
        text = self.get_text(doc_id)
        return hashlib.sha1(text.encode()).hexdigest()

    def load_manifest(self) -> Dict[str, Any]:
        """Load the manifest.json file"""
        try:
            content = self.layout.manifest_path.read_text(encoding='utf-8')
            return json.loads(content) if content.strip() else {}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    # ============================================================================
    # WRITING OPERATIONS (only available when read_only=False)
    # ============================================================================
    
    @write_only
    def has_doc(self, stable_id: str) -> bool:
        """Check if a document with the given stable_id exists"""
        return self.layout.doc_dir(stable_id).exists()

    @write_only
    def write_document(self, *, stable_id: str, meta: Dict[str, Any], text: str,
                       raw_bytes: bytes, raw_ext: str, fetch_info: Dict[str, Any]):
        """Write a document to the corpus"""
        doc_dir = self.layout.doc_dir(stable_id)
        doc_dir.mkdir(parents=True, exist_ok=True)
        
        # meta.json
        (doc_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), 
            encoding="utf-8"
        )

        # text.txt (plain text, no compression)
        (doc_dir / "text.txt").write_text(text, encoding="utf-8")

        # raw file (copy from cache)
        (doc_dir / f"raw.{raw_ext}").write_bytes(raw_bytes)

        # fetch.json
        fetch_payload = {**fetch_info}
        (doc_dir / "fetch.json").write_text(
            json.dumps(fetch_payload, indent=2), 
            encoding="utf-8"
        )

    @write_only
    def append_index(self, row: Dict[str, Any]):
        """Append a row to the index.jsonl file"""
        with open(self.layout.index_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    @write_only
    def save_manifest(self, manifest: Dict[str, Any]):
        """Save the manifest.json file"""
        self.layout.manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), 
            encoding="utf-8"
        )
