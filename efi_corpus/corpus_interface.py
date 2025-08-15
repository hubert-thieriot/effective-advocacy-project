"""
Unified corpus handle that handles both reading and writing operations
"""

import json
import hashlib
import zstandard as zstd
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional

from .types import Document
from .layout import LocalFilesystemLayout


class CorpusHandle:
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
        
        # Core paths
        self.index_path = self.corpus_path / "index.jsonl"
        self.manifest_path = self.corpus_path / "manifest.json"
        self.docs_dir = self.corpus_path / "documents"
        
        # Validate paths
        if not self.corpus_path.exists():
            raise ValueError(f"Corpus path does not exist: {corpus_path}")
        
        if not self.index_path.exists():
            raise ValueError(f"Index file not found: {self.index_path}")
        
        # Create documents directory if writing is enabled
        if not self.read_only:
            self.docs_dir.mkdir(parents=True, exist_ok=True)
            self.index_path.touch(exist_ok=True)
            self.manifest_path.touch(exist_ok=True)

    # ============================================================================
    # READING OPERATIONS (always available)
    # ============================================================================
    
    def list_ids(self) -> List[str]:
        """Get list of all document IDs in the corpus"""
        doc_ids = []
        with open(self.index_path, 'r', encoding='utf-8') as f:
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

    def read_text(self, doc_id: str) -> str:
        """Read document text by ID"""
        text_path = self.docs_dir / doc_id / "text.txt"
        if not text_path.exists():
            raise FileNotFoundError(f"Text file not found for document {doc_id}: {text_path}")
        
        return text_path.read_text(encoding='utf-8')

    def read_metadata(self, doc_id: str) -> Dict[str, Any]:
        """Read document metadata by ID"""
        meta_path = self.docs_dir / doc_id / "meta.json"
        if not meta_path.exists():
            return {}
        
        try:
            return json.loads(meta_path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def read_fetch_info(self, doc_id: str) -> Dict[str, Any]:
        """Read document fetch information by ID"""
        fetch_path = self.docs_dir / doc_id / "fetch.json"
        if not fetch_path.exists():
            return {}
        
        try:
            return json.loads(fetch_path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def read_document(self, doc_id: str) -> Optional[Document]:
        """Read a complete document by ID"""
        try:
            # Get metadata from index
            doc_meta = None
            with open(self.index_path, 'r', encoding='utf-8') as f:
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
            
            # Read text and additional metadata
            text = self.read_text(doc_id)
            meta = self.read_metadata(doc_id)
            fetch_info = self.read_fetch_info(doc_id)
            
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

    def read_documents(self) -> Iterator[Document]:
        """Read all documents from the corpus"""
        for doc_id in self.list_ids():
            doc = self.read_document(doc_id)
            if doc:
                yield doc

    def get_document_count(self) -> int:
        """Get total number of documents in corpus"""
        return len(self.list_ids())

    def get_corpus_info(self) -> Dict[str, Any]:
        """Get basic information about the corpus"""
        manifest = {}
        if self.manifest_path.exists():
            try:
                manifest = json.loads(self.manifest_path.read_text(encoding='utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        
        return {
            "corpus_path": str(self.corpus_path),
            "document_count": self.get_document_count(),
            "read_only": self.read_only,
            "manifest": manifest
        }

    def fingerprint(self, doc_id: str) -> str:
        """Generate content fingerprint for a document"""
        text = self.read_text(doc_id)
        return hashlib.sha1(text.encode()).hexdigest()

    # ============================================================================
    # WRITING OPERATIONS (only available when read_only=False)
    # ============================================================================
    
    def has_doc(self, stable_id: str) -> bool:
        """Check if a document with the given stable_id exists"""
        if self.read_only:
            raise RuntimeError("Cannot check document existence in read-only mode")
        return (self.docs_dir / stable_id).exists()

    def write_document(self, *, stable_id: str, meta: Dict[str, Any], text: str,
                       raw_bytes: bytes, raw_ext: str, fetch_info: Dict[str, Any]):
        """Write a document to the corpus"""
        if self.read_only:
            raise RuntimeError("Cannot write documents in read-only mode")
        
        doc_dir = self.docs_dir / stable_id
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

    def append_index(self, row: Dict[str, Any]):
        """Append a row to the index.jsonl file"""
        if self.read_only:
            raise RuntimeError("Cannot modify index in read-only mode")
        
        with open(self.index_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def load_manifest(self) -> Dict[str, Any]:
        """Load the manifest.json file"""
        try:
            content = self.manifest_path.read_text(encoding='utf-8')
            return json.loads(content) if content.strip() else {}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save_manifest(self, manifest: Dict[str, Any]):
        """Save the manifest.json file"""
        if self.read_only:
            raise RuntimeError("Cannot modify manifest in read-only mode")
        
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), 
            encoding="utf-8"
        )
