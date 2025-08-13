"""
Corpus reader for loading documents from corpus format
"""

import json
from pathlib import Path
from typing import Iterator, Dict, Any, Optional
from .types import Document


class CorpusReader:
    """Read documents from a corpus directory"""
    
    def __init__(self, corpus_path: Path):
        """
        Initialize corpus reader
        
        Args:
            corpus_path: Path to corpus directory
        """
        self.corpus_path = Path(corpus_path)
        self.index_path = self.corpus_path / "index.jsonl"
        self.docs_dir = self.corpus_path / "documents"
        
        if not self.corpus_path.exists():
            raise ValueError(f"Corpus path does not exist: {corpus_path}")
        
        if not self.index_path.exists():
            raise ValueError(f"Index file not found: {self.index_path}")
    
    def read_documents(self) -> Iterator[Document]:
        """Read all documents from the corpus"""
        with open(self.index_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    index_entry = json.loads(line.strip())
                    doc = self._load_document(index_entry)
                    if doc:
                        yield doc
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON in index line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"Warning: Error loading document from line {line_num}: {e}")
                    continue
    
    def _load_document(self, index_entry: Dict[str, Any]) -> Optional[Document]:
        """Load a single document from index entry"""
        doc_id = index_entry.get('id')
        if not doc_id:
            return None
        
        # Load document content - handle two-character prefix directory structure
        # Document ID is split into prefix directory + full ID
        prefix = doc_id[:2]
        doc_path = self.docs_dir / prefix / doc_id / "text.txt"
        text = ""
        if doc_path.exists():
            try:
                with open(doc_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except Exception as e:
                print(f"Warning: Error loading content for document {doc_id}: {e}")
                text = ""
        else:
            print(f"Warning: Content file not found for document {doc_id} at {doc_path}")
            text = ""
        
        # Create Document object
        return Document(
            doc_id=doc_id,
            url=index_entry.get('url', ''),
            title=index_entry.get('title'),
            text=text,
            published_at=index_entry.get('published_at'),
            language=index_entry.get('language'),
            meta=index_entry.get('meta', {})
        )
    
    def get_document_count(self) -> int:
        """Get total number of documents in corpus"""
        count = 0
        with open(self.index_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    count += 1
        return count
    
    def get_corpus_info(self) -> Dict[str, Any]:
        """Get basic information about the corpus"""
        manifest_path = self.corpus_path / "manifest.json"
        manifest = {}
        
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
            except Exception as e:
                print(f"Warning: Error loading manifest: {e}")
        
        return {
            "corpus_path": str(self.corpus_path),
            "document_count": self.get_document_count(),
            "manifest": manifest
        }
