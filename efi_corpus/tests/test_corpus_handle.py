"""
Tests for efi_corpus CorpusHandle
"""

import pytest
import json
import tempfile
from pathlib import Path
from efi_corpus.corpus_handle import CorpusHandle


class TestCorpusHandle:
    """Test CorpusHandle class"""
    
    @pytest.fixture
    def temp_corpus(self):
        """Create a temporary corpus structure"""
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_path = Path(temp_dir) / "test_corpus"
            corpus_path.mkdir()
            
            # Create documents directory (new flat structure)
            docs_dir = corpus_path / "documents"
            docs_dir.mkdir()
            
            # Create index.jsonl
            index_path = corpus_path / "index.jsonl"
            with open(index_path, 'w') as f:
                f.write(json.dumps({
                    "id": "doc1",
                    "url": "https://example.com/doc1",
                    "title": "Document 1",
                    "published_at": "2025-01-01",
                    "language": "en"
                }) + "\n")
                f.write(json.dumps({
                    "id": "doc2",
                    "url": "https://example.com/doc2",
                    "title": "Document 2",
                    "published_at": "2025-01-02",
                    "language": "en"
                }) + "\n")
            
            # Create document content and metadata with new flat structure
            for doc_id in ["doc1", "doc2"]:
                doc_dir = docs_dir / doc_id
                doc_dir.mkdir()
                
                # Content file
                with open(doc_dir / "text.txt", 'w') as f:
                    f.write(f"Content for {doc_id}")
                
                # Metadata file
                with open(doc_dir / "meta.json", 'w') as f:
                    json.dump({
                        "source": "test",
                        "category": "news"
                    }, f)
                
                # Fetch info file
                with open(doc_dir / "fetch.json", 'w') as f:
                    json.dump({
                        "fetch_time": "2025-01-01T00:00:00Z",
                        "status_code": 200
                    }, f)
            
            yield corpus_path
    
    def test_corpus_handle_init(self, temp_corpus):
        """Test CorpusHandle initialization"""
        corpus = CorpusHandle(temp_corpus)
        assert corpus.corpus_path == temp_corpus
        assert corpus.layout.index_path == temp_corpus / "index.jsonl"
        assert corpus.layout.docs_dir == temp_corpus / "documents"
        assert corpus.read_only is True  # Default is read-only
    
    def test_corpus_handle_init_nonexistent_path(self):
        """Test CorpusHandle with non-existent path"""
        with pytest.raises(ValueError, match="Corpus path does not exist"):
            CorpusHandle(Path("/nonexistent/path"))
    
    def test_corpus_handle_init_missing_index(self, temp_corpus):
        """Test CorpusHandle with missing index file"""
        (temp_corpus / "index.jsonl").unlink()
        
        with pytest.raises(ValueError, match="Index file not found"):
            CorpusHandle(temp_corpus)
    
    def test_corpus_handle_get_document_count(self, temp_corpus):
        """Test getting document count"""
        corpus = CorpusHandle(temp_corpus)
        assert corpus.get_document_count() == 2
    
    def test_corpus_handle_read_documents(self, temp_corpus):
        """Test reading documents from corpus"""
        corpus = CorpusHandle(temp_corpus)
        documents = list(corpus.read_documents())
        
        assert len(documents) == 2
        
        # Check first document
        doc1 = documents[0]
        assert doc1.doc_id == "doc1"
        assert doc1.url == "https://example.com/doc1"
        assert doc1.title == "Document 1"
        assert doc1.text == "Content for doc1"
    
    def test_corpus_handle_read_only_restrictions(self, temp_corpus):
        """Test that read-only mode prevents writing operations"""
        corpus = CorpusHandle(temp_corpus, read_only=True)
        
        # These should raise RuntimeError in read-only mode
        with pytest.raises(RuntimeError, match="Cannot call write_document in read-only mode"):
            corpus.write_document(
                stable_id="test123",
                meta={},
                text="test",
                raw_bytes=b"test",
                raw_ext="txt",
                fetch_info={}
            )
        
        with pytest.raises(RuntimeError, match="Cannot call append_index in read-only mode"):
            corpus.append_index({"id": "test123"})
    
    def test_corpus_handle_writable_mode(self, temp_corpus):
        """Test that writable mode allows writing operations"""
        corpus = CorpusHandle(temp_corpus, read_only=False)
        
        # These should work in writable mode
        corpus.write_document(
            stable_id="test123",
            meta={"source": "test"},
            text="test content",
            raw_bytes=b"test",
            raw_ext="txt",
            fetch_info={"status": 200}
        )
        
        corpus.append_index({"id": "test123", "url": "https://example.com/test"})
        
        # Verify the document was written
        assert corpus.has_doc("test123")
        assert corpus.read_text("test123") == "test content"
