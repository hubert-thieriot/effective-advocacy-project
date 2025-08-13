"""
Tests for efi_corpus corpus reader
"""

import pytest
import json
import tempfile
from pathlib import Path
from efi_corpus.corpus_reader import CorpusReader


class TestCorpusReader:
    """Test CorpusReader class"""
    
    @pytest.fixture
    def temp_corpus(self):
        """Create a temporary corpus structure"""
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus_path = Path(temp_dir) / "test_corpus"
            corpus_path.mkdir()
            
            # Create documents directory (current structure)
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
            
            # Create document content and metadata with current structure
            for doc_id in ["doc1", "doc2"]:
                # Create two-character prefix directory
                prefix = doc_id[:2]
                prefix_dir = docs_dir / prefix
                prefix_dir.mkdir(exist_ok=True)
                
                doc_dir = prefix_dir / doc_id
                doc_dir.mkdir()
                
                # Content file (text.txt in current structure)
                with open(doc_dir / "text.txt", 'w') as f:
                    f.write(f"Content for {doc_id}")
                
                # Metadata file
                with open(doc_dir / "meta.json", 'w') as f:
                    json.dump({
                        "source": "test",
                        "category": "news"
                    }, f)
            
            yield corpus_path
    
    def test_corpus_reader_init(self, temp_corpus):
        """Test CorpusReader initialization"""
        reader = CorpusReader(temp_corpus)
        assert reader.corpus_path == temp_corpus
        assert reader.index_path == temp_corpus / "index.jsonl"
        assert reader.docs_dir == temp_corpus / "documents"
    
    def test_corpus_reader_init_nonexistent_path(self):
        """Test CorpusReader with non-existent path"""
        with pytest.raises(ValueError, match="Corpus path does not exist"):
            CorpusReader(Path("/nonexistent/path"))
    
    def test_corpus_reader_init_missing_index(self, temp_corpus):
        """Test CorpusReader with missing index file"""
        (temp_corpus / "index.jsonl").unlink()
        
        with pytest.raises(ValueError, match="Index file not found"):
            CorpusReader(temp_corpus)
    
    def test_corpus_reader_get_document_count(self, temp_corpus):
        """Test getting document count"""
        reader = CorpusReader(temp_corpus)
        assert reader.get_document_count() == 2
    
    def test_corpus_reader_read_documents(self, temp_corpus):
        """Test reading documents from corpus"""
        reader = CorpusReader(temp_corpus)
        documents = list(reader.read_documents())
        
        assert len(documents) == 2
        
        # Check first document
        doc1 = documents[0]
        assert doc1.doc_id == "doc1"
        assert doc1.url == "https://example.com/doc1"
        assert doc1.title == "Document 1"
        assert doc1.text == "Content for doc1"
        assert doc1.published_at == "2025-01-01"
        assert doc1.language == "en"
        # Note: meta.json files are not loaded in current implementation
        assert doc1.meta == {}
        
        # Check second document
        doc2 = documents[1]
        assert doc2.doc_id == "doc2"
        assert doc2.url == "https://example.com/doc2"
        assert doc2.title == "Document 2"
        assert doc2.text == "Content for doc2"
    
    def test_corpus_reader_missing_content_file(self, temp_corpus):
        """Test handling missing content file"""
        # Remove content file for doc1
        (temp_corpus / "documents" / "do" / "doc1" / "text.txt").unlink()
        
        reader = CorpusReader(temp_corpus)
        documents = list(reader.read_documents())
        
        # Should return both documents, but doc1 will have empty text
        assert len(documents) == 2
        
        # doc1 should have empty text
        doc1 = next(d for d in documents if d.doc_id == "doc1")
        assert doc1.text == ""
        
        # doc2 should have normal text
        doc2 = next(d for d in documents if d.doc_id == "doc2")
        assert doc2.text == "Content for doc2"
    
    def test_corpus_reader_missing_metadata_file(self, temp_corpus):
        """Test handling missing metadata file"""
        # Remove metadata file for doc1 (using correct path with prefix)
        (temp_corpus / "documents" / "do" / "doc1" / "meta.json").unlink()
        
        reader = CorpusReader(temp_corpus)
        documents = list(reader.read_documents())
        
        # Should still return doc1 but with empty metadata (since meta.json is not loaded anyway)
        assert len(documents) == 2
        assert documents[0].meta == {}
    
    def test_corpus_reader_invalid_json_in_index(self, temp_corpus):
        """Test handling invalid JSON in index file"""
        # Add invalid JSON line
        with open(temp_corpus / "index.jsonl", 'a') as f:
            f.write("invalid json line\n")
        
        reader = CorpusReader(temp_corpus)
        documents = list(reader.read_documents())
        
        # Should still return valid documents
        assert len(documents) == 2
    
    def test_corpus_reader_get_corpus_info(self, temp_corpus):
        """Test getting corpus information"""
        reader = CorpusReader(temp_corpus)
        info = reader.get_corpus_info()
        
        assert info["corpus_path"] == str(temp_corpus)
        assert info["document_count"] == 2
        assert "manifest" in info
