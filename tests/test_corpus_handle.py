"""
Tests for CorpusHandle
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch

from efi_corpus.corpus_handle import CorpusHandle


class TestCorpusHandle:
    """Test cases for CorpusHandle"""
    
    @pytest.fixture
    def corpus_dir(self, tmp_path):
        """Create a temporary corpus directory"""
        corpus_dir = tmp_path / "test_corpus"
        return corpus_dir
    
    @pytest.fixture
    def corpus_handle(self, corpus_dir):
        """Create a CorpusHandle instance"""
        return CorpusHandle(corpus_dir)
    
    def test_init_creates_directories(self, corpus_dir):
        """Test that initialization creates necessary directories"""
        handle = CorpusHandle(corpus_dir)
        
        assert (corpus_dir / "documents").exists()
        assert (corpus_dir / "index.jsonl").exists()
        assert (corpus_dir / "manifest.json").exists()
    
    def test_has_doc_false_when_not_exists(self, corpus_handle):
        """Test has_doc returns False for non-existent documents"""
        assert not corpus_handle.has_doc("nonexistent_id")
    
    def test_has_doc_true_when_exists(self, corpus_handle, corpus_dir):
        """Test has_doc returns True for existing documents"""
        # Create a mock document directory
        doc_dir = corpus_dir / "documents" / "ab" / "ab1234567890"
        doc_dir.mkdir(parents=True)
        
        assert corpus_handle.has_doc("ab1234567890")
    
    def test_write_document_creates_structure(self, corpus_handle, corpus_dir):
        """Test that write_document creates the proper directory structure"""
        stable_id = "ab1234567890"
        meta = {"title": "Test Document"}
        text = "This is test content"
        raw_bytes = b"Test document content"
        raw_ext = "txt"
        fetch_info = {"status": 200}
        
        corpus_handle.write_document(
            stable_id=stable_id,
            meta=meta,
            text=text,
            raw_bytes=raw_bytes,
            raw_ext=raw_ext,
            fetch_info=fetch_info
        )
        
        # Check that directory was created
        doc_dir = corpus_dir / "documents" / "ab" / stable_id
        assert doc_dir.exists()
        
        # Check that files were created
        assert (doc_dir / "meta.json").exists()
        assert (doc_dir / "text.txt").exists()
        assert (doc_dir / "raw.txt").exists()
        assert (doc_dir / "fetch.json").exists()
    
    def test_write_document_meta_content(self, corpus_handle, corpus_dir):
        """Test that meta.json contains correct content"""
        stable_id = "ab1234567890"
        meta = {"title": "Test Document", "author": "Test Author"}
        text = "This is test content"
        raw_bytes = b"<html>Test HTML</html>"
        raw_ext = "html.zst"
        fetch_info = {"status": 200}
        
        corpus_handle.write_document(
            stable_id=stable_id,
            meta=meta,
            text=text,
            raw_bytes=raw_bytes,
            raw_ext=raw_ext,
            fetch_info=fetch_info
        )
        
        # Check meta.json content
        doc_dir = corpus_dir / "documents" / "ab" / stable_id
        meta_file = doc_dir / "meta.json"
        
        with open(meta_file, 'r', encoding='utf-8') as f:
            saved_meta = json.load(f)
        
        assert saved_meta["title"] == "Test Document"
        assert saved_meta["author"] == "Test Author"
    
    def test_append_index(self, corpus_handle, corpus_dir):
        """Test that append_index adds rows to index.jsonl"""
        row1 = {"id": "doc1", "title": "Document 1"}
        row2 = {"id": "doc2", "title": "Document 2"}
        
        corpus_handle.append_index(row1)
        corpus_handle.append_index(row2)
        
        # Check index.jsonl content
        index_file = corpus_dir / "index.jsonl"
        with open(index_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        assert len(lines) == 2
        assert json.loads(lines[0]) == row1
        assert json.loads(lines[1]) == row2
    
    def test_load_manifest_empty_when_not_exists(self, corpus_handle):
        """Test that load_manifest returns empty dict when file doesn't exist"""
        manifest = corpus_handle.load_manifest()
        assert manifest == {}
    
    def test_save_and_load_manifest(self, corpus_handle, corpus_dir):
        """Test that manifest can be saved and loaded"""
        test_manifest = {
            "name": "test_corpus",
            "params": {"keywords": ["test"]},
            "history": []
        }
        
        corpus_handle.save_manifest(test_manifest)
        
        # Load and verify
        loaded_manifest = corpus_handle.load_manifest()
        assert loaded_manifest == test_manifest
    
    def test_load_manifest_invalid_json(self, corpus_handle, corpus_dir):
        """Test that load_manifest handles invalid JSON gracefully"""
        manifest_file = corpus_dir / "manifest.json"
        manifest_file.write_text("invalid json content")
        
        manifest = corpus_handle.load_manifest()
        assert manifest == {}
    
    def test_write_document_plain_text_content(self, corpus_handle, corpus_dir):
        """Test that text content is written as plain text"""
        stable_id = "ab1234567890"
        meta = {"title": "Test"}
        text = "This is test content that should be written as plain text"
        raw_bytes = b"Test document content"
        raw_ext = "txt"
        fetch_info = {"status": 200}
        
        corpus_handle.write_document(
            stable_id=stable_id,
            meta=meta,
            text=text,
            raw_bytes=raw_bytes,
            raw_ext=raw_ext,
            fetch_info=fetch_info
        )
        
        doc_dir = corpus_dir / "documents" / "ab" / stable_id
        
        # Check that text file contains the exact content
        text_file = doc_dir / "text.txt"
        raw_file = doc_dir / "raw.txt"
        
        # Text should be written as-is
        with open(text_file, 'r', encoding='utf-8') as f:
            saved_text = f.read()
        assert saved_text == text
        
        # Raw file should exist
        assert raw_file.exists()
