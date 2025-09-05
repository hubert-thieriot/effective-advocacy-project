import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch

from efi_core.types import Finding, LibraryDocumentWFindings
from efi_library.library_store import LibraryStore


class TestLibraryStore:
    """Test LibraryStore functionality"""
    
    def test_init_creates_directories(self, temp_library_dir):
        """Test that store creates necessary directories"""
        store = LibraryStore("test_lib", str(temp_library_dir))
        
        assert (temp_library_dir / "test_lib").exists()
        assert (temp_library_dir / "test_lib" / "documents").exists()
        assert (temp_library_dir / "test_lib" / "index.json").exists()
        assert (temp_library_dir / "test_lib" / "metadata.json").exists()
    
    def test_store_findings_creates_document_structure(self, temp_library_dir, sample_document):
        """Test storing findings creates proper document structure"""
        store = LibraryStore("test_lib", str(temp_library_dir))
        
        success = store.store_findings(sample_document)
        assert success
        
        # Check document directory was created
        from efi_core.types import Finding
        doc_id = Finding.generate_doc_id("https://example.com")
        doc_dir = temp_library_dir / "test_lib" / "documents" / doc_id
        assert doc_dir.exists()
        assert (doc_dir / "findings.json").exists()
        assert (doc_dir / "metadata.json").exists()
    
    def test_get_findings_by_url(self, temp_library_dir, sample_document):
        """Test retrieving findings by URL"""
        store = LibraryStore("test_lib", str(temp_library_dir))
        store.store_findings(sample_document)
        
        result = store.get_findings_by_url("https://example.com")
        assert result is not None
        assert result.url == "https://example.com"
        assert len(result.findings) == 2
    
    def test_get_findings_by_doc_id(self, temp_library_dir, sample_document):
        """Test retrieving findings by document ID"""
        store = LibraryStore("test_lib", str(temp_library_dir))
        store.store_findings(sample_document)
        
        from efi_core.types import Finding
        doc_id = Finding.generate_doc_id("https://example.com")
        
        result = store.get_findings_by_doc_id(doc_id)
        assert result is not None
        assert result.doc_id == doc_id
        assert len(result.findings) == 2
    
    def test_list_all_findings(self, temp_library_dir, sample_document):
        """Test listing all findings"""
        store = LibraryStore("test_lib", str(temp_library_dir))
        store.store_findings(sample_document)

        results = store.list_documents()
        assert len(results) == 1
        from efi_core.types import Finding
        expected_doc_id = Finding.generate_doc_id("https://example.com")
        assert results[0].doc_id == expected_doc_id
    
    def test_search_findings(self, temp_library_dir, sample_document):
        """Test searching findings by text"""
        store = LibraryStore("test_lib", str(temp_library_dir))
        store.store_findings(sample_document)
        
        results = store.search_findings("Sample finding 1")
        assert len(results) == 1
        from efi_core.types import Finding
        expected_doc_id = Finding.generate_doc_id("https://example.com")
        assert results[0].doc_id == expected_doc_id
    
    def test_get_storage_stats(self, temp_library_dir):
        """Test getting storage statistics"""
        store = LibraryStore("test_lib", str(temp_library_dir))
        
        stats = store.get_storage_stats()
        assert "name" in stats
        assert "type" in stats
        assert stats["name"] == "test_lib"
