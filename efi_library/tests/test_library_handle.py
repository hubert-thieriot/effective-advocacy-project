import pytest
import json
import time
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from efi_core.types import Finding, LibraryDocumentWFindings
from efi_library.library_handle import LibraryHandle


class TestLibraryHandle:
    """Test LibraryHandle functionality"""
    
    def test_init_creates_store(self, temp_library_dir):
        """Test that handle creates a store instance"""
        with patch('efi_library.library_handle.LibraryLayout'):
            handle = LibraryHandle(temp_library_dir)
            assert handle.store is not None
    
    def test_iter_findings_uses_store(self, temp_library_dir, sample_document):
        """Test that iter_findings uses the store"""
        with patch('efi_library.library_handle.LibraryLayout'):
            with patch('efi_library.library_handle.LibraryStore') as mock_store_class:
                mock_store = Mock()
                mock_store.list_all_findings.return_value = [sample_document]
                mock_store_class.return_value = mock_store
                
                handle = LibraryHandle(temp_library_dir)
                
                results = list(handle.iter_findings())
                assert len(results) == 1
                expected_doc_id = Finding.generate_doc_id("https://example.com")
                assert results[0].doc_id == expected_doc_id
                mock_store.list_all_findings.assert_called_once()
    
    def test_get_finding_by_id(self, temp_library_dir, sample_document):
        """Test getting a finding by ID"""
        with patch('efi_library.library_handle.LibraryLayout') as mock_layout:
            with patch('efi_library.library_handle.LibraryStore') as mock_store_class:
                mock_layout.return_value.get_doc_id_from_finding_id.return_value = "123"
                mock_store = Mock()
                mock_store.get_findings_by_doc_id.return_value = sample_document
                mock_store_class.return_value = mock_store
                
                handle = LibraryHandle(temp_library_dir)
                
                from efi_core.types import Finding
                doc_id = Finding.generate_doc_id("https://example.com")
                finding_id = f"{doc_id}_001"
                
                result = handle.get_finding(finding_id)
                assert result is not None
                assert result.finding_id == finding_id
                assert result.text == "Sample finding 1"
    
    def test_get_document_for_finding(self, temp_library_dir, sample_document):
        """Test getting document containing a finding"""
        with patch('efi_library.library_handle.LibraryLayout') as mock_layout:
            with patch('efi_library.library_handle.LibraryStore') as mock_store_class:
                from efi_core.types import Finding
                doc_id = Finding.generate_doc_id("https://example.com")
                
                # Mock the layout methods
                mock_layout.return_value.get_doc_id_from_finding_id.return_value = doc_id
                mock_layout.return_value.doc_findings_path.return_value = Mock(exists=lambda: True)
                mock_layout.return_value.get_url_mapping.return_value = {doc_id: "https://example.com"}
                
                # Mock the store
                mock_store = Mock()
                mock_store.get_findings_by_doc_id.return_value = sample_document
                mock_store_class.return_value = mock_store
                
                # Mock the file reading
                mock_file_content = {
                    'doc_id': doc_id,
                    'findings': [
                        {
                            'finding_id': f'{doc_id}_001',
                            'text': 'Sample finding 1',
                            'confidence': 0.9,
                            'category': 'test',
                            'keywords': []
                        },
                        {
                            'finding_id': f'{doc_id}_002',
                            'text': 'Sample finding 2',
                            'confidence': 0.8,
                            'category': 'test',
                            'keywords': []
                        }
                    ],
                    'title': 'Test Document',
                    'url': 'https://example.com'
                }
                
                with patch('builtins.open', mock_open(read_data=json.dumps(mock_file_content))):
                    handle = LibraryHandle(temp_library_dir)
                    
                    finding_id = f"{doc_id}_001"
                    
                    result = handle.get_document_for_finding(finding_id)
                    assert result is not None
                    assert result.doc_id == doc_id
                    assert len(result.findings) == 2
    
    def test_search_findings(self, temp_library_dir, sample_document):
        """Test searching findings"""
        with patch('efi_library.library_handle.LibraryLayout'):
            with patch('efi_library.library_handle.LibraryStore') as mock_store_class:
                mock_store = Mock()
                mock_store.list_all_findings.return_value = [sample_document]
                mock_store_class.return_value = mock_store
                
                handle = LibraryHandle(temp_library_dir)
                
                results = list(handle.search_findings("Sample finding"))
                assert len(results) == 1
                expected_doc_id = Finding.generate_doc_id("https://example.com")
                assert results[0].doc_id == expected_doc_id
    
    def test_filter_by_category(self, temp_library_dir, sample_document):
        """Test filtering by category"""
        with patch('efi_library.library_handle.LibraryLayout'):
            with patch('efi_library.library_handle.LibraryStore') as mock_store_class:
                mock_store = Mock()
                mock_store.list_all_findings.return_value = [sample_document]
                mock_store_class.return_value = mock_store
                
                handle = LibraryHandle(temp_library_dir)
                
                results = list(handle.filter_by_category("test"))
                assert len(results) == 1
                expected_doc_id = Finding.generate_doc_id("https://example.com")
                assert results[0].doc_id == expected_doc_id
    
    def test_filter_by_confidence(self, temp_library_dir, sample_document):
        """Test filtering by confidence"""
        with patch('efi_library.library_handle.LibraryLayout'):
            with patch('efi_library.library_handle.LibraryStore') as mock_store_class:
                mock_store = Mock()
                mock_store.list_all_findings.return_value = [sample_document]
                mock_store_class.return_value = mock_store
                
                handle = LibraryHandle(temp_library_dir)
                
                results = list(handle.filter_by_confidence(0.85))
                assert len(results) == 1
                expected_doc_id = Finding.generate_doc_id("https://example.com")
                assert results[0].doc_id == expected_doc_id
    
    def test_get_findings_count(self, temp_library_dir):
        """Test getting findings count"""
        with patch('efi_library.library_handle.LibraryLayout') as mock_layout:
            mock_layout.return_value.get_document_ids.return_value = ["123", "456"]
            
            handle = LibraryHandle(temp_library_dir)
            count = handle.get_findings_count()
            assert count == 2
    
    def test_get_library_info(self, temp_library_dir):
        """Test getting library information"""
        with patch('efi_library.library_handle.LibraryLayout') as mock_layout:
            with patch('efi_library.library_handle.LibraryStore') as mock_store_class:
                # Mock the layout methods to simulate new structure
                index_path = Mock()
                index_path.exists = Mock(return_value=True)
                index_path.stat = Mock(return_value=SimpleNamespace(st_size=123, st_mtime=time.time()))
                
                mock_layout.return_value.metadata_path = Mock(exists=lambda: False)
                mock_layout.return_value.findings_path = Mock(exists=lambda: False)
                mock_layout.return_value.index_path = index_path
                mock_layout.return_value.documents_dir = Mock(exists=lambda: True)
                mock_layout.return_value.get_document_ids = Mock(return_value=["123", "456"])
                mock_layout.return_value.ensure_dirs = Mock()
                
                # Mock the store methods
                mock_store = Mock()
                mock_store.list_all_findings = Mock(return_value=[])
                mock_store.get_findings_by_doc_id = Mock(return_value=None)
                mock_store_class.return_value = mock_store
                
                # Also patch Path.exists/stat used in init for documents dir
                with patch('pathlib.Path.exists', return_value=True), \
                     patch('pathlib.Path.stat', return_value=SimpleNamespace(st_size=1, st_mtime=time.time())):
                    handle = LibraryHandle(temp_library_dir)
                    
                    info = handle.get_library_info()
                    assert "library_path" in info
                    assert "structure_type" in info
