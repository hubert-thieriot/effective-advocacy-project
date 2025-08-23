import pytest
from unittest.mock import Mock, patch

from efi_core.types import Finding, LibraryDocumentWFindings
from efi_library.library_builder import LibraryBuilder


class TestLibraryBuilder:
    """Test LibraryBuilder functionality"""
    
    def test_init_creates_builder(self):
        """Test that builder initializes correctly"""
        mock_collector = Mock()
        mock_extractors = [Mock()]
        mock_store = Mock()
        
        builder = LibraryBuilder("test_lib", mock_collector, mock_extractors, mock_store)
        
        assert builder.name == "test_lib"
        assert builder.collector == mock_collector
        assert builder.extractors == mock_extractors
        assert builder.store == mock_store
    
    def test_build_library_collects_and_processes_urls(self):
        """Test that builder collects URLs and processes them"""
        mock_collector = Mock()
        mock_collector.collect_urls_limited.return_value = ["https://example.com"]
        
        mock_extractor = Mock()
        mock_extractor.name = "test_extractor"
        mock_extractor.extract_findings.return_value = LibraryDocumentWFindings(
            doc_id="123",
            url="https://example.com",
            title="Test",
            findings=[Finding(text="Test finding", finding_id="123_001")]
        )
        
        mock_store = Mock()
        mock_store.store_findings.return_value = True
        
        builder = LibraryBuilder("test_lib", mock_collector, [mock_extractor], mock_store)
        
        result = builder.build_library()
        
        assert result["total_sources_processed"] == 1
        assert result["successful_extractions"] == 1
        assert result["total_findings_count"] == 1
        mock_store.store_findings.assert_called_once()
    
    def test_build_library_handles_failed_extractions(self):
        """Test that builder handles failed extractions gracefully"""
        mock_collector = Mock()
        mock_collector.collect_urls_limited.return_value = ["https://example.com"]
        
        mock_extractor = Mock()
        mock_extractor.name = "test_extractor"
        mock_extractor.extract_findings.return_value = None
        
        mock_store = Mock()
        
        builder = LibraryBuilder("test_lib", mock_collector, [mock_extractor], mock_store)
        
        result = builder.build_library()
        
        assert result["total_sources_processed"] == 1
        assert result["successful_extractions"] == 0
        assert result["total_findings_count"] == 0
    
    def test_build_library_respects_max_sources(self):
        """Test that builder respects max_sources limit"""
        mock_collector = Mock()
        mock_collector.collect_urls_limited.return_value = ["url1", "url2", "url3"]
        
        mock_extractor = Mock()
        mock_extractor.extract_findings.return_value = None
        
        mock_store = Mock()
        
        builder = LibraryBuilder("test_lib", mock_collector, [mock_extractor], mock_store)
        
        result = builder.build_library(max_sources=2)
        
        assert result["total_sources_processed"] == 2
    
    def test_get_library_stats(self):
        """Test getting library statistics"""
        mock_collector = Mock()
        mock_collector.get_collection_info.return_value = {"type": "test"}
        
        mock_extractors = [Mock()]
        mock_extractors[0].name = "test_extractor"
        mock_extractors[0].get_cache_stats.return_value = {"hits": 10}
        
        mock_store = Mock()
        mock_store.get_storage_stats.return_value = {"documents": 5}
        
        builder = LibraryBuilder("test_lib", mock_collector, mock_extractors, mock_store)
        
        stats = builder.get_library_stats()
        
        assert stats["library_name"] == "test_lib"
        assert "collector_info" in stats
        assert "extractor_info" in stats
        assert "store_info" in stats
    
    def test_export_library(self):
        """Test exporting library data"""
        mock_collector = Mock()
        mock_extractors = [Mock()]
        mock_store = Mock()
        
        mock_store.list_all_findings.return_value = [
            LibraryDocumentWFindings(
                doc_id="123",
                url="https://example.com",
                title="Test",
                findings=[Finding(text="Test finding", finding_id="123_001")]
            )
        ]
        
        builder = LibraryBuilder("test_lib", mock_collector, mock_extractors, mock_store)
        
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value = Mock()
            export_path = builder.export_library()
            
            assert "test_lib_findings_export.json" in str(export_path)
            mock_store.list_all_findings.assert_called_once()
