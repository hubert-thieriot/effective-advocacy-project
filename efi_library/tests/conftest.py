import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock

from efi_core.types import Finding, LibraryDocument, LibraryDocumentWFindings
from efi_library.library_store import LibraryStore
from efi_library.library_handle import LibraryHandle


@pytest.fixture
def temp_library_dir():
    """Create a temporary directory for testing"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_findings():
    """Sample findings for testing"""
    from efi_core.types import Finding
    
    # Generate the correct finding IDs based on the URL
    doc_id = Finding.generate_doc_id("https://example.com")
    
    return [
        Finding(
            text="Sample finding 1",
            finding_id=f"{doc_id}_001",
            confidence=0.9,
            category="test"
        ),
        Finding(
            text="Sample finding 2", 
            finding_id=f"{doc_id}_002",
            confidence=0.8,
            category="test"
        )
    ]


@pytest.fixture
def sample_document(sample_findings):
    """Sample document with findings"""
    from efi_core.types import Finding
    
    # Generate the correct doc_id from the URL
    doc_id = Finding.generate_doc_id("https://example.com")
    
    return LibraryDocumentWFindings(
        doc_id=doc_id,
        url="https://example.com",
        title="Test Document",
        findings=sample_findings
    )


@pytest.fixture
def mock_store():
    """Mock store for testing"""
    store = Mock(spec=LibraryStore)
    store.list_all_findings.return_value = []
    store.get_findings_by_doc_id.return_value = None
    return store
