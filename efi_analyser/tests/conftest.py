"""
Pytest configuration and fixtures for efi_analyser tests
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from efi_corpus.types import Document


@pytest.fixture
def sample_documents():
    """Sample documents for testing without real corpora"""
    return [
        Document(
            doc_id="doc1",
            url="https://example.com/doc1",
            title="CREA Report on Air Quality",
            text="CREA published a report on air quality. The organization focuses on coal and transport emissions.",
            published_at="2023-01-01",
            language="en",
            meta={"source": "test"}
        ),
        Document(
            doc_id="doc2",
            url="https://example.com/doc2",
            title="Transport and Coal Analysis",
            text="Transport sector contributes to pollution. Coal mining affects air quality. CREA's findings show this.",
            published_at="2023-01-02",
            language="en",
            meta={"source": "test"}
        ),
        Document(
            doc_id="doc3",
            url="https://example.com/doc3",
            title="No Keywords Here",
            text="This document doesn't contain any of the keywords we're looking for.",
            published_at="2023-01-03",
            language="en",
            meta={"source": "test"}
        )
    ]


@pytest.fixture
def mock_corpus_reader(sample_documents):
    """Mock corpus reader that returns sample documents"""
    mock_reader = Mock()
    mock_reader.read_documents.return_value = sample_documents
    mock_reader.get_document_count.return_value = len(sample_documents)
    return mock_reader


@pytest.fixture
def test_keywords():
    """Test keywords for keyword extraction tests"""
    return ["CREA", "coal", "transport"]


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary output directory for tests"""
    return tmp_path / "test_output"
