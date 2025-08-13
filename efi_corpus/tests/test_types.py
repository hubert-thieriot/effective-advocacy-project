"""
Tests for efi_corpus types
"""

import pytest
from efi_corpus.types import Document


class TestDocument:
    """Test Document class"""
    
    def test_document_creation(self):
        """Test creating a Document instance"""
        doc = Document(
            doc_id="test123",
            url="https://example.com/article",
            title="Test Article",
            text="This is test content",
            published_at="2025-01-01",
            language="en",
            meta={"source": "test", "category": "news"}
        )
        
        assert doc.doc_id == "test123"
        assert doc.url == "https://example.com/article"
        assert doc.title == "Test Article"
        assert doc.text == "This is test content"
        assert doc.published_at == "2025-01-01"
        assert doc.language == "en"
        assert doc.meta["source"] == "test"
        assert doc.meta["category"] == "news"
    
    def test_document_optional_fields(self):
        """Test Document with optional fields as None"""
        doc = Document(
            doc_id="test123",
            url="https://example.com/article",
            title=None,
            text="Content only",
            published_at=None,
            language=None,
            meta={}
        )
        
        assert doc.title is None
        assert doc.published_at is None
        assert doc.language is None
        assert doc.meta == {}
