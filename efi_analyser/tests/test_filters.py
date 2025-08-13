"""
Tests for efi_analyser filters
"""

import pytest
from efi_analyser.filters import TextContainsFilter, MetadataFilter, CompositeFilter
from efi_corpus.types import Document


class TestTextContainsFilter:
    """Test TextContainsFilter"""
    
    def test_filter_init(self):
        """Test filter initialization"""
        filter_obj = TextContainsFilter(
            terms=["CREA", "Coal"],
            case_sensitive=False,
            name="test_filter"
        )
        
        assert filter_obj.terms == ["CREA", "Coal"]
        assert filter_obj.case_sensitive is False
        assert filter_obj.name == "test_filter"
    
    def test_filter_init_default_name(self):
        """Test filter initialization with default name"""
        filter_obj = TextContainsFilter(terms=["CREA"])
        assert filter_obj.name == "text_contains_1_terms"
    
    def test_filter_case_insensitive(self):
        """Test case-insensitive filtering"""
        filter_obj = TextContainsFilter(terms=["CREA", "coal"], case_sensitive=False)
        
        # Should match regardless of case
        doc = Document(
            doc_id="test",
            url="https://example.com",
            title="Test",
            text="This document mentions crea and COAL",
            published_at=None,
            language=None,
            meta={}
        )
        
        assert filter_obj.apply(doc) is True
    
    def test_filter_case_sensitive(self):
        """Test case-sensitive filtering"""
        filter_obj = TextContainsFilter(terms=["CREA", "coal"], case_sensitive=True)
        
        # Should only match exact case
        doc = Document(
            doc_id="test",
            url="https://example.com",
            title="Test",
            text="This document mentions crea and COAL",
            published_at=None,
            language=None,
            meta={}
        )
        
        assert filter_obj.apply(doc) is False
        
        # Should match with correct case
        doc.text = "This document mentions CREA and coal"
        assert filter_obj.apply(doc) is True
    
    def test_filter_no_matches(self):
        """Test filter when no terms match"""
        filter_obj = TextContainsFilter(terms=["CREA", "coal"])
        
        doc = Document(
            doc_id="test",
            url="https://example.com",
            title="Test",
            text="This document mentions neither term",
            published_at=None,
            language=None,
            meta={}
        )
        
        assert filter_obj.apply(doc) is False


class TestMetadataFilter:
    """Test MetadataFilter"""
    
    def test_filter_init(self):
        """Test filter initialization"""
        filter_obj = MetadataFilter(
            field="collection",
            value="National",
            operator="eq",
            name="test_filter"
        )
        
        assert filter_obj.field == "collection"
        assert filter_obj.value == "National"
        assert filter_obj.operator == "eq"
        assert filter_obj.name == "test_filter"
    
    def test_filter_eq_operator(self):
        """Test equals operator"""
        filter_obj = MetadataFilter(field="collection", value="National")
        
        doc = Document(
            doc_id="test",
            url="https://example.com",
            title="Test",
            text="Content",
            published_at=None,
            language=None,
            meta={"collection": "National"}
        )
        
        assert filter_obj.apply(doc) is True
        
        doc.meta["collection"] = "Local"
        assert filter_obj.apply(doc) is False
    
    def test_filter_missing_field(self):
        """Test filter when field is missing"""
        filter_obj = MetadataFilter(field="collection", value="National")
        
        doc = Document(
            doc_id="test",
            url="https://example.com",
            title="Test",
            text="Content",
            published_at=None,
            language=None,
            meta={}  # No collection field
        )
        
        assert filter_obj.apply(doc) is False
    
    def test_filter_in_operator(self):
        """Test 'in' operator"""
        filter_obj = MetadataFilter(
            field="category",
            value=["news", "analysis"],
            operator="in"
        )
        
        doc = Document(
            doc_id="test",
            url="https://example.com",
            title="Test",
            text="Content",
            published_at=None,
            language=None,
            meta={"category": "news"}
        )
        
        assert filter_obj.apply(doc) is True
        
        doc.meta["category"] = "opinion"
        assert filter_obj.apply(doc) is False


class TestCompositeFilter:
    """Test CompositeFilter"""
    
    def test_composite_filter_init(self):
        """Test composite filter initialization"""
        filter1 = TextContainsFilter(terms=["CREA"])
        filter2 = MetadataFilter(field="language", value="en")
        
        composite = CompositeFilter([filter1, filter2], name="test_composite")
        
        assert len(composite.filters) == 2
        assert composite.name == "test_composite"
    
    def test_composite_filter_all_pass(self):
        """Test composite filter when all filters pass"""
        filter1 = TextContainsFilter(terms=["CREA"])
        filter2 = MetadataFilter(field="language", value="en")
        
        composite = CompositeFilter([filter1, filter2])
        
        doc = Document(
            doc_id="test",
            url="https://example.com",
            title="Test",
            text="This mentions CREA",
            published_at=None,
            language=None,
            meta={"language": "en"}
        )
        
        assert composite.apply(doc) is True
    
    def test_composite_filter_one_fails(self):
        """Test composite filter when one filter fails"""
        filter1 = TextContainsFilter(terms=["CREA"])
        filter2 = MetadataFilter(field="language", value="en")
        
        composite = CompositeFilter([filter1, filter2])
        
        doc = Document(
            doc_id="test",
            url="https://example.com",
            title="Test",
            text="This mentions CREA",
            published_at=None,
            language=None,
            meta={"language": "fr"}  # Wrong language
        )
        
        assert composite.apply(doc) is False
