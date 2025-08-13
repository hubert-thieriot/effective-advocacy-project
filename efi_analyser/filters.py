"""
Document filters for the analysis system
"""

import re
from typing import List, Dict, Any
from .types import Filter
from efi_corpus.types import Document


class TextContainsFilter(Filter):
    """Filter documents based on text content"""
    
    def __init__(self, terms: List[str], case_sensitive: bool = False, name: str = None):
        """
        Initialize filter
        
        Args:
            terms: List of terms to search for
            case_sensitive: Whether search should be case sensitive
            name: Optional name for this filter
        """
        self.terms = terms
        self.case_sensitive = case_sensitive
        self.name = name or f"text_contains_{len(terms)}_terms"
        
        # Compile regex patterns for efficiency
        if case_sensitive:
            self.patterns = [re.compile(re.escape(term)) for term in terms]
        else:
            self.patterns = [re.compile(re.escape(term), re.IGNORECASE) for term in terms]
    
    def apply(self, document: Document) -> bool:
        """Check if document contains any of the specified terms"""
        text = document.text
        return any(pattern.search(text) for pattern in self.patterns)


class MetadataFilter(Filter):
    """Filter documents based on metadata fields"""
    
    def __init__(self, field: str, value: Any, operator: str = "eq", name: str = None):
        """
        Initialize metadata filter
        
        Args:
            field: Metadata field name
            value: Value to compare against
            operator: Comparison operator ("eq", "ne", "gt", "lt", "gte", "lte", "in", "contains")
            name: Optional name for this filter
        """
        self.field = field
        self.value = value
        self.operator = operator
        self.name = name or f"metadata_{field}_{operator}_{value}"
    
    def apply(self, document: Document) -> bool:
        """Check if document metadata matches the filter criteria"""
        if self.field not in document.meta:
            return False
            
        field_value = document.meta[self.field]
        
        if self.operator == "eq":
            return field_value == self.value
        elif self.operator == "ne":
            return field_value != self.value
        elif self.operator == "gt":
            return field_value > self.value
        elif self.operator == "lt":
            return field_value < self.value
        elif self.operator == "gte":
            return field_value >= self.value
        elif self.operator == "lte":
            return field_value <= self.value
        elif self.operator == "in":
            return field_value in self.value
        elif self.operator == "contains":
            return self.value in field_value
        else:
            raise ValueError(f"Unknown operator: {self.operator}")


class CompositeFilter(Filter):
    """Combine multiple filters with AND logic"""
    
    def __init__(self, filters: List[Filter], name: str = None):
        """
        Initialize composite filter
        
        Args:
            filters: List of filters to combine
            name: Optional name for this filter
        """
        self.filters = filters
        self.name = name or f"composite_{len(filters)}_filters"
    
    def apply(self, document: Document) -> bool:
        """Apply all filters - document must pass ALL filters (AND logic)"""
        return all(filter_obj.apply(document) for filter_obj in self.filters)
