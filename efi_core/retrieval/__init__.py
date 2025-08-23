"""
Retrieval module for vector search over embedded data.

Provides two retriever implementations:
- RetrieverBrute: Always uses brute-force cosine similarity
- RetrieverIndex: Uses FAISS indexes with auto-rebuild capability
"""

from .retriever import SearchResult
from .retriever_brute_force import RetrieverBrute
from .retriever_index import RetrieverIndex
from .index_builder import IndexBuilder

__all__ = [
    'SearchResult',
    'RetrieverBrute',
    'RetrieverIndex',
    'IndexBuilder'
]
