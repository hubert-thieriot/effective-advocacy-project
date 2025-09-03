"""
Retrieval module for vector search over embedded data.

Provides two retriever implementations:
- RetrieverBrute: Always uses brute-force cosine similarity
- RetrieverIndex: Uses FAISS indexes with auto-rebuild capability
"""

from efi_core.types import Retriever
from .retriever_brute_force import RetrieverBrute
from .retriever_index import RetrieverIndex
from .index_builder import IndexBuilder

__all__ = [
    'Retriever',
    'RetrieverBrute',
    'RetrieverIndex',
    'IndexBuilder'
]
