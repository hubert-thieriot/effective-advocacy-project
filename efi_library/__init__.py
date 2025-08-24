"""
EFI Library module for building and managing findings libraries.

This module provides tools for:
- Building findings libraries from various sources
- Storing and retrieving findings data
- Managing library metadata and organization
"""

from .library_handle import LibraryHandle
from .library_store import LibraryStore
from .example import ExampleLibraryBuilder

__all__ = [
    'LibraryHandle',
    'LibraryStore',
    'ExampleLibraryBuilder'
]
