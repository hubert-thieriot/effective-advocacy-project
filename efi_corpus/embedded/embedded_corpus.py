from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Iterator
import numpy as np

from efi_core.types import Chunk
from efi_core.protocols import Chunker, Embedder
from efi_core.layout import EmbeddedCorpusLayout
from efi_core.stores import IndexStore
from efi_core.embedded import EmbeddedCollection

from efi_corpus import CorpusHandle


@dataclass
class EmbeddedCorpus(EmbeddedCollection[CorpusHandle]):
    """
    Embedded corpus with chunks, embeddings, and search index.

    Provides chunking, embedding, and indexing capabilities for document corpora.
    """

    # Rename fields to match base class expectations
    corpus_path: Path
    workspace_path: Path
    chunker: Chunker
    embedder: Embedder

    def __post_init__(self) -> None:
        # Set data_path for base class
        self.data_path = self.corpus_path
        # Call parent __post_init__
        super().__post_init__()
        # Keep corpus reference for compatibility
        self.corpus = self.handle

    # Implement abstract methods

    def _create_handle(self) -> CorpusHandle:
        """Create CorpusHandle"""
        return CorpusHandle(self.corpus_path)

    def _create_layout(self) -> EmbeddedCorpusLayout:
        """Create EmbeddedCorpusLayout"""
        return EmbeddedCorpusLayout(Path(self.corpus_path), Path(self.workspace_path))

    def _create_index_store(self) -> IndexStore:
        """Create IndexStore for corpus"""
        return IndexStore(
            workspace_path=self.workspace_path,
            corpus_or_library_name=self.corpus_path.name,
            is_corpus=True
        )

    def _get_item_text(self, item_id: str) -> str:
        """Get document text"""
        doc = self.handle.get_document(item_id)
        if doc is None:
            raise FileNotFoundError(f"Document not found: {item_id}")
        return doc.text

    def _iter_items(self) -> Iterator:
        """Iterate documents"""
        return self.handle.iter_documents()

    def _get_item_count(self) -> int:
        """Get document count"""
        return self.handle.get_document_count()

    def _get_item_id(self, item) -> str:
        """Extract doc_id from document"""
        return item.doc_id

    def _get_collection_name(self) -> str:
        """Collection name for display"""
        return "documents"

    def _get_info_dict(self) -> dict:
        """Get corpus information dictionary"""
        return {
            "corpus_path": str(self.corpus_path),
            "workspace_path": str(self.workspace_path),
            "document_count": self.handle.get_document_count(),
            "chunker_spec": self.chunker_spec.name,
            "embedder_spec": self.embedder_spec.model_name
        }

    # Corpus-specific methods

    def get_chunks(self, doc_id: str, materialize_if_necessary: bool = False) -> Optional[List[Chunk]]:
        """
        Get chunks for a document as Chunk objects (corpus-specific behavior).

        Args:
            doc_id: Document ID
            materialize_if_necessary: If True, materialize chunks if they don't exist

        Returns:
            List of Chunk objects if found, None otherwise
        """
        # Get raw chunks from parent
        chunks = super().get_chunks(doc_id, materialize_if_necessary)
        if chunks is None:
            return None

        # Convert to Chunk objects (corpus-specific)
        return [Chunk(chunk_id=i, text=chunk, doc_id=doc_id) for i, chunk in enumerate(chunks)]

    def get_chunk(self, chunk_id: str, materialize_if_necessary: bool = False) -> Optional[Chunk]:
        """
        Get a specific chunk by chunk ID.

        Args:
            chunk_id: Chunk ID in format 'doc_id_chunk_number'
            materialize_if_necessary: If True, materialize chunks if they don't exist

        Returns:
            The specific Chunk object if found, None otherwise
        """
        parsed = self._parse_chunk_id(chunk_id)
        if parsed is None:
            return None

        doc_id, chunk_number = parsed

        chunks = self.get_chunks(doc_id, materialize_if_necessary=materialize_if_necessary)
        if chunks is None:
            return None

        chunk = next((x for x in chunks if x.chunk_id == chunk_number), None)

        return chunk

    def _parse_chunk_id(self, chunk_id: str) -> Optional[tuple[str, int]]:
        """
        Parse chunk ID to extract doc_id and chunk_number.

        Args:
            chunk_id: Chunk ID in format 'doc_id_chunk_number'

        Returns:
            Tuple of (doc_id, chunk_number) if valid, None otherwise
        """
        try:
            # Parse chunk_id to extract doc_id and chunk_number
            chunk_parts = chunk_id.split('_chunk_')
            if len(chunk_parts) != 2:
                return None

            doc_id = chunk_parts[0]
            chunk_number_str = chunk_parts[1]

            chunk_number = int(chunk_number_str)
            return doc_id, chunk_number

        except (ValueError, IndexError):
            return None

    def get_corpus_info(self) -> dict:
        """Alias for get_info_dict() for backward compatibility"""
        return self._get_info_dict()
