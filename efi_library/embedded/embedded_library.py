from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from efi_core.protocols import Chunker, Embedder
from efi_core.layout import EmbeddedLibraryLayout
from efi_core.stores import IndexStore
from efi_core.embedded import EmbeddedCollection

from efi_library import LibraryHandle


@dataclass
class EmbeddedLibrary(EmbeddedCollection[LibraryHandle]):
    """
    Embedded library with chunks, embeddings, and search index.

    Provides chunking, embedding, and indexing capabilities for findings libraries.
    """

    library_path: Path
    workspace_path: Path
    chunker: Chunker
    embedder: Embedder

    def __post_init__(self) -> None:
        # Set data_path for base class
        self.data_path = self.library_path
        # Call parent __post_init__
        super().__post_init__()
        # Keep library reference for compatibility
        self.library = self.handle

    # Implement abstract methods

    def _create_handle(self) -> LibraryHandle:
        """Create LibraryHandle"""
        return LibraryHandle(self.library_path)

    def _create_layout(self) -> EmbeddedLibraryLayout:
        """Create EmbeddedLibraryLayout"""
        return EmbeddedLibraryLayout(Path(self.library_path), Path(self.workspace_path))

    def _create_index_store(self) -> IndexStore:
        """Create IndexStore for library"""
        return IndexStore(
            workspace_path=self.workspace_path,
            corpus_or_library_name=self.library_path.name,
            is_corpus=False
        )

    def _get_item_text(self, item_id: str) -> str:
        """Get finding text"""
        finding = self.handle.get_finding(item_id)
        if not finding:
            raise FileNotFoundError(f"Finding not found: {item_id}")
        return finding.text

    def _iter_items(self) -> Iterator:
        """Iterate findings (flattened)"""
        for doc_findings in self.handle.iter_findings():
            for finding in doc_findings.findings:
                yield finding

    def _get_item_count(self) -> int:
        """Get finding count"""
        return self.handle.get_findings_count()

    def _get_item_id(self, item) -> str:
        """Extract finding_id from finding"""
        return item.finding_id

    def _get_collection_name(self) -> str:
        """Collection name for display"""
        return "findings"

    def _get_info_dict(self) -> dict:
        """Get library information dictionary"""
        return {
            "library_path": str(self.library_path),
            "workspace_path": str(self.workspace_path),
            "document_count": self.handle.get_documents_count(),
            "total_findings_count": self.handle.get_findings_count(),
            "chunker_spec": self.chunker_spec.name,
            "embedder_spec": self.embedder_spec.model_name
        }

    # Library-specific methods (if any)

    def get_library_info(self) -> dict:
        """Alias for get_info_dict() for backward compatibility"""
        return self._get_info_dict()
