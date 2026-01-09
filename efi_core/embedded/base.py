"""
Generic base class for embedded collections (corpora and libraries)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar, Optional, List, Iterator, Tuple
import numpy as np

from tqdm import tqdm

from efi_core.types import ChunkerSpec, EmbedderSpec
from efi_core.protocols import Chunker, Embedder
from efi_core.stores import ChunkStore, EmbeddingStore, IndexStore


# Type variable for the handle type (CorpusHandle or LibraryHandle)
T = TypeVar('T')


@dataclass
class EmbeddedCollection(ABC, Generic[T]):
    """
    Generic base class for embedded collections (corpora or libraries).

    Provides shared functionality for:
    - Chunking text items (documents or findings)
    - Embedding chunks
    - Building FAISS indexes
    - Statistics and iteration

    Subclasses (EmbeddedCorpus, EmbeddedLibrary) provide specific implementations
    for different collection types.
    """

    data_path: Path
    workspace_path: Path
    chunker: Chunker
    embedder: Embedder

    def __post_init__(self) -> None:
        """Initialize stores and layout"""
        # Subclasses initialize their specific handle and layout
        self.handle: T = self._create_handle()
        self.layout = self._create_layout()

        # Shared stores
        self.chunk_store = ChunkStore(self.layout)
        self.embedding_store = EmbeddingStore(self.layout)
        self.index_store = self._create_index_store()

    # Abstract methods that subclasses must implement

    @abstractmethod
    def _create_handle(self) -> T:
        """Create the appropriate handle (CorpusHandle or LibraryHandle)"""
        ...

    @abstractmethod
    def _create_layout(self):
        """Create the appropriate layout (EmbeddedCorpusLayout or EmbeddedLibraryLayout)"""
        ...

    @abstractmethod
    def _create_index_store(self) -> IndexStore:
        """Create index store with correct is_corpus flag"""
        ...

    @abstractmethod
    def _get_item_text(self, item_id: str) -> str:
        """Get text for an item (document or finding)"""
        ...

    @abstractmethod
    def _iter_items(self) -> Iterator:
        """Iterate items from handle (documents or findings)"""
        ...

    @abstractmethod
    def _get_item_count(self) -> int:
        """Get total item count"""
        ...

    @abstractmethod
    def _get_item_id(self, item) -> str:
        """Extract item ID from an item object"""
        ...

    @abstractmethod
    def _get_collection_name(self) -> str:
        """Get collection name for display purposes"""
        ...

    @abstractmethod
    def _get_info_dict(self) -> dict:
        """Get collection-specific info dictionary"""
        ...

    # Properties

    @property
    def chunker_spec(self) -> ChunkerSpec:
        """Get chunker specification from chunker instance"""
        return self.chunker.spec

    @property
    def embedder_spec(self) -> EmbedderSpec:
        """Get embedder specification from embedder instance"""
        return self.embedder.spec

    # Shared methods - chunks and embeddings

    def get_chunks(self, item_id: str, materialize_if_necessary: bool = False) -> Optional[List[str]]:
        """
        Get chunks for an item.

        Args:
            item_id: Item identifier (doc_id or finding_id)
            materialize_if_necessary: If True, materialize chunks if they don't exist

        Returns:
            List of chunks if found, None otherwise
        """
        if materialize_if_necessary:
            text = self._get_item_text(item_id)
            return self.chunk_store.materialize(item_id, text, self.chunker_spec, self.chunker)
        else:
            # Just read existing chunks
            return self.chunk_store.read(item_id, self.chunker_spec)

    def get_embeddings(self, item_id: str, materialize_if_necessary: bool = False) -> Optional[np.ndarray]:
        """
        Get embeddings for an item.

        Args:
            item_id: Item identifier (doc_id or finding_id)
            materialize_if_necessary: If True, materialize embeddings if they don't exist

        Returns:
            Embeddings array if found, None otherwise
        """
        if materialize_if_necessary:
            chunks = self.get_chunks(item_id, materialize_if_necessary=True)
            if chunks is None:
                return None
            return self.embedding_store.materialize(item_id, chunks, self.chunker_spec, self.embedder)
        else:
            # Just read existing embeddings
            return self.embedding_store.read(item_id, self.chunker_spec, self.embedder_spec)

    # Compatibility helpers

    def chunks(self, item_id: str):
        """Compatibility wrapper for get_chunks with materialization"""
        return self.get_chunks(item_id, materialize_if_necessary=True)

    def embeddings(self, item_id: str):
        """Compatibility wrapper for get_embeddings with materialization"""
        return self.get_embeddings(item_id, materialize_if_necessary=True)

    # Building methods

    def build_chunks(self, max_items: Optional[int] = None, item_ids: Optional[List[str]] = None) -> None:
        """
        Build chunks for items in the collection.

        Args:
            max_items: Maximum number of items to process (None for all)
            item_ids: Specific item IDs to process (None for all items)
        """
        if item_ids is not None:
            # Build chunks for specific items
            total_items = len(item_ids)
            print(f"Building chunks for {total_items} specific {self._get_collection_name()}...")
            with tqdm(total=total_items, desc="Building chunks", unit="item") as pbar:
                for item_id in item_ids:
                    _ = self.get_chunks(item_id, materialize_if_necessary=True)
                    pbar.update(1)
        else:
            # Build chunks for all items
            if max_items is None:
                total_items = self._get_item_count()
            else:
                total_items = max_items

            print(f"Building chunks for {total_items} {self._get_collection_name()}...")
            with tqdm(total=total_items, desc="Building chunks", unit="item") as pbar:
                item_count = 0
                for item in self._iter_items():
                    if max_items and item_count >= max_items:
                        break

                    item_id = self._get_item_id(item)
                    _ = self.get_chunks(item_id, materialize_if_necessary=True)
                    pbar.update(1)
                    item_count += 1

    def build_embeddings(self, max_items: Optional[int] = None, item_ids: Optional[List[str]] = None) -> None:
        """
        Build embeddings for items in the collection (assumes chunks exist).

        Args:
            max_items: Maximum number of items to process (None for all)
            item_ids: Specific item IDs to process (None for all items)
        """
        if item_ids is not None:
            # Build embeddings for specific items
            total_items = len(item_ids)
            print(f"Building embeddings for {total_items} specific {self._get_collection_name()}...")
            with tqdm(total=total_items, desc="Building embeddings", unit="item") as pbar:
                for item_id in item_ids:
                    _ = self.get_embeddings(item_id, materialize_if_necessary=True)
                    pbar.update(1)
        else:
            # Build embeddings for all items
            if max_items is None:
                total_items = self._get_item_count()
            else:
                total_items = max_items

            print(f"Building embeddings for {total_items} {self._get_collection_name()}...")
            with tqdm(total=total_items, desc="Building embeddings", unit="item") as pbar:
                item_count = 0
                for item in self._iter_items():
                    if max_items and item_count >= max_items:
                        break

                    item_id = self._get_item_id(item)
                    _ = self.get_embeddings(item_id, materialize_if_necessary=True)
                    pbar.update(1)
                    item_count += 1

    def build_all(self, reindex: bool = True, max_items: Optional[int] = None, item_ids: Optional[List[str]] = None) -> None:
        """
        Build chunks, embeddings, and optionally FAISS index for items.

        Args:
            reindex: Whether to build FAISS index after embeddings
            max_items: Maximum number of items to process (None for all)
            item_ids: Specific item IDs to process (None for all items)
        """
        self.build_chunks(max_items=max_items, item_ids=item_ids)
        self.build_embeddings(max_items=max_items, item_ids=item_ids)

        if reindex:
            self.build_index()

    def build_index(self) -> bool:
        """Build FAISS index for fast retrieval"""
        # Collect all embeddings for the specific chunker and embedder specs
        embeddings_data = []
        total_chunks = 0

        for item_id, embedding_array in self.iter_embeddings():
            if embedding_array.size == 0:
                continue

            # Handle both 1D and 2D arrays
            if embedding_array.ndim == 1:
                # Single chunk - create chunk_0 ID
                chunk_id = f"{item_id}_chunk_0"
                embeddings_data.append((chunk_id, embedding_array))
                total_chunks += 1
            else:
                # Multiple chunks - create separate IDs for each
                for chunk_idx, chunk_embedding in enumerate(embedding_array):
                    chunk_id = f"{item_id}_chunk_{chunk_idx}"
                    embeddings_data.append((chunk_id, chunk_embedding))
                    total_chunks += 1

        if not embeddings_data:
            print("No embeddings found for index building")
            return False

        # Count items for display
        num_items = len(list(self.iter_embeddings()))
        print(f"Building index with {len(embeddings_data)} chunk-level embeddings from {num_items} {self._get_collection_name()}...")

        # Build index using instance index store
        success = self.index_store.build_index(
            chunker_spec=self.chunker_spec,
            embedder_spec=self.embedder_spec,
            embeddings_data=embeddings_data
        )

        if success:
            print(f"Successfully built index with {total_chunks} total chunks")
        else:
            print(f"Failed to build index")

        return success

    # Statistics methods

    def get_stats(self) -> dict:
        """Get detailed statistics about the embedded collection"""
        stats = {
            "collection_info": self._get_info_dict(),
            "storage_stats": {}
        }

        # Fast file system checks for chunks and embeddings
        try:
            chunk_count = self._fast_count_chunks()
            embedding_count = self._fast_count_embeddings()
        except Exception as e:
            chunk_count = f"Error: {e}"
            embedding_count = f"Error: {e}"

        # Check if index exists
        index_exists = False
        try:
            index_exists = self.index_store.has_index(self.chunker_spec, self.embedder_spec)
        except Exception as e:
            index_exists = f"Error: {e}"

        stats["storage_stats"] = {
            "chunks_available": chunk_count,
            "embeddings_available": embedding_count,
            "index_exists": index_exists
        }

        return stats

    def show_stats(self) -> None:
        """Print detailed statistics about the embedded collection"""
        stats = self.get_stats()
        collection_type = self._get_collection_name().title()
        print(f"Embedded {collection_type} Statistics:")
        print("=" * 40)

        # Print collection info
        print(f"{collection_type} Information:")
        for key, value in stats["collection_info"].items():
            print(f"  {key}: {value}")

        print("\nStorage Statistics:")
        for key, value in stats["storage_stats"].items():
            print(f"  {key}: {value}")

    def _fast_count_chunks(self) -> int:
        """Fast count of available chunks using file system"""
        chunks_dir = self.layout.chunks_dir(self.chunker_spec)
        if not chunks_dir.exists():
            return 0

        # Count .chunks.jsonl files in chunks directory
        count = 0
        for chunk_file in chunks_dir.glob("*.chunks.jsonl"):
            count += 1
        return count

    def _fast_count_embeddings(self) -> int:
        """Fast count of available embeddings using file system"""
        embeddings_dir = self.layout.embeddings_dir(self.chunker_spec, self.embedder_spec)
        if not embeddings_dir.exists():
            return 0

        # Count .npy files in embeddings directory
        count = 0
        for embedding_file in embeddings_dir.glob("*.npy"):
            count += 1
        return count

    def iter_embeddings(self) -> Iterator[Tuple[str, np.ndarray]]:
        """
        Iterate through existing embeddings in the collection for the instance's chunker and embedder.

        Yields:
            Tuple[str, np.ndarray]: (item_id, embedding_vector) pairs
        """
        # Get the embeddings directory for this instance's specs
        embeddings_dir = self.layout.embeddings_dir(self.chunker_spec, self.embedder_spec)
        if not embeddings_dir.exists():
            return

        # Iterate through existing .npy files
        for embedding_file in embeddings_dir.glob("*.npy"):
            try:
                # Extract item_id from filename (remove .npy extension)
                item_id = embedding_file.stem

                # Load the embedding
                embedding = np.load(embedding_file)

                if embedding.size > 0:
                    yield (item_id, embedding)

            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Error loading embedding from {embedding_file}: {e}")
                continue
