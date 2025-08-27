from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import numpy as np

from tqdm import tqdm

from efi_core.types import ChunkerSpec, EmbedderSpec, Chunk
from efi_core.protocols import Chunker, Embedder
from efi_core.layout import EmbeddedCorpusLayout
from efi_core.stores import ChunkStore, EmbeddingStore, IndexStore

from efi_corpus import CorpusHandle


@dataclass
class EmbeddedCorpus:
    corpus_path: Path
    workspace_path: Path
    chunker: Chunker
    embedder: Embedder

    def __post_init__(self) -> None:
        self.corpus = CorpusHandle(self.corpus_path)
        self.layout = EmbeddedCorpusLayout(Path(self.corpus_path), Path(self.workspace_path))
        self.chunk_store = ChunkStore(self.layout)
        self.embedding_store = EmbeddingStore(self.layout)

        
        # Initialize index store
        self.index_store = IndexStore(
            workspace_path=self.workspace_path,
            corpus_or_library_name=self.corpus.corpus_path.name,
            is_corpus=True
        )
    
    @property
    def chunker_spec(self) -> ChunkerSpec:
        """Get chunker specification from chunker instance."""
        return self.chunker.spec
    
    @property
    def embedder_spec(self) -> EmbedderSpec:
        """Get embedder specification from embedder instance."""
        return self.embedder.spec

    def get_chunks(self, doc_id: str, materialize_if_necessary: bool = False) -> Optional[List[Chunk]]:
        """
        Get chunks for a document.
        
        Args:
            doc_id: Document ID
            materialize_if_necessary: If True, materialize chunks if they don't exist
            
        Returns:
            List of Chunk objects if found, None otherwise
        """
        if materialize_if_necessary:
            doc = self.corpus.get_document(doc_id)
            if doc is None:
                raise FileNotFoundError(f"Document not found: {doc_id}")
            chunks = self.chunk_store.materialize(doc_id, doc.text, self.chunker.spec, self.chunker)
            # Convert to Chunk objects
            return [Chunk(chunk_id=i, text=chunk, doc_id=doc_id) for i, chunk in enumerate(chunks)]
        else:
            # Just read existing chunks
            chunks = self.chunk_store.read(doc_id, self.chunker.spec)
            if chunks is None:
                return None
            # Convert to Chunk objects
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
        
        chunk = next((x for x in chunks if x.chunk_id == chunk_number), None)
        
        if chunk is None:
            return None
        
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


    def get_embeddings(self, doc_id: str, materialize_if_necessary: bool = False) -> Optional[np.ndarray]:
        """
        Get embeddings for a document.
        
        Args:
            doc_id: Document ID
            materialize_if_necessary: If True, materialize embeddings if they don't exist
            
        Returns:
            Embeddings array if found, None otherwise
        """
        if materialize_if_necessary:
            chunks = self.get_chunks(doc_id, materialize_if_necessary=True)
            # Extract text from Chunk objects for embedding
            chunk_texts = [chunk.text for chunk in chunks]
            return self.embedding_store.materialize(doc_id, chunk_texts, self.chunker_spec, self.embedder)
        else:
            # Just read existing embeddings
            return self.embedding_store.read(doc_id, self.chunker.spec, self.embedder.spec)

    # Compatibility helpers
    def chunks(self, doc_id: str):
        return self.get_chunks(doc_id, materialize_if_necessary=True)

    def embeddings(self, doc_id: str):
        return self.get_embeddings(doc_id, materialize_if_necessary=True)

    def build_chunks(self, max_documents: Optional[int] = None, doc_ids: Optional[List[str]] = None) -> None:
        """Build chunks for documents in the corpus.
        
        Args:
            max_documents: Maximum number of documents to process (None for all)
            doc_ids: Specific document IDs to process (None for all documents)
        """
        if doc_ids is not None:
            # Build chunks for specific documents
            total_documents = len(doc_ids)
            print(f"Building chunks for {total_documents} specific documents...")
            with tqdm(total=total_documents, desc="Building chunks", unit="doc") as pbar:
                for doc_id in doc_ids:
                    _ = self.get_chunks(doc_id, materialize_if_necessary=True)
                    pbar.update(1)
        else:
            # Build chunks for all documents (existing logic)
            documents_iter = self.corpus.iter_documents()
            
            # Count total documents for progress bar (this is the only place we need to materialize)
            if max_documents is None:
                total_documents = self.corpus.get_document_count()
            else:
                total_documents = max_documents
            
            print(f"Building chunks for {total_documents} documents...")
            with tqdm(total=total_documents, desc="Building chunks", unit="doc") as pbar:
                doc_count = 0
                for doc in documents_iter:
                    if max_documents and doc_count >= max_documents:
                        break
                        
                    _ = self.get_chunks(doc.doc_id, materialize_if_necessary=True)
                    pbar.update(1)
                    doc_count += 1
    
    def build_embeddings(self, max_documents: Optional[int] = None, doc_ids: Optional[List[str]] = None) -> None:
        """Build embeddings for documents in the corpus (assumes chunks exist).
        
        Args:
            max_documents: Maximum number of documents to process (None for all)
            doc_ids: Specific document IDs to process (None for all documents)
        """
        if doc_ids is not None:
            # Build embeddings for specific documents
            total_documents = len(doc_ids)
            print(f"Building embeddings for {total_documents} specific documents...")
            with tqdm(total=total_documents, desc="Building embeddings", unit="doc") as pbar:
                for doc_id in doc_ids:
                    _ = self.get_embeddings(doc_id, materialize_if_necessary=True)
                    pbar.update(1)
        else:
            # Build embeddings for all documents (existing logic)
            documents_iter = self.corpus.iter_documents()
            
            # Count total documents for progress bar (this is the only place we need to materialize)
            if max_documents is None:
                total_documents = self.corpus.get_document_count()
            else:
                total_documents = max_documents
            
            print(f"Building embeddings for {total_documents} documents...")
            with tqdm(total=total_documents, desc="Building embeddings", unit="doc") as pbar:
                doc_count = 0
                for doc in documents_iter:
                    if max_documents and doc_count >= max_documents:
                        break
                        
                    _ = self.get_embeddings(doc.doc_id, materialize_if_necessary=True)
                    pbar.update(1)
                    doc_count += 1
    
    def build_all(self, reindex: bool = True, max_documents: Optional[int] = None, doc_ids: Optional[List[str]] = None) -> None:
        """
        Build chunks, embeddings, and optionally FAISS index for documents.
        
        Args:
            reindex: Whether to build FAISS index after embeddings
            max_documents: Maximum number of documents to process (None for all)
            doc_ids: Specific document IDs to process (None for all documents)
        """
        self.build_chunks(max_documents=max_documents, doc_ids=doc_ids)
        self.build_embeddings(max_documents=max_documents, doc_ids=doc_ids)
        
        if reindex:
            self.build_index()
    
    def build_index(self) -> bool:
        """Build FAISS index for fast retrieval"""
        # Collect all embeddings for the specific chunker and embedder specs
        embeddings_data = []
        total_chunks = 0
        
        for doc_id, embedding_array in self.iter_embeddings():
            if embedding_array.size == 0:
                continue
                
            # Handle both 1D and 2D arrays
            if embedding_array.ndim == 1:
                # Single chunk - create chunk_0 ID
                chunk_id = f"{doc_id}_chunk_0"
                embeddings_data.append((chunk_id, embedding_array))
                total_chunks += 1
            else:
                # Multiple chunks - create separate IDs for each
                for chunk_idx, chunk_embedding in enumerate(embedding_array):
                    chunk_id = f"{doc_id}_chunk_{chunk_idx}"
                    embeddings_data.append((chunk_id, chunk_embedding))
                    total_chunks += 1
        
        if not embeddings_data:
            print("No embeddings found for index building")
            return False
        
        print(f"Building index with {len(embeddings_data)} chunk-level embeddings from {len(list(self.iter_embeddings()))} documents...")
        
        # Build index using instance index store
        success = self.index_store.build_index(
            chunker_spec=self.chunker.spec,
            embedder_spec=self.embedder.spec,
            embeddings_data=embeddings_data
        )
        
        if success:
            print(f"Successfully built index with {total_chunks} total chunks")
        else:
            print(f"Failed to build index")
        
        return success
    
    def get_corpus_info(self) -> dict:
        """Get information about the embedded corpus"""
        return {
            "corpus_path": str(self.corpus_path),
            "workspace_path": str(self.workspace_path),
            "document_count": self.corpus.get_document_count(),
            "chunker_spec": self.chunker.spec.name,
            "embedder_spec": self.embedder.spec.model_name
        }
    
    def get_stats(self) -> dict:
        """Get detailed statistics about the embedded corpus"""
        stats = {
            "corpus_info": self.get_corpus_info(),
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
            index_exists = self.index_store.has_index(self.chunker.spec, self.embedder.spec)
        except Exception as e:
            index_exists = f"Error: {e}"
        
        stats["storage_stats"] = {
            "chunks_available": chunk_count,
            "embeddings_available": embedding_count,
            "index_exists": index_exists
        }
        
        return stats
    
    def show_stats(self) -> None:
        """Print detailed statistics about the embedded corpus"""
        stats = self.get_stats()
        print("Embedded Corpus Statistics:")
        print("=" * 40)
        
        # Print corpus info
        print("Corpus Information:")
        for key, value in stats["corpus_info"].items():
            print(f"  {key}: {value}")
        
        print("\nStorage Statistics:")
        for key, value in stats["storage_stats"].items():
            print(f"  {key}: {value}")
    
    def _fast_count_chunks(self) -> int:
        """Fast count of available chunks using file system"""
        chunks_dir = self.layout.chunks_dir(self.chunker.spec)
        if not chunks_dir.exists():
            return 0
        
        # Count .chunks.jsonl files in chunks directory
        count = 0
        for chunk_file in chunks_dir.glob("*.chunks.jsonl"):
            count += 1
        return count
    
    def _fast_count_embeddings(self) -> int:
        """Fast count of available embeddings using file system"""
        embeddings_dir = self.layout.embeddings_dir(self.chunker.spec, self.embedder.spec)
        if not embeddings_dir.exists():
            return 0
        
        # Count .npy files in embeddings directory
        count = 0
        for embedding_file in embeddings_dir.glob("*.npy"):
            count += 1
        return count
    
    def iter_embeddings(self):
        """
        Iterate through existing embeddings in the corpus for the instance's chunker and embedder.
        
        Yields:
            Tuple[str, np.ndarray]: (item_id, embedding_vector) pairs
        """
        # Get the embeddings directory for this instance's specs
        embeddings_dir = self.layout.embeddings_dir(self.chunker.spec, self.embedder.spec)
        if not embeddings_dir.exists():
            return
        
        # Iterate through existing .npy files
        for embedding_file in embeddings_dir.glob("*.npy"):
            try:
                # Extract doc_id from filename (remove .npy extension)
                doc_id = embedding_file.stem
                
                # Load the embedding
                embedding = np.load(embedding_file)
                
                if embedding.size > 0:
                    yield (doc_id, embedding)
                    
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Error loading embedding from {embedding_file}: {e}")
                continue


