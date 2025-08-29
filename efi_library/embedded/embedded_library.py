from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import numpy as np

from tqdm import tqdm

from efi_core.types import ChunkerSpec, EmbedderSpec, Chunk
from efi_core.protocols import Chunker, Embedder
from efi_core.layout import EmbeddedLibraryLayout
from efi_core.stores import ChunkStore, EmbeddingStore, IndexStore

from efi_library import LibraryHandle


@dataclass
class EmbeddedLibrary:
    library_path: Path
    workspace_path: Path
    chunker: Chunker
    embedder: Embedder

    def __post_init__(self) -> None:
        self.library = LibraryHandle(self.library_path)
        self.layout = EmbeddedLibraryLayout(Path(self.library_path), Path(self.workspace_path))
        self.chunk_store = ChunkStore(self.layout)
        self.embedding_store = EmbeddingStore(self.layout)
        self.index_store = IndexStore(
            workspace_path=self.workspace_path,
            corpus_or_library_name=self.library.library_path.name,
            is_corpus=False
        )
    
    @property
    def chunker_spec(self) -> ChunkerSpec:
        """Get chunker specification from chunker instance."""
        return self.chunker.spec
    
    @property
    def embedder_spec(self) -> EmbedderSpec:
        """Get embedder specification from embedder instance."""
        return self.embedder.spec

    def get_chunks(self, finding_id: str, materialize_if_necessary: bool = False) -> Optional[List[str]]:
        """
        Get chunks for a finding.
        
        Args:
            finding_id: Finding ID
            materialize_if_necessary: If True, materialize chunks if they don't exist
            
        Returns:
            List of chunks if found, None otherwise
        """
        if materialize_if_necessary:
            finding = self.library.get_finding(finding_id)
            if not finding:
                raise FileNotFoundError(f"Finding not found: {finding_id}")
            return self.chunk_store.materialize(finding_id, finding.text, self.chunker_spec, self.chunker)
        else:
            # Just read existing chunks
            return self.chunk_store.read(finding_id, self.chunker.spec)

    # Thin wrapper for compatibility with older tests
    def chunks(self, finding_id: str) -> Optional[List[str]]:
        return self.get_chunks(finding_id, materialize_if_necessary=True)



    def get_embeddings(self, finding_id: str, materialize_if_necessary: bool = False) -> Optional[np.ndarray]:
        """
        Get embeddings for a finding.
        
        Args:
            finding_id: Finding ID
            materialize_if_necessary: If True, materialize embeddings if they don't exist
            
        Returns:
            Embeddings array if found, None otherwise
        """
        if materialize_if_necessary:
            chunks = self.get_chunks(finding_id, materialize_if_necessary=True)
            return self.embedding_store.materialize(finding_id, chunks, self.chunker_spec, self.embedder)
        else:
            # Just check if embeddings exist and return them
            if self.embedding_store.has_embeddings(finding_id, self.chunker_spec, self.embedder.spec):
                chunks = self.get_chunks(finding_id, materialize_if_necessary=False)
                if chunks:
                    return self.embedding_store.materialize(finding_id, chunks, self.chunker_spec, self.embedder)
            return None

    # Thin wrapper for compatibility with older tests
    def embeddings(self, finding_id: str) -> Optional[np.ndarray]:
        return self.get_embeddings(finding_id, materialize_if_necessary=True)
    


    def build_chunks(self, max_findings: Optional[int] = None) -> None:
        """
        Build chunks for findings.
        
        Args:
            max_findings: Maximum number of findings to process (None for all)
        """
        # Get total count for progress bar
        total_findings = self.library.get_findings_count()
        if max_findings:
            total_findings = min(total_findings, max_findings)
        
        print(f"Building chunks for {total_findings} findings...")
        with tqdm(total=total_findings, desc="Building chunks", unit="finding") as pbar:
            count = 0
            for doc_findings in self.library.iter_findings():
                if max_findings and count >= max_findings:
                    break
                for finding in doc_findings.findings:
                    if max_findings and count >= max_findings:
                        break
                    _ = self.get_chunks(finding.finding_id, materialize_if_necessary=True)
                    count += 1
                    pbar.update(1)
    
    def build_embeddings(self, max_findings: Optional[int] = None) -> None:
        """
        Build embeddings for findings.
        
        Args:
            max_findings: Maximum number of findings to process (None for all)
        """
        # Get total count for progress bar
        total_findings = self.library.get_findings_count()
        if max_findings:
            total_findings = min(total_findings, max_findings)
        
        print(f"Building embeddings for {total_findings} findings...")
        with tqdm(total=total_findings, desc="Building embeddings", unit="finding") as pbar:
            count = 0
            for doc_findings in self.library.iter_findings():
                if max_findings and count >= max_findings:
                    break
                for finding in doc_findings.findings:
                    if max_findings and count >= max_findings:
                        break
                    _ = self.get_embeddings(finding.finding_id, materialize_if_necessary=True)
                    count += 1
                    pbar.update(1)

    def build_all(self, reindex: bool = True, max_findings: Optional[int] = None) -> None:
        """
        Build chunks and embeddings for findings and optionally build FAISS index.
        
        Args:
            reindex: Whether to build FAISS index after embeddings
            max_findings: Maximum number of findings to process (None for all)
        """
        self.build_chunks(max_findings=max_findings)
        self.build_embeddings(max_findings=max_findings)
        
        if reindex:
            self.build_index()
    
    def build_index(self) -> bool:
        """Build FAISS index for fast retrieval"""
        # Collect all embeddings for the specific chunker and embedder specs
        embeddings_data = []
        total_chunks = 0
        
        for finding_id, embedding_array in self.iter_embeddings():
            if embedding_array.size == 0:
                continue
                
            # Handle both 1D and 2D arrays
            if embedding_array.ndim == 1:
                # Single chunk - create chunk_0 ID
                chunk_id = f"{finding_id}_chunk_0"
                embeddings_data.append((chunk_id, embedding_array))
                total_chunks += 1
            else:
                # Multiple chunks - create separate IDs for each
                for chunk_idx, chunk_embedding in enumerate(embedding_array):
                    chunk_id = f"{finding_id}_chunk_{chunk_idx}"
                    embeddings_data.append((chunk_id, chunk_embedding))
                    total_chunks += 1
        
        if not embeddings_data:
            print("No embeddings found for index building")
            return False
        
        print(f"Building index with {len(embeddings_data)} chunk-level embeddings from {len(list(self.iter_embeddings()))} findings...")
        
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
    
    def get_library_info(self) -> dict:
        """Get information about the embedded library"""
        return {
            "library_path": str(self.library_path),
            "workspace_path": str(self.workspace_path),
            "document_count": self.library.get_documents_count(),
            "total_findings_count": self.library.get_findings_count(),
            "chunker_spec": self.chunker_spec.name,
            "embedder_spec": self.embedder.spec.model_name
        }
    
    def get_stats(self) -> dict:
        """Get detailed statistics about the embedded library"""
        stats = {
            "library_info": self.get_library_info(),
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
        """Print detailed statistics about the embedded library"""
        stats = self.get_stats()
        print("Embedded Library Statistics:")
        print("=" * 40)
        
        # Print library info
        print("Library Information:")
        for key, value in stats["library_info"].items():
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
        Iterate through existing embeddings in the library for the instance's chunker and embedder.
        
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
                # Extract finding_id from filename (remove .npy extension)
                finding_id = embedding_file.stem
                
                # Load the embedding
                embedding = np.load(embedding_file)
                
                if embedding.size > 0:
                    yield (finding_id, embedding)
                    
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Error loading embedding from {embedding_file}: {e}")
                continue
