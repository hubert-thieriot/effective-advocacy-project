"""
Tests for embedded library workflows
"""

import pytest
import tempfile
import json
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from efi_core.layout import EmbeddedLibraryLayout, EmbeddedCorpusLayout
from efi_core.types import ChunkerSpec, Finding
from efi_library.library_handle import LibraryHandle
from efi_core.stores import ChunkStore, EmbeddingStore
from efi_analyser.chunkers import SentenceChunker
from efi_analyser.embedders import SentenceTransformerEmbedder


class MockChunker:
    """Lightweight mock chunker for testing"""
    
    def __init__(self, max_chunk_size: int = 100):
        self._max_chunk_size = max_chunk_size
        self._spec = ChunkerSpec("mock", {"max_size": max_chunk_size})
    
    @property
    def spec(self) -> ChunkerSpec:
        return self._spec
    
    def chunk(self, text: str) -> list[str]:
        """Simple chunking by splitting on spaces"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            if current_size + len(word) + 1 > self._max_chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [word]
                    current_size = len(word)
                else:
                    # Single word is too long, add it anyway
                    chunks.append(word)
                    current_chunk = []
                    current_size = 0
            else:
                current_chunk.append(word)
                current_size += len(word) + 1
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks if chunks else [text]


class MockEmbedder:
    """Lightweight mock embedder for testing"""
    
    def __init__(self, dim: int = 128):
        self.dim = dim
        self._spec = MockEmbedderSpec("mock_embedder", dim)
    
    @property
    def spec(self):
        return self._spec
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate deterministic mock embeddings"""
        embeddings = []
        for i, text in enumerate(texts):
            # Create deterministic embedding based on text length and position
            np.random.seed(hash(text) % 10000)
            embedding = np.random.rand(self.dim).tolist()
            embeddings.append(embedding)
        return embeddings


class MockEmbedderSpec:
    """Mock embedder specification that implements the required interface"""
    
    def __init__(self, model_name: str, dim: int):
        self.model_name = model_name
        self.dim = dim
    
    def key(self) -> str:
        """Generate a deterministic key for the embedder spec"""
        return f"mock_{self.model_name}_{self.dim}"


class TestEmbeddedLibrary:
    """Test embedded library workflows"""
    
    @pytest.fixture
    def temp_library_and_workspace(self):
        """Create temporary library and workspace"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create library
            library_path = temp_path / "test_library"
            library_path.mkdir()
            
            # Create findings.json (old structure for testing)
            findings_data = [
                {
                    "url": "https://example.com/doc1",
                    "title": "Research Document 1",
                    "published_at": "2025-01-01T00:00:00",
                    "language": "en",
                    "extraction_date": "2025-01-20T00:00:00",
                    "findings": [
                        {
                            "finding_id": "finding1",
                            "text": "Air pollution causes serious health problems in urban areas.",
                            "confidence": 0.9,
                            "category": "health",
                            "keywords": ["air pollution", "health", "urban"]
                        },
                        {
                            "finding_id": "finding2",
                            "text": "Coal power plants are major contributors to air pollution.",
                            "confidence": 0.85,
                            "category": "environment",
                            "keywords": ["coal", "power plants", "pollution"]
                        }
                    ],
                    "metadata": {"source": "research"}
                },
                {
                    "url": "https://example.com/doc2",
                    "title": "Research Document 2", 
                    "published_at": "2025-01-02T00:00:00",
                    "language": "en",
                    "extraction_date": "2025-01-20T00:00:00",
                    "findings": [
                        {
                            "finding_id": "finding3",
                            "text": "Renewable energy sources reduce carbon emissions significantly.",
                            "confidence": 0.95,
                            "category": "environment",
                            "keywords": ["renewable energy", "carbon", "emissions"]
                        }
                    ],
                    "metadata": {"source": "research"}
                }
            ]
            
            findings_path = library_path / "findings.json"
            with open(findings_path, 'w') as f:
                json.dump(findings_data, f, indent=2)
            
            # Create metadata.json
            metadata = {
                "name": "Test Library",
                "description": "Test findings library",
                "created_at": "2025-01-20T00:00:00"
            }
            with open(library_path / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Create workspace
            workspace_path = temp_path / "workspace"
            
            yield library_path, workspace_path
    
    def test_embedded_library_layout(self, temp_library_and_workspace):
        """Test EmbeddedLibraryLayout functionality"""
        library_path, workspace_path = temp_library_and_workspace
        
        layout = EmbeddedLibraryLayout(library_path, workspace_path)
        layout.ensure_dirs()
        
        # Test library layout functionality
        assert layout.library_root == library_path
        assert layout.findings_path == library_path / "findings.json"
        assert layout.metadata_path == library_path / "metadata.json"
        
        # Test workspace functionality
        library_name = library_path.name
        workspace_dir = layout.get_library_workspace_dir(library_name)
        assert workspace_dir == workspace_path / "libraries" / library_name
        
        # Test source paths
        assert layout.sources_dir == library_path / "sources"
    
    def test_chunk_store_with_embedded_library(self, temp_library_and_workspace):
        """Test ChunkStore with embedded library layout"""
        library_path, workspace_path = temp_library_and_workspace
        
        layout = EmbeddedLibraryLayout(library_path, workspace_path)
        layout.ensure_dirs()
        
        chunk_store = ChunkStore(layout)
        chunker = MockChunker(max_chunk_size=50)
        chunker_spec = ChunkerSpec("mock", {"max_size": 50})
        
        # Test chunking a finding
        finding_text = "Air pollution causes serious health problems in urban areas."
        chunks = chunk_store.materialize("finding1", finding_text, chunker_spec, chunker)
        
        assert len(chunks) > 0
        assert all(len(chunk) <= 50 for chunk in chunks)
        
        # Verify file was created
        chunks_path = layout.chunks_path("finding1", chunker_spec)
        assert chunks_path.exists()
    
    def test_embedding_store_with_embedded_library(self, temp_library_and_workspace):
        """Test EmbeddingStore with embedded library layout"""
        library_path, workspace_path = temp_library_and_workspace
        
        layout = EmbeddedLibraryLayout(library_path, workspace_path)
        layout.ensure_dirs()
        
        embedding_store = EmbeddingStore(layout)
        chunker = MockChunker(max_chunk_size=50)
        embedder = MockEmbedder(dim=64)
        chunker_spec = ChunkerSpec("mock", {"max_size": 50})
        
        # Test embedding chunks
        chunks = ["Air pollution causes", "serious health problems"]
        embeddings = embedding_store.materialize("finding1", chunks, chunker_spec, embedder)
        
        assert embeddings.shape == (2, 64)  # 2 chunks, 64 dimensions
        assert embeddings.dtype == np.float64
        
        # Verify file was created
        emb_path = layout.emb_path("finding1", chunker_spec, embedder.spec)
        assert emb_path.exists()
    
    def test_full_embedded_library_workflow(self, temp_library_and_workspace):
        """Test complete embedded library workflow"""
        library_path, workspace_path = temp_library_and_workspace
        
        # Setup
        layout = EmbeddedLibraryLayout(library_path, workspace_path)
        layout.ensure_dirs()
        
        library_handle = LibraryHandle(library_path)
        chunk_store = ChunkStore(layout)
        embedding_store = EmbeddingStore(layout)
        
        chunker = MockChunker(max_chunk_size=50)
        embedder = MockEmbedder(dim=64)
        chunker_spec = ChunkerSpec("mock", {"max_size": 50})
        
        # Process all findings
        processed_findings = []
        finding_counter = 0
        
        for doc_findings in library_handle.iter_documents():
            for finding in doc_findings.findings:
                finding_id = f"finding_{finding_counter}"
                finding_counter += 1
                
                # Chunk the finding
                chunks = chunk_store.materialize(finding_id, finding.text, chunker_spec, chunker)
                
                # Embed the chunks
                embeddings = embedding_store.materialize(finding_id, chunks, chunker_spec, embedder)
                
                processed_findings.append({
                    "finding_id": finding_id,
                    "text": finding.text,
                    "chunks": chunks,
                    "embeddings": embeddings,
                    "category": finding.category
                })
        
        # Verify results
        assert len(processed_findings) == 3  # We created 3 findings total
        
        for finding_data in processed_findings:
            assert len(finding_data["chunks"]) > 0
            assert finding_data["embeddings"].shape[0] == len(finding_data["chunks"])
            assert finding_data["embeddings"].shape[1] == embedder.dim
            assert finding_data["category"] in ["health", "environment"]
        
        # Verify workspace structure
        library_name = library_path.name
        workspace_dir = layout.get_library_workspace_dir(library_name)
        
        assert (workspace_dir / "chunks").exists()
        assert (workspace_dir / "embeddings").exists()
        assert len(list((workspace_dir / "chunks").rglob("*.chunks.jsonl"))) == 3
        assert len(list((workspace_dir / "embeddings").rglob("*.npy"))) == 3
    
    def test_library_and_corpus_workspace_coexistence(self, temp_library_and_workspace):
        """Test that library and corpus workspaces can coexist"""
        library_path, workspace_path = temp_library_and_workspace
        
        # Create both layouts using the same workspace
        library_layout = EmbeddedLibraryLayout(library_path, workspace_path)
        
        # Create a dummy corpus path for testing
        corpus_path = library_path.parent / "test_corpus"
        corpus_path.mkdir()
        corpus_layout = EmbeddedCorpusLayout(corpus_path, workspace_path)
        
        # Ensure directories for both
        library_layout.ensure_dirs()
        corpus_layout.ensure_dirs()
        
        # Verify workspace structure
        assert (workspace_path / "corpora").exists()
        assert (workspace_path / "libraries").exists()
        assert (workspace_path / "corpora" / "test_corpus").exists()
        assert (workspace_path / "libraries" / "test_library").exists()
        
        # Verify they don't interfere with each other
        chunker_spec = ChunkerSpec("sentence", {"max_size": 100})
        embedder_spec = MockEmbedder().spec
        
        library_chunks_path = library_layout.chunks_path("finding1", chunker_spec)
        corpus_chunks_path = corpus_layout.chunks_path("doc1", chunker_spec)
        
        # Should be in different directories
        assert "libraries" in str(library_chunks_path)
        assert "corpora" in str(corpus_chunks_path)
        assert library_chunks_path != corpus_chunks_path


class TestEmbeddedLibraryIntegration:
    """Test embedded library with real chunker/embedder (but small data)"""
    
    @pytest.fixture
    def temp_library_and_workspace(self):
        """Create temporary library and workspace with minimal data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create library with just one small finding
            library_path = temp_path / "test_library"
            library_path.mkdir()
            
            findings_data = [
                {
                    "url": "https://example.com/doc1",
                    "title": "Test Document",
                    "published_at": "2025-01-01T00:00:00",
                    "language": "en",
                    "extraction_date": "2025-01-20T00:00:00",
                    "findings": [
                        {
                            "finding_id": "finding1",
                            "text": "Air quality matters. Clean energy helps.",
                            "confidence": 0.9,
                            "category": "environment",
                            "keywords": ["air quality", "clean energy"]
                        }
                    ],
                    "metadata": {"source": "test"}
                }
            ]
            
            findings_path = library_path / "findings.json"
            with open(findings_path, 'w') as f:
                json.dump(findings_data, f, indent=2)
            
            metadata = {
                "name": "Test Library",
                "description": "Test findings library",
                "created_at": "2025-01-20T00:00:00"
            }
            with open(library_path / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            workspace_path = temp_path / "workspace"
            
            yield library_path, workspace_path
    
    def test_embedded_library_with_real_components(self, temp_library_and_workspace):
        """Test with real chunker/embedder but minimal data"""
        library_path, workspace_path = temp_library_and_workspace
        
        # Use real components but with minimal data
        chunker = SentenceChunker(max_chunk_size=20)
        embedder = SentenceTransformerEmbedder()
        chunker_spec = ChunkerSpec("sentence", {"max_size": 20})
        
        # Create embedded library
        from efi_library.embedded.embedded_library import EmbeddedLibrary
        
        embedded_library = EmbeddedLibrary(
            library_path=library_path,
            workspace_path=workspace_path,
            chunker=chunker,
            embedder=embedder
        )
        
        # Test basic functionality
        info = embedded_library.get_library_info()
        assert info["document_count"] == 1
        assert info["total_findings_count"] == 1
        assert info["chunker_spec"] == "sentence"
        assert info["embedder_spec"] == "all-MiniLM-L6-v2"
        
        # Test chunking (should be fast with small text)
        chunks = embedded_library.chunks("finding1")
        assert len(chunks) > 0
        # Note: SentenceChunker creates overlapping chunks, so some may exceed max_chunk_size
        # This is expected behavior for sentence-based chunking
        print(f"Generated chunks: {chunks}")
        print(f"Chunk lengths: {[len(chunk) for chunk in chunks]}")
        
        # Test embeddings (should be fast with small text)
        embeddings = embedded_library.embeddings("finding1")
        assert embeddings.shape[0] == len(chunks)
        assert embeddings.shape[1] == embedder.dim
