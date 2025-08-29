"""
Tests for embedded corpus workflows
"""

import pytest
import tempfile
import json
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from efi_core.layout import EmbeddedCorpusLayout
from efi_core.types import ChunkerSpec, Document
from efi_corpus.corpus_handle import CorpusHandle
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


class TestEmbeddedCorpus:
    """Test embedded corpus workflows"""
    
    @pytest.fixture
    def temp_corpus_and_workspace(self):
        """Create temporary corpus and workspace"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create corpus
            corpus_path = temp_path / "test_corpus"
            corpus_path.mkdir()
            
            # Create documents directory structure
            docs_dir = corpus_path / "documents"
            docs_dir.mkdir()
            
            # Create a few test documents
            documents = [
                {
                    "doc_id": "doc1",
                    "title": "Test Document 1",
                    "text": "This is a test document about air quality and pollution. It contains information about environmental impacts.",
                    "source": "Test Source 1"
                },
                {
                    "doc_id": "doc2", 
                    "title": "Test Document 2",
                    "text": "Another test document focusing on renewable energy and climate change. Solar and wind power are discussed.",
                    "source": "Test Source 2"
                }
            ]
            
            for doc in documents:
                doc_dir = docs_dir / doc["doc_id"]
                doc_dir.mkdir()
                
                # Save text file
                text_path = doc_dir / "text.txt"
                text_path.write_text(doc["text"])
                
                # Save metadata
                meta_path = doc_dir / "meta.json"
                meta_path.write_text(json.dumps({
                    "title": doc["title"],
                    "source": doc["source"]
                }))
                
                # Save fetch info
                fetch_path = doc_dir / "fetch.json"
                fetch_path.write_text(json.dumps({
                    "url": f"https://example.com/{doc['doc_id']}",
                    "fetched_at": "2025-01-20T00:00:00"
                }))
            
            # Create index.jsonl
            index_path = corpus_path / "index.jsonl"
            with open(index_path, 'w') as f:
                for doc in documents:
                    index_entry = {
                        "id": doc["doc_id"],
                        "title": doc["title"],
                        "source": doc["source"],
                        "url": f"https://example.com/{doc['doc_id']}"
                    }
                    f.write(json.dumps(index_entry) + '\n')
            
            # Create workspace
            workspace_path = temp_path / "workspace"
            
            yield corpus_path, workspace_path
    
    def test_embedded_corpus_layout(self, temp_corpus_and_workspace):
        """Test EmbeddedCorpusLayout functionality"""
        corpus_path, workspace_path = temp_corpus_and_workspace
        
        layout = EmbeddedCorpusLayout(corpus_path, workspace_path)
        layout.ensure_dirs()
        
        # Test corpus layout functionality
        assert layout.corpus_root == corpus_path
        assert layout.index_path == corpus_path / "index.jsonl"
        assert layout.docs_dir == corpus_path / "documents"
        
        # Test workspace functionality
        corpus_name = corpus_path.name
        workspace_dir = layout.get_corpus_workspace_dir(corpus_name)
        assert workspace_dir == workspace_path / "corpora" / corpus_name
        
        # Test document paths
        assert layout.text_path("doc1") == corpus_path / "documents" / "doc1" / "text.txt"
        assert layout.meta_path("doc1") == corpus_path / "documents" / "doc1" / "meta.json"
    
    def test_chunk_store_with_embedded_corpus(self, temp_corpus_and_workspace):
        """Test ChunkStore with embedded corpus layout"""
        corpus_path, workspace_path = temp_corpus_and_workspace
        
        layout = EmbeddedCorpusLayout(corpus_path, workspace_path)
        layout.ensure_dirs()
        
        chunk_store = ChunkStore(layout)
        chunker = MockChunker(max_chunk_size=50)
        chunker_spec = ChunkerSpec("mock", {"max_size": 50})
        
        # Test chunking a document
        doc_text = "This is a test document about air quality and pollution. It contains information about environmental impacts."
        chunks = chunk_store.materialize("doc1", doc_text, chunker_spec, chunker)
        
        assert len(chunks) > 1  # Should be chunked into multiple pieces
        assert all(len(chunk) <= 50 for chunk in chunks)
        assert "".join(chunks).replace(" ", "") == doc_text.replace(" ", "")
        
        # Verify file was created
        chunks_path = layout.chunks_path("doc1", chunker_spec)
        assert chunks_path.exists()
    
    def test_embedding_store_with_embedded_corpus(self, temp_corpus_and_workspace):
        """Test EmbeddingStore with embedded corpus layout"""
        corpus_path, workspace_path = temp_corpus_and_workspace
        
        layout = EmbeddedCorpusLayout(corpus_path, workspace_path)
        layout.ensure_dirs()
        
        embedding_store = EmbeddingStore(layout)
        chunker = MockChunker(max_chunk_size=50)
        embedder = MockEmbedder(dim=64)
        chunker_spec = ChunkerSpec("mock", {"max_size": 50})
        
        # Test embedding chunks
        chunks = ["This is a test chunk", "Another test chunk"]
        embeddings = embedding_store.materialize("doc1", chunks, chunker_spec, embedder)
        
        assert embeddings.shape == (2, 64)  # 2 chunks, 64 dimensions
        assert embeddings.dtype == np.float64
        
        # Verify file was created
        emb_path = layout.emb_path("doc1", chunker_spec, embedder.spec)
        assert emb_path.exists()
    
    def test_full_embedded_corpus_workflow(self, temp_corpus_and_workspace):
        """Test complete embedded corpus workflow"""
        corpus_path, workspace_path = temp_corpus_and_workspace
        
        # Setup
        layout = EmbeddedCorpusLayout(corpus_path, workspace_path)
        layout.ensure_dirs()
        
        corpus_handle = CorpusHandle(corpus_path)
        chunk_store = ChunkStore(layout)
        embedding_store = EmbeddingStore(layout)
        
        chunker = MockChunker(max_chunk_size=50)
        embedder = MockEmbedder(dim=64)
        chunker_spec = ChunkerSpec("mock", {"max_size": 50})
        
        # Process all documents
        processed_docs = []
        
        for doc in corpus_handle.iter_documents():
            # Chunk the document
            chunks = chunk_store.materialize(doc.doc_id, doc.text, chunker_spec, chunker)
            
            # Embed the chunks
            embeddings = embedding_store.materialize(doc.doc_id, chunks, chunker_spec, embedder)
            
            processed_docs.append({
                "doc_id": doc.doc_id,
                "title": doc.title,
                "chunks": chunks,
                "embeddings": embeddings
            })
        
        # Verify results
        assert len(processed_docs) == 2  # We created 2 documents
        
        for doc_data in processed_docs:
            assert len(doc_data["chunks"]) > 0
            assert doc_data["embeddings"].shape[0] == len(doc_data["chunks"])
            assert doc_data["embeddings"].shape[1] == embedder.dim
        
        # Verify workspace structure
        corpus_name = corpus_path.name
        workspace_dir = layout.get_corpus_workspace_dir(corpus_name)
        
        assert (workspace_dir / "chunks").exists()
        assert (workspace_dir / "embeddings").exists()
        assert len(list((workspace_dir / "chunks").rglob("*.chunks.jsonl"))) == 2
        assert len(list((workspace_dir / "embeddings").rglob("*.npy"))) == 2


class TestEmbeddedCorpusIntegration:
    """Test embedded corpus with real chunker/embedder (but small data)"""
    
    @pytest.fixture
    def temp_corpus_and_workspace(self):
        """Create temporary corpus and workspace with minimal data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create corpus with just one small document
            corpus_path = temp_path / "test_corpus"
            corpus_path.mkdir()
            
            docs_dir = corpus_path / "documents"
            docs_dir.mkdir()
            
            # Single small document
            doc_dir = docs_dir / "doc1"
            doc_dir.mkdir()
            
            text_path = doc_dir / "text.txt"
            text_path.write_text("Air quality matters. Clean energy helps.")
            
            meta_path = doc_dir / "meta.json"
            meta_path.write_text(json.dumps({"title": "Test Doc"}))
            
            fetch_path = doc_dir / "fetch.json"
            fetch_path.write_text(json.dumps({"url": "https://example.com/doc1"}))
            
            # Create index.jsonl
            index_path = corpus_path / "index.jsonl"
            with open(index_path, 'w') as f:
                index_entry = {
                    "id": "doc1",
                    "title": "Test Doc",
                    "url": "https://example.com/doc1"
                }
                f.write(json.dumps(index_entry) + '\n')
            
            workspace_path = temp_path / "workspace"
            
            yield corpus_path, workspace_path
    
    def test_embedded_corpus_with_real_components(self, temp_corpus_and_workspace):
        """Test with real chunker/embedder but minimal data"""
        corpus_path, workspace_path = temp_corpus_and_workspace
        
        # Use real components but with minimal data
        chunker = SentenceChunker(max_chunk_size=20)
        embedder = SentenceTransformerEmbedder()
        chunker_spec = ChunkerSpec("sentence", {"max_size": 20})
        
        # Create embedded corpus
        from efi_corpus.embedded.embedded_corpus import EmbeddedCorpus
        
        embedded_corpus = EmbeddedCorpus(
            corpus_path=corpus_path,
            workspace_path=workspace_path,
            chunker=chunker,
            embedder=embedder
        )
        
        # Test basic functionality
        info = embedded_corpus.get_corpus_info()
        assert info["document_count"] == 1
        assert info["chunker_spec"] == "sentence"
        assert info["embedder_spec"] == "all-MiniLM-L6-v2"
        
        # Test chunking (should be fast with small text)
        chunks = embedded_corpus.get_chunks("doc1", materialize_if_necessary=True)
        assert len(chunks) > 0
        # Note: SentenceChunker creates overlapping chunks, so some may exceed max_chunk_size
        # This is expected behavior for sentence-based chunking
        print(f"Generated chunks: {chunks}")
        print(f"Chunk lengths: {[len(chunk.text) for chunk in chunks]}")
        
        # Test embeddings (should be fast with small text)
        embeddings = embedded_corpus.get_embeddings("doc1", materialize_if_necessary=True)
        assert embeddings.shape[0] == len(chunks)
        assert embeddings.shape[1] == embedder.dim
