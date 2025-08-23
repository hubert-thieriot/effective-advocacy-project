import json
from pathlib import Path
from typing import List, Protocol, Optional

from efi_core.types import ChunkerSpec
from efi_core.protocols import Chunker


class ChunkStorageLayout(Protocol):
    """Protocol for layouts that support chunk storage"""
    def chunks_path(self, item_id: str, chunker: ChunkerSpec) -> Path: ...


class ChunkStore:
    """Generic chunk store that works with any layout supporting chunk storage"""
    
    def __init__(self, layout: ChunkStorageLayout):
        self.layout = layout

    def read(self, item_id: str, chunker: ChunkerSpec) -> Optional[List[str]]:
        """
        Read existing chunks for a text item.
        
        Args:
            item_id: Unique identifier for the item
            chunker: Chunker specification
            
        Returns:
            List of text chunks if they exist, None otherwise
        """
        path = self.layout.chunks_path(item_id, chunker)
        if not path.exists():
            return None
            
        chunks: List[str] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            chunks.append(json.loads(line)["text"])
        return chunks

    def materialize(self, item_id: str, text: str, chunker: ChunkerSpec, chunker_impl: Chunker) -> List[str]:
        """
        Get or create chunks for a text item (document, finding, etc.)
        
        Args:
            item_id: Unique identifier for the item being chunked
            text: Text content to chunk
            chunker: Chunker specification
            chunker_impl: Chunker implementation
            
        Returns:
            List of text chunks
        """
        path = self.layout.chunks_path(item_id, chunker)
        if path.exists():
            chunks: List[str] = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                chunks.append(json.loads(line)["text"])
            return chunks
        
        path.parent.mkdir(parents=True, exist_ok=True)
        chunks = chunker_impl.chunk(text)
        with open(path, "w", encoding="utf-8") as f:
            for i, ch in enumerate(chunks):
                f.write(json.dumps({"chunk_id": i, "text": ch}, ensure_ascii=False) + "\n")
        return chunks


