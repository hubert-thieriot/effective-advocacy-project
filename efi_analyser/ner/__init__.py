"""Named Entity Recognition module for discourse analysis."""

from .extractor import StanzaNERExtractor
from .types import EntityMention, ChunkEntities, DocumentEntities, NERResult

__all__ = [
    "StanzaNERExtractor",
    "EntityMention",
    "ChunkEntities",
    "DocumentEntities",
    "NERResult",
]
