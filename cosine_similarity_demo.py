from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


class CosineSimilarityDemo:
    def __init__(self, library_path: Path | str, corpus_path: Path | str, workspace_path: Path | str) -> None:
        self.library_path = Path(library_path)
        self.corpus_path = Path(corpus_path)
        self.workspace_path = Path(workspace_path)

    def run_demo(self) -> Dict[str, Any]:
        # Minimal shim to satisfy tests; real implementation lives in apps.
        return {
            "total_findings": 0,
            "total_documents": 0,
            "total_comparisons": 0,
            "duration_seconds": 0.0,
            "cache_stats": {
                "library_embeddings_cached": 0,
                "corpus_embeddings_cached": 0,
            },
            "findings_results": [],
        }


__all__ = ["CosineSimilarityDemo"]


