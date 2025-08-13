"""
CorpusHandle - Manages file I/O operations within a corpus folder
"""

import json
import zstandard as zstd
from pathlib import Path
from typing import Dict, Any


class CorpusHandle:
    """Responsible for all file I/O inside a corpus folder"""
    
    def __init__(self, corpus_dir: Path):
        self.dir = corpus_dir
        (self.dir / "documents").mkdir(parents=True, exist_ok=True)
        (self.dir / "index.jsonl").touch(exist_ok=True)
        (self.dir / "manifest.json").touch(exist_ok=True)

    def has_doc(self, stable_id: str) -> bool:
        """Check if a document with the given stable_id exists"""
        return (self.dir / "documents" / stable_id[:2] / stable_id).exists()

    def _doc_dir(self, stable_id: str) -> Path:
        """Get the document directory path, creating it if necessary"""
        d = self.dir / "documents" / stable_id[:2] / stable_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_document(self, *, stable_id: str, meta: Dict[str, Any], text: str,
                       raw_bytes: bytes, raw_ext: str, fetch_info: Dict[str, Any]):
        """Write a document to the corpus"""
        d = self._doc_dir(stable_id)
        
        # meta.json
        (d / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), 
            encoding="utf-8"
        )

        # text.txt (plain text, no compression)
        (d / "text.txt").write_text(text, encoding="utf-8")

        # raw file (copy from cache)
        # Use the raw_ext as provided - could be html, pdf, txt, etc.
        (d / f"raw.{raw_ext}").write_bytes(raw_bytes)

        # fetch.json
        fetch_payload = {**fetch_info}
        (d / "fetch.json").write_text(
            json.dumps(fetch_payload, indent=2), 
            encoding="utf-8"
        )

    def append_index(self, row: Dict[str, Any]):
        """Append a row to the index.jsonl file"""
        with open(self.dir / "index.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def load_manifest(self) -> Dict[str, Any]:
        """Load the manifest.json file"""
        try:
            content = (self.dir / "manifest.json").read_text(encoding="utf-8")
            return json.loads(content) if content.strip() else {}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save_manifest(self, manifest: Dict[str, Any]):
        """Save the manifest.json file"""
        (self.dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), 
            encoding="utf-8"
        )
