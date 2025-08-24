from pathlib import Path
from typing import Dict

from efi_core.types import Finding, LibraryDocumentWFindings
from .library_store import LibraryStore


class ExampleLibraryBuilder:
    """Build a simple findings library from local text files."""

    def __init__(self, raw_dir: Path, library_dir: Path):
        self.raw_dir = Path(raw_dir)
        self.library_dir = Path(library_dir)
        self.source = self.raw_dir.parent.name
        self.library_dir.parent.mkdir(parents=True, exist_ok=True)
        self.store = LibraryStore(self.library_dir.name, str(self.library_dir.parent))

    def build(self) -> Dict[str, int]:
        """Build the library from raw finding files."""
        added = 0
        for path in sorted(self.raw_dir.glob("*.txt")):
            text = path.read_text(encoding="utf-8").strip()
            url = f"https://example.com/library/{path.stem}"
            finding = Finding(
                finding_id=Finding.generate_id(url, 0),
                text=text,
            )
            doc = LibraryDocumentWFindings(
                doc_id=Finding.generate_doc_id(url),
                url=url,
                title=path.stem.replace("_", " ").title(),
                findings=[finding],
                metadata={"source": self.source}
            )
            self.store.store_findings(doc)
            added += 1
        return {"findings": added}
