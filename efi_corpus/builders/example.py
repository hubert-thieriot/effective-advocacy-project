from pathlib import Path
from hashlib import sha1
from typing import Dict

from ..corpus_handle import CorpusHandle


class ExampleCorpusBuilder:
    """Simple corpus builder that reads local text files."""

    def __init__(self, raw_dir: Path, corpus_dir: Path):
        self.raw_dir = Path(raw_dir)
        self.corpus_dir = Path(corpus_dir)
        self.source = self.raw_dir.parent.name
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        # CorpusHandle requires existing index.jsonl
        (self.corpus_dir / "index.jsonl").touch()
        self.corpus = CorpusHandle(self.corpus_dir, read_only=False)

    def build(self) -> Dict[str, int]:
        """Build the corpus from raw text files."""
        added = 0
        for path in sorted(self.raw_dir.glob("*.txt")):
            text = path.read_text(encoding="utf-8").strip()
            stable_id = sha1(path.stem.encode("utf-8")).hexdigest()
            url = f"https://example.com/corpus/{path.stem}"
            title = path.stem.replace("_", " ").title()
            meta = {
                "doc_id": stable_id,
                "uri": url,
                "title": title,
                "source": self.source,
                "language": "en",
            }
            fetch_info = {"status": "local"}
            self.corpus.write_document(
                stable_id=stable_id,
                meta=meta,
                text=text,
                raw_bytes=text.encode("utf-8"),
                raw_ext="txt",
                fetch_info=fetch_info,
            )
            self.corpus.append_index({
                "id": stable_id,
                "url": url,
                "title": title,
                "language": "en",
            })
            added += 1
        manifest = {
            "name": self.corpus_dir.name,
            "source": self.source,
            "doc_count": added,
        }
        self.corpus.save_manifest(manifest)
        return {"added": added}
