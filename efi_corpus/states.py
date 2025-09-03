"""
State stores for documents and findings.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from efi_core.types import DocState, FindingState, ChunkerSpec, EmbedderSpec
from efi_core.utils import DateTimeEncoder
from efi_core.layout import EmbeddedCorpusLayout


class DocStateStore:
    def __init__(self, layout: EmbeddedCorpusLayout):
        self.layout = layout

    def read(self, doc_id: str) -> Optional[DocState]:
        state_path = self.layout.doc_state_path(doc_id)
        if not state_path.exists():
            return None
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            return DocState(**data)
        except Exception:
            return None

    def write(self, state: DocState) -> None:
        state_path = self.layout.doc_state_path(state.document_id)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(asdict(state), indent=2, ensure_ascii=False, cls=DateTimeEncoder), encoding="utf-8")

    def needs_rebuild(self, *, doc_id: str, fingerprint: str, chunker: ChunkerSpec, embedder: EmbedderSpec) -> bool:
        state = self.read(doc_id)
        if state is None:
            return True
        return (
            state.fingerprint != fingerprint
            or state.chunker_key != chunker.key()
            or state.embedder_key != embedder.key()
        )


class FindingStateStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def path(self, finding_id: str) -> Path:
        return self.root / "finding_meta" / f"{finding_id}.meta.json"

    def read(self, finding_id: str) -> Optional[FindingState]:
        p = self.path(finding_id)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return FindingState(**data)
        except Exception:
            return None

    def write(self, state: FindingState) -> None:
        p = self.path(state.finding_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(state), indent=2, ensure_ascii=False, cls=DateTimeEncoder), encoding="utf-8")

    def needs_rebuild(self, *, finding_id: str, fingerprint: str, embedder: EmbedderSpec) -> bool:
        st = self.read(finding_id)
        if st is None:
            return True
        return st.fingerprint != fingerprint or st.embedder_key != embedder.key()


