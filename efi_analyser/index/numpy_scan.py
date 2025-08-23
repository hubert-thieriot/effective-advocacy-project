"""
Simple exact scan index using numpy (dev use only).
"""

import json
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np

from efi_core.protocols import AnnIndex


@dataclass
class NumpyScanIndex:
    _vectors: List[np.ndarray] = field(default_factory=list)
    _refs: List[Tuple[str, int]] = field(default_factory=list)
    _dim: Optional[int] = None

    def add(self, doc_id: str, chunk_ids: List[int], vectors: np.ndarray) -> None:
        if self._dim is None:
            self._dim = vectors.shape[1]
        elif vectors.shape[1] != self._dim:
            raise ValueError(f"Vector dimension mismatch: expected {self._dim}, got {vectors.shape[1]}")
        self._vectors.append(vectors)
        self._refs.extend([(doc_id, i) for i in chunk_ids])

    def query(self, q: List[float], top_k: int) -> List[Tuple[str, int, float]]:
        if not self._vectors:
            return []
        query = np.array(q, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        all_vectors = np.vstack(self._vectors)
        qn = query / (np.linalg.norm(query) + 1e-9)
        vn = all_vectors / (np.linalg.norm(all_vectors, axis=1, keepdims=True) + 1e-9)
        sims = (vn @ qn.T).flatten()
        k = min(top_k, len(sims))
        idx = np.argpartition(-sims, k-1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        out: List[Tuple[str, int, float]] = []
        for i in idx:
            doc_id, chunk_id = self._refs[i]
            out.append((doc_id, chunk_id, float(sims[i])))
        return out

    def persist(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        if self._vectors:
            arr = np.vstack(self._vectors)
        else:
            arr = np.zeros((0, self._dim or 1))
        np.save(path / "vectors.npy", arr)
        (path / "refs.json").write_text(json.dumps(self._refs))
        meta = {"dim": self._dim, "num_vectors": len(self._refs)}
        (path / "metadata.json").write_text(json.dumps(meta, indent=2))

    def load(self, path: Path) -> None:
        vp = path / "vectors.npy"
        if vp.exists():
            arr = np.load(vp)
            if arr.size:
                self._vectors = [arr]
                self._dim = arr.shape[1]
            else:
                self._vectors = []
                self._dim = None
        else:
            self._vectors = []
            self._dim = None
        rp = path / "refs.json"
        self._refs = []
        if rp.exists():
            self._refs = [tuple(x) for x in json.loads(rp.read_text())]


