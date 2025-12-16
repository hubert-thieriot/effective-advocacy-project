"""Helpers for working with collections of embedded corpora."""

from __future__ import annotations

from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from efi_corpus.embedded.embedded_corpus import EmbeddedCorpus
from efi_core.utils import normalize_date

from .identifiers import make_global_doc_id, split_global_doc_id


def _date_in_windows(date_str: str, windows: Optional[Sequence[Tuple[str, str]]]) -> bool:
    """Check if a date (YYYY-MM-DD) falls within any of the specified date windows.
    
    Returns True if windows is None (no filtering) or if date falls within at least one window.
    """
    if windows is None:
        return True
    
    for from_date, to_date in windows:
        if from_date <= date_str <= to_date:
            return True
    
    return False


class EmbeddedCorpora(Mapping[str, EmbeddedCorpus]):
    """Lightweight wrapper around a mapping of corpus name → EmbeddedCorpus.

    This centralises a few common operations that previously lived as free
    functions in the narrative framing workflow (global document id handling,
    date-based filtering, and corpus resolution).
    """

    def __init__(self, corpora: Mapping[str, EmbeddedCorpus]) -> None:
        if not corpora:
            raise ValueError("EmbeddedCorpora requires at least one corpus.")
        self._corpora: Dict[str, EmbeddedCorpus] = dict(corpora)

    # Mapping interface -----------------------------------------------------
    def __getitem__(self, key: str) -> EmbeddedCorpus:  # type: ignore[override]
        return self._corpora[key]

    def __iter__(self) -> Iterator[str]:  # type: ignore[override]
        return iter(self._corpora)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self._corpora)

    # Convenience helpers ---------------------------------------------------
    @property
    def names(self) -> List[str]:
        """Return corpus names in deterministic order."""
        return list(self._corpora.keys())

    def get_embedded(self, corpus_name: Optional[str]) -> EmbeddedCorpus:
        """Resolve an EmbeddedCorpus from an optional corpus name.

        - When a name is provided, it must exist in the collection.
        - When no name is provided and there is a single corpus, that corpus is returned.
        """
        if corpus_name and corpus_name in self._corpora:
            return self._corpora[corpus_name]
        if len(self._corpora) == 1:
            return next(iter(self._corpora.values()))
        raise KeyError(f"Corpus '{corpus_name}' not found among {list(self._corpora.keys())}")

    # Global document id helpers -------------------------------------------
    def list_global_doc_ids(self) -> List[str]:
        """Return all global document ids across corpora.

        For multi-corpus runs, ids are prefixed with the corpus name; for
        single-corpus runs, the local document id is used directly.
        """
        doc_ids: List[str] = []
        multi = len(self._corpora) > 1
        for corpus_name, embedded in self._corpora.items():
            for local_doc_id in embedded.corpus.list_ids():
                doc_ids.append(
                    make_global_doc_id(corpus_name if multi else None, local_doc_id)
                )
        return doc_ids

    def filter_global_doc_ids_by_date(
        self,
        global_doc_ids: Sequence[str],
        *,
        date_from: Optional[str] = None,
        date_windows: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> List[str]:
        """Filter global document ids by date.

        Supports both date_from (lower bound) and date_windows (multiple ranges).
        Documents without a parseable ``published_at`` are dropped.
        """
        if not date_from and not date_windows:
            return list(global_doc_ids)
        
        df_norm: Optional[str] = None
        if date_from:
            df_norm = str(date_from).strip() or None

        keep: List[str] = []
        for gid in global_doc_ids:
            try:
                corpus_name, local_id = split_global_doc_id(gid)
                embedded = self.get_embedded(corpus_name)
                meta = embedded.corpus.get_metadata(local_id)
                pub = meta.get("published_at")
                dt = normalize_date(pub)
                if not dt:
                    continue
                
                date_str = dt.date().isoformat()
                
                # Apply date_from filter if specified
                if df_norm and date_str < df_norm:
                    continue
                
                # Apply date_windows filter if specified
                if date_windows and not _date_in_windows(date_str, date_windows):
                    continue
                
                keep.append(gid)
            except Exception:
                continue
        return keep

    def list_global_doc_ids_from(
        self, 
        date_from: Optional[str] = None,
        date_windows: Optional[Sequence[Tuple[str, str]]] = None,
    ) -> List[str]:
        """Convenience: list global ids and apply optional date filters."""
        return self.filter_global_doc_ids_by_date(
            self.list_global_doc_ids(), date_from=date_from, date_windows=date_windows
        )


__all__ = ["EmbeddedCorpora"]

