"""Helpers for working with corpus/document/passage identifiers.

These utilities provide a consistent way to attach corpus aliases to
document and passage identifiers without breaking backwards
compatibility with historical single-corpus identifiers.
"""

from __future__ import annotations

from typing import Optional, Tuple


GLOBAL_ID_DELIMITER = "@@"


def make_global_doc_id(corpus_name: Optional[str], doc_id: str) -> str:
    """Prefix ``doc_id`` with the corpus alias when provided."""

    if not corpus_name:
        return doc_id
    return f"{corpus_name}{GLOBAL_ID_DELIMITER}{doc_id}"


def split_global_doc_id(identifier: str) -> Tuple[Optional[str], str]:
    """Split a potentially-prefixed document identifier.

    Returns a tuple of (corpus_name, local_doc_id). ``corpus_name`` is
    ``None`` when the identifier does not contain a prefix.
    """

    if GLOBAL_ID_DELIMITER in identifier:
        corpus_name, local_id = identifier.split(GLOBAL_ID_DELIMITER, 1)
        corpus_name = corpus_name or None
        return corpus_name, local_id
    return None, identifier


def make_global_passage_id(corpus_name: Optional[str], local_passage_id: str) -> str:
    """Attach a corpus prefix to a passage identifier.

    ``local_passage_id`` is expected to follow the ``doc_id[:suffix]``
    convention used across the frame tooling. The returned identifier
    preserves the suffix while prefixing the document component with the
    corpus alias when provided.
    """

    if not corpus_name:
        return local_passage_id

    if ":" in local_passage_id:
        doc_part, remainder = local_passage_id.split(":", 1)
        global_doc = make_global_doc_id(corpus_name, doc_part)
        return f"{global_doc}:{remainder}"
    global_doc = make_global_doc_id(corpus_name, local_passage_id)
    return global_doc


def split_passage_id(passage_id: str) -> Tuple[Optional[str], str, str]:
    """Parse a passage identifier into corpus, document, and local parts.

    Returns a tuple of ``(corpus_name, local_doc_id, local_passage_id)``.
    ``local_passage_id`` retains any suffix (e.g. ``doc:chunk001``).
    """

    if ":" in passage_id:
        doc_part, remainder = passage_id.split(":", 1)
    else:
        doc_part, remainder = passage_id, ""

    corpus_name, local_doc_id = split_global_doc_id(doc_part)
    local_passage_id = f"{local_doc_id}:{remainder}" if remainder else local_doc_id
    return corpus_name, local_doc_id, local_passage_id


__all__ = [
    "GLOBAL_ID_DELIMITER",
    "make_global_doc_id",
    "make_global_passage_id",
    "split_global_doc_id",
    "split_passage_id",
]
