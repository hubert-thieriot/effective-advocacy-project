from __future__ import annotations

"""Utility helpers for corpus builders."""

from datetime import date, datetime
from typing import Union


def ensure_date(value: Union[str, date, datetime], field: str | None = None) -> date:
    """Coerce an input value into a date object.

    Parameters
    ----------
    value:
        A date, datetime, or ISO 8601 string.
    field:
        Optional field name for error messages.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).date()
        except ValueError as exc:  # pragma: no cover - defensive
            name = f" for {field}" if field else ""
            raise ValueError(f"Invalid date{name}: {value}") from exc
    raise TypeError(f"Unsupported date type: {type(value)!r}")
