"""
Utility helpers used across the corpus project
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Union


def ensure_date(value: Union[str, date, datetime], param_name: str = "date") -> date:
    """
    Convert a value that may be a string (YYYY-MM-DD), date, or datetime into a date.
    Raises a TypeError/ValueError if conversion is not possible.
    """
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        # Accept ISO format (YYYY-MM-DD) primarily
        try:
            return datetime.fromisoformat(value).date()
        except ValueError:
            # Fallback strict format
            try:
                return datetime.strptime(value, "%Y-%m-%d").date()
            except ValueError as exc:
                raise ValueError(f"Invalid {param_name} format: expected YYYY-MM-DD, got {value!r}") from exc
    raise TypeError(f"{param_name} must be a str, date, or datetime; got {type(value).__name__}")


