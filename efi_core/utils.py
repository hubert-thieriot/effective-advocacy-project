"""
Utility functions for efi_core
"""

import json
from datetime import datetime, date
from typing import Optional, Union, Sequence, Tuple
from dateutil import parser

# Type alias for date-like fields (API may return date, datetime, or cached JSON string)
DateField = Union[datetime, date, str, None]


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime and date objects"""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, date):
            return obj.isoformat()
        return super().default(obj)


def normalize_date(date_value: DateField) -> Optional[datetime]:
    """
    Normalize various date formats to a comparable datetime for sorting/filtering.

    Handles datetime, date, ISO/other date strings (e.g. from cached JSON),
    and None. All successful results are datetime so comparisons never mix types.

    Args:
        date_value: Date value that could be datetime, date, string, or None.

    Returns:
        datetime object if parsing successful, None otherwise.
    """
    if date_value is None:
        return None

    if isinstance(date_value, datetime):
        return date_value

    if isinstance(date_value, date):
        return datetime.combine(date_value, datetime.min.time())

    if isinstance(date_value, str):
        val = date_value.strip()
        if not val:
            return None
        try:
            return parser.parse(val)
        except (ValueError, TypeError):
            return None

    # Objects with a .date() method (e.g. datetime-like from another library)
    if hasattr(date_value, "date"):
        try:
            d = date_value.date()
            return datetime.combine(d, datetime.min.time()) if d else None
        except (ValueError, TypeError):
            return None

    return None


def date_in_windows(date_str: str, windows: Optional[Sequence[Tuple[str, str]]]) -> bool:
    """
    Check if a date (YYYY-MM-DD) falls within any of the specified date windows.

    Args:
        date_str: Date string in YYYY-MM-DD format
        windows: List of (from_date, to_date) tuples, or None for no filtering

    Returns:
        True if windows is None (no filtering) or if date falls within at least one window
    """
    if windows is None:
        return True

    for from_date, to_date in windows:
        if from_date <= date_str <= to_date:
            return True

    return False
