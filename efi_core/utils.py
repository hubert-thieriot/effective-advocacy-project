"""
Utility functions for efi_core
"""

import json
from datetime import datetime, date
from typing import Optional, Union, Sequence, Tuple
from dateutil import parser

# Type alias for date fields
DateField = Union[datetime, str, None]


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
    Normalize various date formats to datetime objects
    
    Args:
        date_value: Date value that could be datetime, string, or None
        
    Returns:
        datetime object if parsing successful, None otherwise
    """
    if date_value is None:
        return None
    
    if isinstance(date_value, datetime):
        return date_value
    
    if isinstance(date_value, str):
        try:
            # Try to parse the date string
            return parser.parse(date_value)
        except (ValueError, TypeError):
            # If parsing fails, return None
            return None
    
    # For any other type, return None
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
