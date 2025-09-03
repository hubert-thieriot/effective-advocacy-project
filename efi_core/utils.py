"""
Utility functions for efi_core
"""

import json
from datetime import datetime, date
from typing import Optional, Union
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
