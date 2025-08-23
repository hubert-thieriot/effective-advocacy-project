"""
URL collectors for different content sources
"""

from .base_collector import BaseURLCollector
from .crea_collector import CREAPublicationsCollector

__all__ = [
    'BaseURLCollector',
    'CREAPublicationsCollector'
]
