"""
Base storer for findings
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

from ..types import DocumentFindings

logger = logging.getLogger(__name__)


class BaseStorer(ABC):
    """Abstract base class for findings storers"""
    
    def __init__(self, name: str, base_path: str = "libraries"):
        """
        Initialize the storer
        
        Args:
            name: Name of the storer
            base_path: Base path for storage
        """
        self.name = name
        self.base_path = Path(base_path)
        self.library_path = self.base_path / name
        self.library_path.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def store_findings(self, findings: DocumentFindings) -> bool:
        """
        Store findings
        
        Args:
            findings: DocumentFindings to store
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_findings(self, url: str) -> Optional[DocumentFindings]:
        """
        Retrieve findings by URL
        
        Args:
            url: URL to retrieve findings for
            
        Returns:
            DocumentFindings object or None if not found
        """
        pass
    
    @abstractmethod
    def list_all_findings(self) -> List[DocumentFindings]:
        """
        List all stored findings
        
        Returns:
            List of all DocumentFindings
        """
        pass
    
    @abstractmethod
    def search_findings(self, query: str) -> List[DocumentFindings]:
        """
        Search findings by query
        
        Args:
            query: Search query
            
        Returns:
            List of matching DocumentFindings
        """
        pass
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get storage information"""
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'base_path': str(self.base_path),
            'library_path': str(self.library_path)
        }
