"""
Library reader for findings libraries
"""

import json
import logging
from typing import Iterator, Optional, Dict, Any
from pathlib import Path
from datetime import datetime

from .types import DocumentFindings, Finding

logger = logging.getLogger(__name__)


class LibraryReader:
    """Read findings from a findings library"""
    
    def __init__(self, library_path: Path):
        """
        Initialize the library reader
        
        Args:
            library_path: Path to the findings library directory
        """
        self.library_path = Path(library_path)
        self.findings_file = self.library_path / "findings.json"
        self.metadata_file = self.library_path / "metadata.json"
        
        if not self.library_path.exists():
            raise ValueError(f"Library path does not exist: {library_path}")
        
        if not self.findings_file.exists():
            raise ValueError(f"Findings file not found: {self.findings_file}")
    
    def read_findings(self) -> Iterator[DocumentFindings]:
        """
        Read all findings from the library
        
        Yields:
            DocumentFindings objects
        """
        try:
            with open(self.findings_file, 'r', encoding='utf-8') as f:
                findings_data = json.load(f)
            
            for doc_data in findings_data:
                try:
                    doc_findings = self._dict_to_findings(doc_data)
                    yield doc_findings
                except Exception as e:
                    logger.warning(f"Error parsing findings for {doc_data.get('url', 'unknown')}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error reading findings file: {e}")
            raise
    
    def get_findings_count(self) -> int:
        """Get the total number of documents with findings"""
        try:
            with open(self.findings_file, 'r', encoding='utf-8') as f:
                findings_data = json.load(f)
            return len(findings_data)
        except Exception as e:
            logger.error(f"Error getting findings count: {e}")
            return 0
    
    def get_total_findings_count(self) -> int:
        """Get the total number of individual findings across all documents"""
        try:
            with open(self.findings_file, 'r', encoding='utf-8') as f:
                findings_data = json.load(f)
            
            total = 0
            for doc_data in findings_data:
                findings_list = doc_data.get('findings', [])
                total += len(findings_list)
            
            return total
        except Exception as e:
            logger.error(f"Error getting total findings count: {e}")
            return 0
    
    def get_library_info(self) -> Dict[str, Any]:
        """Get information about the library"""
        try:
            # Load metadata if available
            metadata = {}
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            
            # Get current stats
            current_stats = {
                'library_path': str(self.library_path),
                'findings_file': str(self.findings_file),
                'metadata_file': str(self.metadata_file),
                'documents_count': self.get_findings_count(),
                'total_findings_count': self.get_total_findings_count(),
                'file_size_bytes': self.findings_file.stat().st_size if self.findings_file.exists() else 0,
                'last_modified': datetime.fromtimestamp(self.findings_file.stat().st_mtime).isoformat() if self.findings_file.exists() else None
            }
            
            # Merge with metadata
            return {**metadata, **current_stats}
            
        except Exception as e:
            logger.error(f"Error getting library info: {e}")
            return {
                'library_path': str(self.library_path),
                'error': str(e)
            }
    
    def search_findings(self, query: str, case_sensitive: bool = False) -> Iterator[DocumentFindings]:
        """
        Search findings by text query
        
        Args:
            query: Text to search for
            case_sensitive: Whether search should be case sensitive
            
        Yields:
            DocumentFindings objects that contain the query
        """
        search_query = query if case_sensitive else query.lower()
        
        for doc_findings in self.read_findings():
            # Check if query appears in any finding text
            found = False
            for finding in doc_findings.findings:
                finding_text = finding.text if case_sensitive else finding.text.lower()
                if search_query in finding_text:
                    found = True
                    break
            
            if found:
                yield doc_findings
    
    def filter_by_category(self, category: str) -> Iterator[DocumentFindings]:
        """
        Filter findings by category
        
        Args:
            category: Category to filter by
            
        Yields:
            DocumentFindings objects that contain findings in the specified category
        """
        for doc_findings in self.read_findings():
            # Check if any finding has the specified category
            found = False
            for finding in doc_findings.findings:
                if finding.category == category:
                    found = True
                    break
            
            if found:
                yield doc_findings
    
    def filter_by_confidence(self, min_confidence: float) -> Iterator[DocumentFindings]:
        """
        Filter findings by minimum confidence
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Yields:
            DocumentFindings objects that contain findings above the confidence threshold
        """
        for doc_findings in self.read_findings():
            # Check if any finding has confidence above threshold
            found = False
            for finding in doc_findings.findings:
                if finding.confidence and finding.confidence >= min_confidence:
                    found = True
                    break
            
            if found:
                yield doc_findings
    
    def _dict_to_findings(self, data: Dict[str, Any]) -> DocumentFindings:
        """Convert dictionary data to DocumentFindings object"""
        # Parse findings
        findings = []
        for f_data in data.get('findings', []):
            findings.append(Finding(
                text=f_data.get('text', ''),
                confidence=f_data.get('confidence'),
                category=f_data.get('category'),
                keywords=f_data.get('keywords', [])
            ))
        
        # Parse dates
        published_at = None
        if data.get('published_at'):
            try:
                published_at = datetime.fromisoformat(data['published_at'])
            except:
                pass
        
        extraction_date = datetime.now()
        if data.get('extraction_date'):
            try:
                extraction_date = datetime.fromisoformat(data['extraction_date'])
            except:
                pass
        
        return DocumentFindings(
            url=data.get('url', ''),
            title=data.get('title'),
            published_at=published_at,
            language=data.get('language'),
            extraction_date=extraction_date,
            findings=findings,
            metadata=data.get('metadata', {})
        )
    
    def __iter__(self):
        """Make the reader iterable"""
        return self.read_findings()
    
    def __len__(self):
        """Return the number of documents with findings"""
        return self.get_findings_count()
