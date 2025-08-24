"""
Library handle for reading findings from a library.

Updated to work with the new library structure:
- index.json: maps URLs to document IDs
- documents/{doc_id}/findings.json: contains findings for each document
"""

import json
import logging
from pathlib import Path
from typing import Iterator, Dict, Any, Optional
from datetime import datetime


from .library_store import LibraryStore
from efi_core.types import Finding, LibraryDocument, LibraryDocumentWFindings
from efi_core.layout import LibraryLayout
from efi_core.protocols import Library

logger = logging.getLogger(__name__)


class LibraryHandle(Library):
    """
    Handle for reading findings from a library.
    
    Works with the new library structure:
    - index.json: URL to document ID mapping
    - documents/{doc_id}/findings.json: findings for each document
    """
    
    def __init__(self, library_path: Path):
        """
        Initialize the library handle.
        
        Args: 
            library_path: Path to the findings library directory
        """
        self.library_path = Path(library_path)
        self.layout = LibraryLayout(self.library_path)
        
        if not self.library_path.exists():
            raise ValueError(f"Library path does not exist: {library_path}")
        
        # Detect structure BEFORE initializing the store (store may create files/dirs)
        if (self.layout.index_path.exists() and 
            (self.library_path / "documents").exists()):
            self._structure = "new"
            logger.info("Using new library structure (index.json + documents/)")
        elif self.layout.findings_path.exists():
            self._structure = "old"
            logger.info("Using old library structure (findings.json)")
        else:
            raise ValueError(f"Library structure not recognized. Expected either index.json + documents/ or findings.json in {library_path}")
        
        # Create a store instance for data operations AFTER detection
        self.store = LibraryStore(self.library_path.name, str(self.library_path.parent))
        
        # Ensure all necessary directories exist
        self.layout.ensure_dirs()
    
    
    def iter_findings(self) -> Iterator[LibraryDocumentWFindings]:
        """Iterate over findings in the library.
        Prefer store-backed listing when available; otherwise, fall back to findings.json (old structure).
        """
        # First try using the store (works for new structure and is easily mockable in tests)
        try:
            store_results = self.store.list_all_findings()
            if store_results:
                return iter(store_results)
        except Exception:
            pass

        # Old structure fallback: read from findings.json
        def _iter_old():
            try:
                if not self.layout.findings_path.exists():
                    return
                with open(self.layout.findings_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for doc in data:
                    url = doc.get('url', '')
                    doc_id = Finding.generate_doc_id(url)
                    findings_list = []
                    for fdata in doc.get('findings', []):
                        findings_list.append(Finding(
                            finding_id=fdata.get('finding_id', ''),
                            text=fdata.get('text', ''),
                            confidence=fdata.get('confidence'),
                            category=fdata.get('category'),
                            keywords=fdata.get('keywords', [])
                        ))
                    yield LibraryDocumentWFindings(
                        doc_id=doc_id,
                        url=url,
                        title=doc.get('title', ''),
                        published_at=doc.get('published_at'),
                        language=doc.get('language'),
                        extraction_date=doc.get('extraction_date'),
                        findings=findings_list,
                        metadata=doc.get('metadata', {})
                    )
            except Exception as e:
                logger.error(f"Error iterating old findings structure: {e}")
                return
        return _iter_old()
    

    
    def get_finding(self, finding_id: str) -> Optional[Finding]:
        """
        Get a finding by its ID using the store.
        
        Args:
            finding_id: Finding ID
            
        Returns:
            Finding object if found, None otherwise
        """
        # Try new structure: derive doc_id from finding_id and load via store
        doc_id = self.layout.get_doc_id_from_finding_id(finding_id)
        if doc_id:
            doc_findings = self.store.get_findings_by_doc_id(doc_id)
            if doc_findings:
                for finding in doc_findings.findings:
                    if finding.finding_id == finding_id:
                        return finding
            # Fall through to old-structure check if not found

        # Old structure fallback: linear search in findings.json
        try:
            if self.layout.findings_path.exists():
                with open(self.layout.findings_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for doc in data:
                    for fdata in doc.get('findings', []):
                        if fdata.get('finding_id') == finding_id:
                            return Finding(
                                finding_id=finding_id,
                                text=fdata.get('text', ''),
                                confidence=fdata.get('confidence'),
                                category=fdata.get('category'),
                                keywords=fdata.get('keywords', [])
                            )
        except Exception as e:
            logger.error(f"Error searching old findings structure for {finding_id}: {e}")
        
        return None
    

    
    def get_document_for_finding(self, finding_id: str) -> Optional[LibraryDocumentWFindings]:
        """
        Get the document containing a specific finding.
        
        Args:
            finding_id: Finding ID
            
        Returns:
            LibraryDocumentWFindings object if found, None otherwise
        """
        # Get document ID from finding ID efficiently
        doc_id = self.layout.get_doc_id_from_finding_id(finding_id)
        if not doc_id:
            return None
        
        # Read the specific document
        doc_findings_path = self.layout.doc_findings_path(doc_id)
        if not doc_findings_path.exists():
            return None
        
        try:
            with open(doc_findings_path, 'r', encoding='utf-8') as f:
                doc_data = json.load(f)
            
            # Get URL from mapping
            url_mapping = self.layout.get_url_mapping()
            url = url_mapping.get(doc_id, doc_data.get('url', ''))
            
            # Build findings list
            findings = []
            for finding_data in doc_data.get('findings', []):
                finding = Finding(
                    finding_id=finding_data.get('finding_id', ''),
                    text=finding_data.get('text', ''),
                    confidence=finding_data.get('confidence'),
                    category=finding_data.get('category'),
                    keywords=finding_data.get('keywords', [])
                )
                findings.append(finding)
            
            return LibraryDocumentWFindings(
                doc_id=doc_id,
                url=url,
                title=doc_data.get('title', ''),
                published_at=doc_data.get('published_at'),
                language=doc_data.get('language'),
                extraction_date=doc_data.get('extraction_date'),
                findings=findings,
                metadata=doc_data.get('metadata', {})
            )
            
        except Exception as e:
            logger.error(f"Error reading document for finding {finding_id}: {e}")
            return None
    

    
    def get_findings_count(self) -> int:
        """Get the total number of documents with findings"""
        try:
            # Prefer store if it returns anything
            try:
                store_results = self.store.list_all_findings()
                if store_results:
                    return len(store_results)
            except Exception:
                pass

            # Try layout-provided document IDs (new structure)
            try:
                doc_ids = self.layout.get_document_ids()
                if doc_ids:
                    return len(doc_ids)
            except Exception:
                pass

            # Old structure: count entries in findings.json
            if self.layout.findings_path.exists():
                with open(self.layout.findings_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return len(data) if isinstance(data, list) else 0
            return 0
        except Exception as e:
            logger.error(f"Error getting findings count: {e}")
            return 0
    
    def get_total_findings_count(self) -> int:
        """Get the total number of individual findings across all documents"""
        try:
            total = 0
            for doc_findings in self.iter_findings():
                total += len(doc_findings.findings)
            return total
        except Exception as e:
            logger.error(f"Error getting total findings count: {e}")
            return 0
    
    def get_layout_info(self) -> Dict[str, Any]:
        """Get information about the library layout"""
        return {
            'library_root': str(self.layout.library_root),
            'findings_path': str(self.layout.findings_path),
            'metadata_path': str(self.layout.metadata_path),
            'sources_dir': str(self.layout.sources_dir),
            'structure_type': self._structure,
            'index_path': str(self.layout.index_path) if self._structure == "new" else None
        }
    
    def get_library_info(self) -> Dict[str, Any]:
        """Get information about the library"""
        try:
            # Load metadata if available
            metadata = {}
            if self.layout.metadata_path.exists():
                with open(self.layout.metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            
            # Get current stats
            current_stats = {
                'library_path': str(self.library_path),
                'findings_file': str(self.layout.findings_path),
                'metadata_file': str(self.layout.metadata_path),
                'documents_count': self.get_findings_count(),
                'total_findings_count': self.get_total_findings_count(),
                'structure_type': self._structure,
                'file_size_bytes': 0,
                'last_modified': None
            }
            
            # Add structure-specific info
            if self._structure == "new":
                current_stats['index_file'] = str(self.layout.index_path)
                if self.layout.index_path.exists():
                    current_stats['file_size_bytes'] = self.layout.index_path.stat().st_size
                    current_stats['last_modified'] = datetime.fromtimestamp(self.layout.index_path.stat().st_mtime).isoformat()
            else:
                if self.layout.findings_path.exists():
                    current_stats['file_size_bytes'] = self.layout.findings_path.stat().st_size
                    current_stats['last_modified'] = datetime.fromtimestamp(self.layout.findings_path.stat().st_mtime).isoformat()
            
            # Merge with metadata
            return {**current_stats, **metadata}
            
        except Exception as e:
            logger.error(f"Error getting library info: {e}")
            return {'error': str(e)}
    
    def search_findings(self, query: str, case_sensitive: bool = False) -> Iterator[LibraryDocumentWFindings]:
        """
        Search findings by text query
        
        Args:
            query: Text to search for
            case_sensitive: Whether search should be case sensitive
            
        Yields:
            LibraryDocumentWFindings objects that contain the query
        """
        search_query = query if case_sensitive else query.lower()
        
        for doc_findings in self.iter_findings():
            # Check if query appears in any finding text
            found = False
            for finding in doc_findings.findings:
                finding_text = finding.text if case_sensitive else finding.text.lower()
                if search_query in finding_text:
                    found = True
                    break
            
            if found:
                yield doc_findings
    
    def filter_by_category(self, category: str) -> Iterator[LibraryDocumentWFindings]:
        """
        Filter findings by category
        
        Args:
            category: Category to filter by
            
        Yields:
            LibraryDocumentWFindings objects that contain findings in the specified category
        """
        for doc_findings in self.iter_findings():
            # Check if any finding has the specified category
            found = False
            for finding in doc_findings.findings:
                if finding.category == category:
                    found = True
                    break
            
            if found:
                yield doc_findings
    
    def filter_by_confidence(self, min_confidence: float) -> Iterator[LibraryDocumentWFindings]:
        """
        Filter findings by minimum confidence
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Yields:
            LibraryDocumentWFindings objects that contain findings above the confidence threshold
        """
        for doc_findings in self.iter_findings():
            # Check if any finding has confidence above threshold
            found = False
            for finding in doc_findings.findings:
                if finding.confidence and finding.confidence >= min_confidence:
                    found = True
                    break
            
            if found:
                yield doc_findings
    
    def __iter__(self):
        """Make the reader iterable"""
        return self.iter_findings()
    
    def __len__(self):
        """Return the number of documents with findings"""
        return self.get_findings_count()
