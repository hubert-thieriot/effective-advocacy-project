"""
JSON storer for findings with document-level storage
"""

import json
import logging
import hashlib
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from .base_storer import BaseStorer
from ..types import DocumentFindings


logger = logging.getLogger(__name__)


class JSONStorer(BaseStorer):
    """Store findings in JSON format with document-level organization"""
    
    def __init__(self, name: str, base_path: str = "libraries"):
        """
        Initialize the JSON storer
        
        Args:
            name: Name of the library
            base_path: Base path for storage
        """
        super().__init__(name, base_path)
        self.documents_dir = self.library_path / "documents"
        self.documents_dir.mkdir(exist_ok=True)
        
        self.metadata_file = self.library_path / "metadata.json"
        self.index_file = self.library_path / "index.json"
        
        # Initialize metadata and index
        self._init_metadata()
        self._init_index()
    
    def _init_metadata(self):
        """Initialize library metadata"""
        if not self.metadata_file.exists():
            metadata = {
                'name': self.name,
                'created_at': datetime.now().isoformat(),
                'last_updated': None,
                'total_documents': 0,
                'total_findings': 0,
                'storage_type': 'json_document_level',
                'storage_version': '2.0'
            }
            self._save_metadata(metadata)
    
    def _init_index(self):
        """Initialize URL to folder mapping index"""
        if not self.index_file.exists():
            index = {
                'version': '2.0',
                'created_at': datetime.now().isoformat(),
                'urls': {}
            }
            self._save_index(index)
    
    def _generate_document_folder_name(self, url: str, title: Optional[str] = None) -> str:
        """Generate folder name: YYYYMM_hash_lowercase_title"""
        # Generate short hash from URL (first 6 characters of MD5)
        url_hash = hashlib.md5(url.encode()).hexdigest()[:6]
        
        # Get current date for YYYYMM prefix
        current_date = datetime.now()
        date_prefix = current_date.strftime("%Y%m")
        
        # Sanitize title for folder name
        if title:
            # Convert to lowercase and remove special characters, replace spaces with underscores
            sanitized_title = title.lower()
            sanitized_title = re.sub(r'[^\w\s-]', '', sanitized_title)
            sanitized_title = re.sub(r'[-\s]+', '_', sanitized_title)
            sanitized_title = sanitized_title.strip('_')
            
            # Limit length to avoid overly long folder names
            if len(sanitized_title) > 50:
                sanitized_title = sanitized_title[:50].rstrip('_')
            
            # Ensure we have a title
            if sanitized_title:
                return f"{date_prefix}_{url_hash}_{sanitized_title}"
        
        # Fallback to just date and hash if no title
        return f"{date_prefix}_{url_hash}"
    
    def _get_document_path(self, url: str, title: Optional[str] = None) -> Path:
        """Get the path for a document's folder"""
        folder_name = self._generate_document_folder_name(url, title)
        return self.documents_dir / folder_name
    
    def _save_document_findings(self, doc_path: Path, findings: DocumentFindings) -> bool:
        """Save findings to a document's folder"""
        try:
            # Create document directory
            doc_path.mkdir(parents=True, exist_ok=True)
            
            # Convert findings to dict and ensure serializability
            findings_dict = self._findings_to_dict(findings)
            
            # Save findings
            findings_file = doc_path / "findings.json"
            with open(findings_file, 'w', encoding='utf-8') as f:
                json.dump(findings_dict, f, indent=2, ensure_ascii=False)
            
            # Save document metadata
            doc_metadata = {
                'url': findings.url,
                'title': findings.title,
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'folder_name': doc_path.name,
                'findings_count': len(findings.findings)
            }
            
            metadata_file = doc_path / "metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(doc_metadata, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving document findings to {doc_path}: {e}")
            return False
    
    def store_findings(self, findings: DocumentFindings) -> bool:
        """
        Store findings in document-level JSON format
        
        Args:
            findings: DocumentFindings to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get document path
            doc_path = self._get_document_path(findings.url, findings.title)
            
            # Check if document already exists
            if doc_path.exists():
                # Update existing document
                logger.info(f"Updating existing document: {doc_path.name}")
            else:
                # Create new document
                logger.info(f"Creating new document: {doc_path.name}")
            
            # Save findings to document folder
            if not self._save_document_findings(doc_path, findings):
                return False
            
            # Update index
            self._update_index(findings.url, doc_path.name)
            
            # Update library metadata
            self._update_library_metadata()
            
            logger.info(f"Stored findings for {findings.url} in {doc_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing findings for {findings.url}: {e}")
            return False
    
    def get_findings(self, url: str) -> Optional[DocumentFindings]:
        """
        Retrieve findings by URL
        
        Args:
            url: URL to retrieve findings for
            
        Returns:
            DocumentFindings object or None if not found
        """
        try:
            # Check index for folder mapping
            index = self._load_index()
            folder_name = index.get('urls', {}).get(url)
            
            if not folder_name:
                return None
            
            doc_path = self.documents_dir / folder_name
            if not doc_path.exists():
                return None
            
            # Load findings from document folder
            findings_file = doc_path / "findings.json"
            if not findings_file.exists():
                return None
            
            with open(findings_file, 'r', encoding='utf-8') as f:
                findings_data = json.load(f)
            
            return self._dict_to_findings(findings_data)
            
        except Exception as e:
            logger.error(f"Error retrieving findings for {url}: {e}")
            return None
    
    def list_all_findings(self) -> List[DocumentFindings]:
        """
        List all stored findings
        
        Returns:
            List of all DocumentFindings
        """
        try:
            all_findings = []
            
            # Iterate through all document folders
            for doc_folder in self.documents_dir.iterdir():
                if not doc_folder.is_dir():
                    continue
                
                findings_file = doc_folder / "findings.json"
                if not findings_file.exists():
                    continue
                
                try:
                    with open(findings_file, 'r', encoding='utf-8') as f:
                        findings_data = json.load(f)
                    
                    findings = self._dict_to_findings(findings_data)
                    all_findings.append(findings)
                    
                except Exception as e:
                    logger.warning(f"Error loading findings from {doc_folder}: {e}")
                    continue
            
            return all_findings
            
        except Exception as e:
            logger.error(f"Error listing all findings: {e}")
            return []
    
    def search_findings(self, query: str) -> List[DocumentFindings]:
        """
        Search findings by query
        
        Args:
            query: Search query
            
        Returns:
            List of matching DocumentFindings
        """
        try:
            query_lower = query.lower()
            results = []
            
            # Load all findings and search
            all_findings = self.list_all_findings()
            
            for findings in all_findings:
                # Search in findings text
                for finding in findings.findings:
                    if query_lower in finding.text.lower():
                        if not any(r.url == findings.url for r in results):
                            results.append(findings)
                        break
                
                # Search in title
                if findings.title and query_lower in findings.title.lower():
                    if not any(r.url == findings.url for r in results):
                        results.append(findings)
                
                # Search in URL
                if query_lower in findings.url.lower():
                    if not any(r.url == findings.url for r in results):
                        results.append(findings)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching findings: {e}")
            return []
    
    def _update_index(self, url: str, folder_name: str):
        """Update the URL to folder mapping index"""
        try:
            index = self._load_index()
            index['urls'][url] = folder_name
            index['last_updated'] = datetime.now().isoformat()
            self._save_index(index)
        except Exception as e:
            logger.error(f"Error updating index: {e}")
    
    def _load_index(self) -> Dict[str, Any]:
        """Load the URL index"""
        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            return {'version': '2.0', 'created_at': datetime.now().isoformat(), 'urls': {}}
    
    def _save_index(self, index: Dict[str, Any]):
        """Save the URL index"""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(index, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving index: {e}")
    
    def _update_library_metadata(self):
        """Update library metadata with current counts"""
        try:
            # Count documents and findings
            total_documents = len(list(self.documents_dir.iterdir()))
            total_findings = sum(len(f.findings) for f in self.list_all_findings())
            
            metadata = {
                'name': self.name,
                'created_at': self._get_created_at(),
                'last_updated': datetime.now().isoformat(),
                'total_documents': total_documents,
                'total_findings': total_findings,
                'storage_type': 'json_document_level',
                'storage_version': '2.0'
            }
            
            self._save_metadata(metadata)
            
        except Exception as e:
            logger.error(f"Error updating library metadata: {e}")
    
    def _save_metadata(self, metadata: Dict[str, Any]):
        """Save metadata to file"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving metadata file: {e}")
    
    def _get_created_at(self) -> str:
        """Get creation date from existing metadata"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    return metadata.get('created_at', datetime.now().isoformat())
        except:
            pass
        
        return datetime.now().isoformat()
    
    def _findings_to_dict(self, doc_findings: DocumentFindings) -> Dict[str, Any]:
        """Convert DocumentFindings to dictionary for storage"""
        # Handle published_at field - it might be a string or datetime
        published_at = None
        if doc_findings.published_at:
            if hasattr(doc_findings.published_at, 'isoformat'):
                # It's a datetime object
                published_at = doc_findings.published_at.isoformat()
            else:
                # It's already a string
                published_at = str(doc_findings.published_at)
        
        # Convert metadata to ensure all datetime objects are serializable
        serializable_metadata = self._make_metadata_serializable(doc_findings.metadata)
        
        return {
            'url': doc_findings.url,
            'title': doc_findings.title,
            'published_at': published_at,
            'language': doc_findings.language,
            'extraction_date': doc_findings.extraction_date.isoformat(),
            'findings': [
                {
                    'text': f.text,
                    'confidence': f.confidence,
                    'category': f.category,
                    'keywords': f.keywords
                }
                for f in doc_findings.findings
            ],
            'metadata': serializable_metadata
        }
    
    def _make_metadata_serializable(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively convert metadata to ensure JSON serializability"""
        if not isinstance(metadata, dict):
            return metadata
        
        serializable_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, datetime):
                # Convert datetime to ISO format string
                serializable_metadata[key] = value.isoformat()
            elif isinstance(value, dict):
                # Recursively handle nested dictionaries
                serializable_metadata[key] = self._make_metadata_serializable(value)
            elif isinstance(value, list):
                # Handle lists (e.g., if metadata contains lists of items)
                serializable_metadata[key] = [
                    self._make_metadata_serializable(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                # Keep other types as-is
                serializable_metadata[key] = value
        
        return serializable_metadata
    
    def _dict_to_findings(self, data: Dict[str, Any]) -> DocumentFindings:
        """Convert dictionary back to DocumentFindings"""
        from ..types import DocumentFindings, Finding
        
        findings = []
        for f_data in data.get('findings', []):
            findings.append(Finding(
                text=f_data.get('text', ''),
                confidence=f_data.get('confidence'),
                category=f_data.get('category'),
                keywords=f_data.get('keywords')
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
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            # Count documents
            total_documents = len([d for d in self.documents_dir.iterdir() if d.is_dir()])
            
            # Count findings
            total_findings = sum(len(f.findings) for f in self.list_all_findings())
            
            # Calculate total file size
            total_size = 0
            for doc_folder in self.documents_dir.iterdir():
                if doc_folder.is_dir():
                    for file_path in doc_folder.rglob('*'):
                        if file_path.is_file():
                            total_size += file_path.stat().st_size
            
            return {
                'total_documents': total_documents,
                'total_findings': total_findings,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'storage_path': str(self.library_path),
                'storage_type': 'document_level'
            }
            
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {}
