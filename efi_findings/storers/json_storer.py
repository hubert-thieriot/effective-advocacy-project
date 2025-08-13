"""
JSON storer for findings
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from .base_storer import BaseStorer
from ..types import DocumentFindings


logger = logging.getLogger(__name__)


class JSONStorer(BaseStorer):
    """Store findings in JSON format"""
    
    def __init__(self, name: str, base_path: str = "libraries"):
        """
        Initialize the JSON storer
        
        Args:
            name: Name of the library
            base_path: Base path for storage
        """
        super().__init__(name, base_path)
        self.findings_file = self.library_path / "findings.json"
        self.metadata_file = self.library_path / "metadata.json"
        
        # Initialize metadata
        self._init_metadata()
    
    def _init_metadata(self):
        """Initialize library metadata"""
        if not self.metadata_file.exists():
            metadata = {
                'name': self.name,
                'created_at': datetime.now().isoformat(),
                'last_updated': None,
                'total_documents': 0,
                'total_findings': 0,
                'storage_type': 'json'
            }
            self._save_metadata(metadata)
    
    def store_findings(self, findings: DocumentFindings) -> bool:
        """
        Store findings in JSON format
        
        Args:
            findings: DocumentFindings to store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load existing findings
            existing_findings = self._load_findings()
            
            # Check if URL already exists
            url_exists = False
            for i, existing in enumerate(existing_findings):
                if existing['url'] == findings.url:
                    # Update existing entry
                    existing_findings[i] = self._findings_to_dict(findings)
                    url_exists = True
                    break
            
            if not url_exists:
                # Add new findings
                existing_findings.append(self._findings_to_dict(findings))
            
            # Save updated findings
            self._save_findings(existing_findings)
            
            # Update metadata
            self._update_metadata(existing_findings)
            
            logger.info(f"Stored findings for {findings.url}")
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
            findings_list = self._load_findings()
            
            for finding_data in findings_list:
                if finding_data.get('url') == url:
                    return self._dict_to_findings(finding_data)
            
            return None
            
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
            findings_list = self._load_findings()
            return [self._dict_to_findings(f) for f in findings_list]
            
        except Exception as e:
            logger.error(f"Error listing findings: {e}")
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
            findings_list = self._load_findings()
            query_lower = query.lower()
            
            results = []
            for finding_data in findings_list:
                # Search in findings text
                findings_text = finding_data.get('findings', [])
                for finding in findings_text:
                    if query_lower in finding.get('text', '').lower():
                        results.append(self._dict_to_findings(finding_data))
                        break
                
                # Search in title
                title = finding_data.get('title', '')
                if title and query_lower in title.lower():
                    if not any(r.url == finding_data.get('url') for r in results):
                        results.append(self._dict_to_findings(finding_data))
                
                # Search in URL
                url = finding_data.get('url', '')
                if query_lower in url.lower():
                    if not any(r.url == url for r in results):
                        results.append(self._dict_to_findings(finding_data))
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching findings: {e}")
            return []
    
    def _load_findings(self) -> List[Dict[str, Any]]:
        """Load findings from JSON file"""
        if not self.findings_file.exists():
            return []
        
        try:
            with open(self.findings_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading findings file: {e}")
            return []
    
    def _save_findings(self, findings_list: List[Dict[str, Any]]):
        """Save findings to JSON file"""
        try:
            with open(self.findings_file, 'w', encoding='utf-8') as f:
                json.dump(findings_list, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving findings file: {e}")
    
    def _save_metadata(self, metadata: Dict[str, Any]):
        """Save metadata to JSON file"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving metadata file: {e}")
    
    def _update_metadata(self, findings_list: List[Dict[str, Any]]):
        """Update metadata with current findings"""
        try:
            total_documents = len(findings_list)
            total_findings = sum(len(f.get('findings', [])) for f in findings_list)
            
            metadata = {
                'name': self.name,
                'created_at': self._get_created_at(),
                'last_updated': datetime.now().isoformat(),
                'total_documents': total_documents,
                'total_findings': total_findings,
                'storage_type': 'json'
            }
            
            self._save_metadata(metadata)
            
        except Exception as e:
            logger.error(f"Error updating metadata: {e}")
    
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
            'metadata': doc_findings.metadata
        }
    
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
            findings_list = self._load_findings()
            
            total_documents = len(findings_list)
            total_findings = sum(len(f.get('findings', [])) for f in findings_list)
            
            # Calculate file size
            file_size = 0
            if self.findings_file.exists():
                file_size = self.findings_file.stat().st_size
            
            return {
                'total_documents': total_documents,
                'total_findings': total_findings,
                'file_size_bytes': file_size,
                'file_size_mb': round(file_size / (1024 * 1024), 2),
                'storage_path': str(self.library_path)
            }
            
        except Exception as e:
            logger.error(f"Error getting storage stats: {e}")
            return {}
