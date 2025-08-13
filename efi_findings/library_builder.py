"""
New LibraryBuilder using collectors, extractors, and storers
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import time
import json

from .collectors import BaseURLCollector
from .extractors import BaseFindingExtractor
from .storers import BaseStorer
from .types import DocumentFindings, ExtractionConfig

logger = logging.getLogger(__name__)


class LibraryBuilder:
    """Build findings libraries using collectors, extractors, and storers"""
    
    def __init__(self, 
                 name: str,
                 collector: BaseURLCollector,
                 extractors: List[BaseFindingExtractor],
                 storer: BaseStorer,
                 extraction_config: Optional[ExtractionConfig] = None):
        """
        Initialize the library builder
        
        Args:
            name: Name of the library
            collector: URL collector to use
            extractors: List of finding extractors to try sequentially
            storer: Storer to use for saving findings
            extraction_config: Configuration for extraction
        """
        self.name = name
        self.collector = collector
        self.extractors = extractors
        self.storer = storer
        self.extraction_config = extraction_config or ExtractionConfig()
        
        logger.info(f"Initialized LibraryBuilder: {name}")
        logger.info(f"  Collector: {collector.__class__.__name__}")
        logger.info(f"  Extractors: {[e.__class__.__name__ for e in extractors]}")
        logger.info(f"  Storer: {storer.__class__.__name__}")
    
    def build_library(self, 
                     max_sources: Optional[int] = None,
                     batch_size: int = 5,
                     delay: float = 2.0) -> Dict[str, Any]:
        """
        Build the findings library
        
        Args:
            max_sources: Maximum number of sources to process
            batch_size: Number of sources to process in each batch
            delay: Delay between batches in seconds
            
        Returns:
            Build summary statistics
        """
        logger.info(f"Starting to build findings library: {self.name}")
        
        # Step 1: Collect URLs
        logger.info("Step 1: Collecting URLs...")
        urls = self.collector.collect_urls_limited()
        
        if max_sources:
            urls = urls[:max_sources]
        
        logger.info(f"Collected {len(urls)} URLs to process")
        
        if not urls:
            logger.warning("No URLs collected. Check collector configuration.")
            return self._get_empty_build_summary()
        
        # Step 2: Process URLs in batches
        logger.info("Step 2: Processing URLs...")
        results = []
        
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(urls) + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} URLs)")
            
            batch_results = self._process_batch(batch)
            results.extend(batch_results)
            
            # Add delay between batches
            if i + batch_size < len(urls) and delay > 0:
                logger.info(f"Waiting {delay} seconds before next batch...")
                time.sleep(delay)
        
        # Step 3: Generate build summary
        logger.info("Step 3: Generating build summary...")
        summary = self._get_build_summary(results)
        
        logger.info(f"Completed building findings library: {self.name}")
        logger.info(f"  Total processed: {summary['total_sources_processed']}")
        logger.info(f"  Successful: {summary['successful_extractions']}")
        logger.info(f"  Total findings: {summary['total_findings_extracted']}")
        
        return summary
    
    def _process_batch(self, urls: List[str]) -> List[DocumentFindings]:
        """Process a batch of URLs"""
        results = []
        
        for url in urls:
            try:
                logger.info(f"Processing URL: {url}")
                
                # Try extractors sequentially until one succeeds
                findings = None
                successful_extractor = None
                
                for extractor in self.extractors:
                    try:
                        logger.info(f"  Trying extractor: {extractor.name}")
                        findings = extractor.extract_findings(url)
                        
                        if findings and findings.findings:
                            successful_extractor = extractor.name
                            logger.info(f"  Successfully extracted {len(findings.findings)} findings using {extractor.name}")
                            break
                        else:
                            logger.info(f"  No findings extracted using {extractor.name}")
                            
                    except Exception as e:
                        logger.warning(f"  Error with extractor {extractor.name}: {e}")
                        continue
                
                # Store findings if extraction was successful
                if findings and findings.findings:
                    storage_success = self.storer.store_findings(findings)
                    if storage_success:
                        logger.info(f"  Stored findings successfully")
                        results.append(findings)
                    else:
                        logger.error(f"  Failed to store findings")
                        # Still add to results but mark storage failure
                        findings.metadata['storage_error'] = True
                        results.append(findings)
                else:
                    logger.warning(f"  No findings extracted from {url}")
                    # Create empty findings for failed URLs
                    failed_findings = DocumentFindings(
                        url=url,
                        findings=[],
                        metadata={'error': 'No extractor succeeded', 'extractors_tried': [e.name for e in self.extractors]}
                    )
                    results.append(failed_findings)
                
            except Exception as e:
                logger.error(f"Error processing URL {url}: {e}")
                # Create error findings
                error_findings = DocumentFindings(
                    url=url,
                    findings=[],
                    metadata={'error': str(e)}
                )
                results.append(error_findings)
        
        return results
    
    def _get_build_summary(self, results: List[DocumentFindings]) -> Dict[str, Any]:
        """Generate build summary statistics"""
        total_sources = len(results)
        successful_extractions = sum(1 for r in results if r.findings)
        total_findings = sum(len(r.findings) for r in results)
        
        # Count findings by extractor
        extractor_counts = {}
        for result in results:
            if result.findings:
                # Try to determine which extractor was used (this would need to be added to metadata)
                extractor_name = result.metadata.get('extractor_used', 'unknown')
                extractor_counts[extractor_name] = extractor_counts.get(extractor_name, 0) + 1
        
        return {
            'library_name': self.name,
            'build_timestamp': datetime.now().isoformat(),
            'total_sources_processed': total_sources,
            'successful_extractions': successful_extractions,
            'failed_extractions': total_sources - successful_extractions,
            'success_rate': round(successful_extractions / total_sources * 100, 2) if total_sources > 0 else 0,
            'total_findings_extracted': total_findings,
            'average_findings_per_source': round(total_findings / successful_extractions, 2) if successful_extractions > 0 else 0,
            'extractor_usage': extractor_counts,
            'collector_info': self.collector.get_collection_info(),
            'storer_info': self.storer.get_storage_info()
        }
    
    def _get_empty_build_summary(self) -> Dict[str, Any]:
        """Get empty build summary when no URLs are collected"""
        return {
            'library_name': self.name,
            'build_timestamp': datetime.now().isoformat(),
            'total_sources_processed': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'success_rate': 0.0,
            'total_findings_extracted': 0,
            'average_findings_per_source': 0.0,
            'extractor_usage': {},
            'collector_info': self.collector.get_collection_info(),
            'storer_info': self.storer.get_storage_info(),
            'error': 'No URLs collected'
        }
    
    def get_library_stats(self) -> Dict[str, Any]:
        """Get current library statistics"""
        storer_stats = self.storer.get_storage_stats()
        
        return {
            'library_name': self.name,
            'collector_info': self.collector.get_collection_info(),
            'extractor_info': {
                'count': len(self.extractors),
                'names': [e.name for e in self.extractors],
                'cache_stats': {e.name: e.get_cache_stats() for e in self.extractors}
            },
            'storer_info': self.storer.get_storage_info(),
            **storer_stats
        }
    
    def search_library(self, query: str) -> List[DocumentFindings]:
        """Search the findings library"""
        return self.storer.search_findings(query)
    
    def export_library(self, export_path: Optional[Path] = None) -> Path:
        """Export the findings library"""
        if export_path is None:
            export_path = Path(f"{self.name}_findings_export.json")
        
        # Get all findings
        all_findings = self.storer.list_all_findings()
        
        # Convert to export format
        export_data = []
        for doc_findings in all_findings:
            export_data.append({
                'url': doc_findings.url,
                'title': doc_findings.title,
                'published_at': doc_findings.published_at.isoformat() if doc_findings.published_at else None,
                'extraction_date': doc_findings.extraction_date.isoformat(),
                'findings_count': len(doc_findings.findings),
                'findings': [
                    {
                        'text': finding.text,
                        'confidence': finding.confidence,
                        'category': finding.category,
                        'keywords': finding.keywords
                    }
                    for finding in doc_findings.findings
                ],
                'metadata': doc_findings.metadata
            })
        
        # Save export
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported findings library to: {export_path}")
        return export_path
    
    def clear_cache(self):
        """Clear cache for all extractors"""
        for extractor in self.extractors:
            extractor.clear_cache()
        logger.info(f"Cleared cache for all extractors in library: {self.name}")
    
    def close(self):
        """Close all resources"""
        # Close collector if it has a close method
        if hasattr(self.collector, 'close'):
            self.collector.close()
        
        # Close extractors if they have a close method
        for extractor in self.extractors:
            if hasattr(extractor, 'close'):
                extractor.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
