"""
Document Matching Pipeline

A simple orchestration class for running document matching between corpus and library.
"""

import pickle
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from efi_core.types import (
    FindingFilters, DocumentMatchingResults, FindingResults, DocumentMatch
)
from efi_core.retrieval import RetrieverIndex
from efi_core.protocols import ReScorer
from efi_corpus.embedded.embedded_corpus import EmbeddedCorpus
from efi_library.embedded.embedded_library import EmbeddedLibrary


class DocumentMatchingPipeline:
    """Simple pipeline for document matching between corpus and library."""
    
    def __init__(self, 
                 embedded_corpus: EmbeddedCorpus,
                 embedded_library: EmbeddedLibrary,
                 rescorers: List[ReScorer],
                 workspace_path: Path):
        self.embedded_corpus = embedded_corpus
        self.embedded_library = embedded_library
        self.rescorers = rescorers
        self.workspace_path = workspace_path
        
        # Initialize retriever
        self.retriever = RetrieverIndex(
            embedded_data_source=embedded_corpus,
            workspace_path=workspace_path,
            chunker_spec=embedded_corpus.chunker.spec,
            embedder_spec=embedded_corpus.embedder.spec,
            auto_rebuild=True,
        )
    
    def run_matching(self,
                    finding_filters: Optional[FindingFilters] = None,
                    n_findings: Optional[int] = None,
                    top_k: int = 10) -> DocumentMatchingResults:
        """Run document matching with given parameters."""
        
        # Collect findings
        findings = []
        findings_processed = 0
        
        for finding in self.embedded_library.library.iter_findings():
            if n_findings and findings_processed >= n_findings:
                break
                
            # Apply filters
            if finding_filters and not self._apply_filters(finding, finding_filters):
                continue
                
            findings.append(finding)
            findings_processed += 1
        
        # Process findings
        results_by_finding = {}
        total_matches = 0
        timing_stats = {}
        score_stats = {}
        
        for i, finding in enumerate(findings):
            finding_results = self._process_finding(finding, top_k)
            # Generate unique ID if finding_id is None
            finding_key = finding.finding_id if finding.finding_id else f"finding_{i+1}"
            results_by_finding[finding_key] = finding_results
            total_matches += len(finding_results.matches)
        
        # Create results container
        results = DocumentMatchingResults(
            findings_processed=len(findings),
            total_matches=total_matches,
            results_by_finding=results_by_finding,
            timing_stats=timing_stats,
            score_stats=score_stats,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "top_k": top_k,
                "filters_applied": finding_filters is not None
            }
        )
        
        return results
    
    def _apply_filters(self, finding, filters: FindingFilters) -> bool:
        """Apply filters to a finding."""
        text_lower = finding.text.lower()
        
        # Include keywords
        if filters.include_keywords:
            if not any(keyword.lower() in text_lower for keyword in filters.include_keywords):
                return False
        
        # Exclude keywords
        if filters.exclude_keywords:
            if any(keyword.lower() in text_lower for keyword in filters.exclude_keywords):
                return False
        
        # Date range
        if filters.date_range and finding.published_at:
            start_date, end_date = filters.date_range
            if not (start_date <= finding.published_at <= end_date):
                return False
        
        # Confidence threshold
        if filters.confidence_threshold and finding.confidence:
            if finding.confidence < filters.confidence_threshold:
                return False
        
        return True
    
    def _process_finding(self, finding, top_k: int) -> FindingResults:
        """Process a single finding."""
        start_time = time.time()
        
        # Retrieve candidates
        candidates = self.retriever.query(finding.text, top_k=top_k)
        
        # Process with each rescorer
        matches = []
        rescorer_scores = {}
        timing = {}
        
        for rescorer in self.rescorers:
            rescorer_start = time.time()
            
            # Get unique rescorer name from the name property
            rescorer_name = rescorer.name
            
            # Enrich candidates with text
            enriched = []
            for candidate in candidates:
                try:
                    chunk = self.embedded_corpus.get_chunk(
                        chunk_id=candidate.item_id, 
                        materialize_if_necessary=False
                    )
                    candidate.metadata["text"] = chunk.text if chunk else ""
                    enriched.append(candidate)
                except Exception:
                    enriched.append(candidate)
            
            # Rescore
            try:
                rescored = rescorer.rescore(finding.text, enriched)
                rescorer_scores[rescorer_name] = [
                    float(r.score) for r in rescored
                ]
            except Exception:
                rescorer_scores[rescorer_name] = [
                    float(r.score) for r in enriched
                ]
            
            timing[rescorer_name] = time.time() - rescorer_start
        
        # Create matches
        for i, candidate in enumerate(candidates):
            match = DocumentMatch(
                chunk_id=candidate.item_id,
                chunk_text=candidate.metadata.get("text", ""),
                cosine_score=float(candidate.score),
                rescorer_scores={
                    name: scores[i] if i < len(scores) else 0.0
                    for name, scores in rescorer_scores.items()
                },
                metadata=candidate.metadata
            )
            matches.append(match)
        
        return FindingResults(
            finding_id=finding.finding_id,
            finding_text=finding.text,
            matches=matches,
            rescorer_scores=rescorer_scores,
            timing=timing
        )
    
    def save_results(self, results: DocumentMatchingResults, output_path: Path):
        """Save results to pickle file."""
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
