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
                 rescorers_stage1: List[ReScorer],
                 rescorers_stage2: List[ReScorer],
                 workspace_path: Path):
        self.embedded_corpus = embedded_corpus
        self.embedded_library = embedded_library
        self.rescorers_stage1 = rescorers_stage1
        self.rescorers_stage2 = rescorers_stage2
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
                    top_n_retrieval: int = 100,
                    top_n_rescoring_stage1: int = 10) -> DocumentMatchingResults:
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
        
        # Process findings with progress bar
        results_by_finding = {}
        total_matches = 0
        timing_stats = {}
        score_stats = {}
        
        try:
            from tqdm import tqdm
            progress_bar = tqdm(total=len(findings), desc="Processing findings", unit="finding")
        except ImportError:
            progress_bar = None
        
        for i, finding in enumerate(findings):
            if progress_bar:
                progress_bar.set_description(f"Processing finding {i+1}/{len(findings)}")
            
            finding_results = self._process_finding(finding, top_n_retrieval, top_n_rescoring_stage1)
            # Generate unique ID if finding_id is None
            finding_key = finding.finding_id if finding.finding_id else f"finding_{i+1}"
            results_by_finding[finding_key] = finding_results
            total_matches += len(finding_results.matches)
            
            if progress_bar:
                progress_bar.update(1)
        
        if progress_bar:
            progress_bar.close()
        
        # Create results container
        results = DocumentMatchingResults(
            findings_processed=len(findings),
            total_matches=total_matches,
            results_by_finding=results_by_finding,
            timing_stats=timing_stats,
            score_stats=score_stats,
            metadata={
                "timestamp": datetime.now().isoformat(),
                "top_n_retrieval": top_n_retrieval,
                "top_n_rescoring_stage1": top_n_rescoring_stage1,
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
    
    def _process_finding(self, finding, top_n_retrieval: int, top_n_rescoring_stage1: int) -> FindingResults:
        """Process a single finding with two-stage rescoring."""
        start_time = time.time()
        
        # Stage 1: Initial retrieval with more candidates
        initial_candidates = self.retriever.query(finding.text, top_k=top_n_retrieval)
        
        # Enrich all initial candidates with text
        enriched_candidates = []
        for candidate in initial_candidates:
            try:
                chunk = self.embedded_corpus.get_chunk(
                    chunk_id=candidate.item_id, 
                    materialize_if_necessary=False
                )
                candidate.metadata["text"] = chunk.text if chunk else ""
                enriched_candidates.append(candidate)
            except Exception:
                enriched_candidates.append(candidate)
        
        # Stage 2: Use stage1 rescorers to filter down to top candidates
        if self.rescorers_stage1:
            try:
                # Apply all stage1 rescorers and combine scores
                stage1_scores = {}
                for rescorer in self.rescorers_stage1:
                    rescorer_start = time.time()
                    rescorer_name = rescorer.name
                    
                    try:
                        rescored = rescorer.rescore(finding.text, enriched_candidates)
                        stage1_scores[rescorer_name] = [float(r.score) for r in rescored]
                    except Exception:
                        stage1_scores[rescorer_name] = [0.0] * len(enriched_candidates)
                    
                    timing[rescorer_name] = time.time() - rescorer_start
                
                # Combine stage1 scores (simple average for now)
                combined_scores = []
                for i in range(len(enriched_candidates)):
                    scores = [stage1_scores[name][i] for name in stage1_scores.keys()]
                    combined_scores.append(sum(scores) / len(scores))
                
                # Sort by combined stage1 scores and take top candidates
                candidate_scores = list(zip(enriched_candidates, combined_scores))
                candidate_scores.sort(key=lambda x: x[1], reverse=True)
                top_candidates = [c[0] for c in candidate_scores[:top_n_rescoring_stage1]]
                
            except Exception:
                # Fallback to cosine similarity if stage1 fails
                top_candidates = enriched_candidates[:top_n_rescoring_stage1]
        else:
            # No stage1 rescorers, use cosine similarity
            top_candidates = enriched_candidates[:top_n_rescoring_stage1]
        
        # Stage 3: Apply stage2 rescorers only to top candidates
        rescorer_scores = {}
        timing = {}
        
        # Add stage1 scores for all initial candidates
        if self.rescorers_stage1:
            for rescorer_name, scores in stage1_scores.items():
                rescorer_scores[rescorer_name] = scores
        
        # Apply stage2 rescorers only to top candidates
        for rescorer in self.rescorers_stage2:
            rescorer_start = time.time()
            rescorer_name = rescorer.name
            
            try:
                stage2_rescored = rescorer.rescore(finding.text, top_candidates)
                # Create full score list (stage2 scores for top candidates, 0 for others)
                full_scores = [0.0] * len(initial_candidates)
                for i, candidate in enumerate(top_candidates):
                    # Find the position of this candidate in the original list
                    original_idx = next(j for j, c in enumerate(initial_candidates) 
                                     if c.item_id == candidate.item_id)
                    full_scores[original_idx] = float(stage2_rescored[i].score)
                
                rescorer_scores[rescorer_name] = full_scores
            except Exception:
                rescorer_scores[rescorer_name] = [0.0] * len(initial_candidates)
            
            timing[rescorer_name] = time.time() - rescorer_start
        
        # Create matches only for the top candidates (those with both stage1 and stage2 scores)
        matches = []
        for candidate in top_candidates:
            # Find the position of this candidate in the original list
            original_idx = next(j for j, c in enumerate(initial_candidates) 
                             if c.item_id == candidate.item_id)
            
            match = DocumentMatch(
                chunk_id=candidate.item_id,
                chunk_text=candidate.metadata.get("text", ""),
                cosine_score=float(candidate.score),
                rescorer_scores={
                    name: scores[original_idx] if original_idx < len(scores) else 0.0
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
