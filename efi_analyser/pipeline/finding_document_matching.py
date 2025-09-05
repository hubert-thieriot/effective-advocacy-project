"""
Finding Document Matching Pipeline

A simple orchestration class for running document matching between corpus and library.
"""

import pickle
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from efi_core.types import (
    FindingFilters, DocumentMatchingResults, FindingResults, DocumentMatch,
    Candidate, RescoreEngine, RerankPolicy, RerankingEngine, Task
)
from efi_core.retrieval import RetrieverIndex
from efi_analyser.scorers import PairScorer
from efi_corpus.embedded.embedded_corpus import EmbeddedCorpus
from efi_library.embedded.embedded_library import EmbeddedLibrary


class FindingDocumentMatchingPipeline:
    """Simple pipeline for document matching between corpus and library."""
    
    def __init__(self,
                 embedded_corpus: EmbeddedCorpus,
                 embedded_library: EmbeddedLibrary,
                 scorers_stage1: List[PairScorer],
                 scorers_stage2: List[PairScorer],
                 workspace_path: Path):
        self.embedded_corpus = embedded_corpus
        self.embedded_library = embedded_library
        self.scorers_stage1 = scorers_stage1
        self.scorers_stage2 = scorers_stage2
        self.workspace_path = workspace_path

        # Initialize retriever
        self.retriever = RetrieverIndex(
            embedded_data_source=embedded_corpus,
            workspace_path=workspace_path,
            chunker_spec=embedded_corpus.chunker.spec,
            embedder_spec=embedded_corpus.embedder.spec,
            auto_rebuild=True,
        )

        # Initialize rescoring engines
        self.rescore_engine_stage1 = RescoreEngine(scorers_stage1) if scorers_stage1 else None
        self.rescore_engine_stage2 = RescoreEngine(scorers_stage2) if scorers_stage2 else None

        # Initialize reranking policies (simple positive labels for NLI/STANCE)
        self.rerank_policies = {}
        for scorer in scorers_stage1 + scorers_stage2:
            if scorer.task == Task.NLI:
                self.rerank_policies[scorer.name] = RerankPolicy(positive_labels={"entails": 1.0})
            elif scorer.task == Task.STANCE:
                self.rerank_policies[scorer.name] = RerankPolicy(positive_labels={"pro": 1.0})
            else:
                # Default positive label for other tasks
                self.rerank_policies[scorer.name] = RerankPolicy(positive_labels={"positive": 1.0})
    
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
        """Process a single finding with two-stage rescoring using new architecture."""
        start_time = time.time()

        # Stage 1: Initial retrieval with more candidates
        # Retriever now populates candidate text directly
        candidates = self.retriever.query(finding.text, top_k=top_n_retrieval)

        # Stage 1: Apply stage1 scorers to all candidates
        timing = {}
        rescorer_scores = {}

        if self.rescore_engine_stage1:
            stage1_start = time.time()
            candidates = self.rescore_engine_stage1.rescore(candidates, finding.text, premise_is_target=True)
            timing["stage1_rescoring"] = time.time() - stage1_start

            # Store stage1 scores
            for candidate in candidates:
                for scorer_name, score_dict in candidate.scores.items():
                    if scorer_name not in rescorer_scores:
                        rescorer_scores[scorer_name] = []
                    rescorer_scores[scorer_name].append(score_dict)

        # Apply reranking to get top candidates for stage2
        if self.scorers_stage1:
            reranking_engine = RerankingEngine(self.rerank_policies)
            candidates = reranking_engine.rerank(candidates)

        # Take top candidates after stage1
        top_candidates = candidates[:top_n_rescoring_stage1]

        # Stage 2: Apply stage2 scorers only to top candidates
        if self.rescore_engine_stage2:
            stage2_start = time.time()
            top_candidates = self.rescore_engine_stage2.rescore(top_candidates, finding.text, premise_is_target=True)
            timing["stage2_rescoring"] = time.time() - stage2_start

            # Store stage2 scores - scores are dicts like {"entails": 0.8, "contradicts": 0.1, "neutral": 0.1}
            for candidate in top_candidates:
                for scorer_name, score_dict in candidate.scores.items():
                    if scorer_name not in rescorer_scores:
                        # Initialize with empty dicts for all candidates
                        rescorer_scores[scorer_name] = [{"entails": 0.0, "contradicts": 0.0, "neutral": 0.0} for _ in range(len(candidates))]
                    # Find original position and update score
                    original_idx = next(j for j, c in enumerate(candidates)
                                       if c.item_id == candidate.item_id)
                    rescorer_scores[scorer_name][original_idx] = score_dict
        else:
            # If no stage2 scorers, fill with default dicts for remaining candidates
            for scorer_name in rescorer_scores.keys():
                if len(rescorer_scores[scorer_name]) < len(candidates):
                    default_score = {"entails": 0.0, "contradicts": 0.0, "neutral": 0.0}
                    rescorer_scores[scorer_name].extend([default_score] * (len(candidates) - len(rescorer_scores[scorer_name])))

        # Apply final reranking to top candidates
        if self.scorers_stage2:
            reranking_engine = RerankingEngine(self.rerank_policies)
            top_candidates = reranking_engine.rerank(top_candidates)

        # Create matches for top candidates
        matches = []
        for candidate in top_candidates:
            # Find the position of this candidate in the original list
            original_idx = next(j for j, c in enumerate(candidates)
                               if c.item_id == candidate.item_id)

            match = DocumentMatch(
                chunk_id=candidate.item_id,
                chunk_text=candidate.text,
                cosine_score=float(candidate.ann_score),
                rescorer_scores={
                    name: scores[original_idx] if original_idx < len(scores) else {"entails": 0.0, "contradicts": 0.0, "neutral": 0.0}
                    for name, scores in rescorer_scores.items()
                },
                metadata=candidate.meta
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
