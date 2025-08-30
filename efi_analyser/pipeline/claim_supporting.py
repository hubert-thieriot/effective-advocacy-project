"""
Claim Supporting Pipeline

Orchestrates the workflow for analyzing claims against a corpus.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from efi_core.retrieval import RetrieverIndex, SearchResult
from efi_analyser.chunkers import SentenceChunker
from efi_analyser.embedders import SentenceTransformerEmbedder
from efi_analyser.rescorers import NLIReScorer
from efi_corpus.embedded.embedded_corpus import EmbeddedCorpus
from ..types import ClaimSupportingConfig, ClaimSupportingResults, ClaimSupportingResult
from ..utils import MediaSourceMapper

logger = logging.getLogger(__name__)


class ClaimSupportingPipeline:
    """Orchestrates the claim supporting workflow."""
    
    def __init__(self, config: ClaimSupportingConfig):
        self.config = config
        self.workspace_path = config.workspace_path
        
        # Initialize components
        self.chunker = SentenceChunker()
        self.embedder = SentenceTransformerEmbedder(lazy_load=True)
        self.media_mapper = MediaSourceMapper()
        
        # Load embedded corpus
        self.embedded_corpus = EmbeddedCorpus(
            corpus_path=config.corpus_path,
            workspace_path=config.workspace_path,
            chunker=self.chunker,
            embedder=self.embedder
        )
        
        # Initialize retriever
        self.retriever = RetrieverIndex(
            embedded_data_source=self.embedded_corpus,
            workspace_path=config.workspace_path,
            chunker_spec=self.chunker,
            embedder_spec=self.embedder
        )
        
        # Initialize NLI rescorer
        from ..rescorers.nli_rescorer import NLIReScorerConfig
        nli_config = NLIReScorerConfig(model_name=config.nli_model)
        self.nli_rescorer = NLIReScorer(config=nli_config)
    
    def run(self, claims: List[str]) -> List[ClaimSupportingResults]:
        """Run the complete claim supporting pipeline."""
        logger.info(f"Starting claim supporting analysis for {len(claims)} claims")
        
        results = []
        for i, claim in enumerate(claims):
            logger.info(f"Processing claim {i+1}/{len(claims)}: {claim[:50]}...")
            
            try:
                claim_results = self._process_single_claim(claim)
                results.append(claim_results)
                logger.info(f"Successfully processed claim {i+1}")
                
            except Exception as e:
                logger.error(f"Failed to process claim {i+1}: {e}")
                # Continue with other claims
                continue
        
        logger.info(f"Completed claim supporting analysis. Processed {len(results)} claims successfully.")
        return results
    
    def _process_single_claim(self, claim: str) -> ClaimSupportingResults:
        """Process a single claim through the pipeline."""
        # Step 1: Retrieve relevant chunks
        chunks = self._retrieve_chunks(claim)
        
        # Step 2: Apply NLI scoring
        scored_chunks = self._apply_nli_scoring(claim, chunks)
        
        # Step 3: Extract and classify results
        claim_results = self._extract_claim_results(claim, scored_chunks)
        
        # Step 4: Aggregate results
        return self._aggregate_results(claim, claim_results)
    
    def _retrieve_chunks(self, claim: str) -> List[SearchResult]:
        """Retrieve relevant chunks for a claim."""
        try:
            # Query using the retriever
            results = self.retriever.query(
                query_vector=claim,
                top_k=self.config.top_k_retrieval
            )
            
            # Apply cosine threshold filter
            if self.config.cosine_threshold > 0.0:
                results = [r for r in results if r.score >= self.config.cosine_threshold]
            
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving chunks for claim: {e}")
            return []
    
    def _apply_nli_scoring(self, claim: str, chunks: List[SearchResult]) -> List[SearchResult]:
        """Apply NLI rescoring to chunks."""
        try:
            # Enrich chunks with text content
            enriched_chunks = []
            for chunk in chunks:
                try:
                    chunk_data = self.embedded_corpus.get_chunk(
                        chunk_id=chunk.item_id,
                        materialize_if_necessary=False
                    )
                    if chunk_data and chunk_data.text:
                        chunk.metadata["text"] = chunk_data.text
                        enriched_chunks.append(chunk)
                except Exception as e:
                    logger.warning(f"Failed to enrich chunk {chunk.item_id}: {e}")
                    continue
            
            if not enriched_chunks:
                logger.warning("No chunks with text content found")
                return []
            
            # Apply NLI rescoring
            scored_chunks = self.nli_rescorer.rescore(claim, enriched_chunks)
            return scored_chunks
            
        except Exception as e:
            logger.error(f"Error applying NLI scoring: {e}")
            return []
    
    def _extract_claim_results(self, claim: str, chunks: List[SearchResult]) -> List[ClaimSupportingResult]:
        """Extract results for a single claim."""
        results = []
        
        for chunk in chunks:
            try:
                # Get document metadata
                doc_id = chunk.item_id.split('_chunk_')[0] if '_chunk_' in chunk.item_id else chunk.item_id
                doc = self.embedded_corpus.corpus.get_document(doc_id)
                
                if not doc:
                    logger.warning(f"Document not found for chunk {chunk.item_id}")
                    continue
                
                # Extract date
                published_date = doc.published_at
                if published_date:
                    try:
                        if isinstance(published_date, str):
                            parsed_date = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
                        else:
                            parsed_date = published_date
                        date_str = parsed_date.strftime('%Y-%m-%d')
                    except:
                        date_str = "Unknown"
                else:
                    date_str = "Unknown"
                
                # Extract media source
                media_source = self.media_mapper.get_media_source(doc.url)
                
                # Extract NLI scores
                nli_scores = {
                    'contradiction': chunk.metadata.get('nli_contradiction', 0.0),
                    'neutral': chunk.metadata.get('nli_neutral', 0.0),
                    'entailment': chunk.metadata.get('nli_entailment', 0.0)
                }
                
                # Classify chunk based on threshold
                classification = self._classify_chunk(nli_scores)
                
                # Create result record
                result = ClaimSupportingResult(
                    claim=claim,
                    chunk_id=chunk.item_id,
                    document_id=doc_id,
                    classification=classification,
                    entailment_score=nli_scores['entailment'],
                    neutral_score=nli_scores['neutral'],
                    contradiction_score=nli_scores['contradiction'],
                    cosine_score=chunk.score,
                    media_source=media_source,
                    date=date_str,
                    url=doc.url,
                    title=doc.title or "No title"
                )
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error extracting result for chunk {chunk.item_id}: {e}")
                continue
        
        return results
    
    def _classify_chunk(self, nli_scores: Dict[str, float]) -> str:
        """Classify a chunk based on NLI scores using threshold-based logic."""
        entailment = nli_scores['entailment']
        contradiction = nli_scores['contradiction']
        
        # Use config threshold
        threshold = self.config.classification_threshold
        
        if entailment > threshold:
            return 'entailment'
        elif contradiction > threshold:
            return 'contradiction'
        else:
            return 'neutral'
    
    def _aggregate_results(self, claim: str, results: List[ClaimSupportingResult]) -> ClaimSupportingResults:
        """Aggregate results for a single claim."""
        total_chunks = len(results)
        
        # Count classifications
        entailment_count = len([r for r in results if r.classification == 'entailment'])
        contradiction_count = len([r for r in results if r.classification == 'contradiction'])
        neutral_count = len([r for r in results if r.classification == 'neutral'])
        
        # Media breakdown
        media_breakdown = {}
        for result in results:
            media = result.media_source
            if media not in media_breakdown:
                media_breakdown[media] = {'total': 0, 'entailment': 0, 'neutral': 0, 'contradiction': 0}
            
            media_breakdown[media]['total'] += 1
            media_breakdown[media][result.classification] += 1
        
        return ClaimSupportingResults(
            claim=claim,
            results=results,
            total_chunks=total_chunks,
            entailment_count=entailment_count,
            contradiction_count=contradiction_count,
            neutral_count=neutral_count,
            media_breakdown=media_breakdown
        )
