#!/usr/bin/env python3
"""
Coal Overcapacity Analysis Script

This script performs systematic analysis of coal overcapacity discourse by:
1. Retrieving relevant statements using multiple queries per target
2. Rescoring statements with NLI hypotheses to determine stance
3. Aggregating results for comprehensive analysis

Targets: build_more_coal, coal_overcapacity
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Set
import json
from datetime import datetime
import logging
from collections import defaultdict
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from efi_corpus.embedded import EmbeddedCorpus
from efi_core.retrieval.retriever_index import RetrieverIndex
from efi_core.types import ChunkerSpec, EmbedderSpec, Candidate
from efi_analyser.chunkers import TextChunker
from efi_analyser.embedders import SentenceTransformerEmbedder
from efi_analyser.scorers import NLIHFScorer, NLILLMScorer, LLMScorerConfig, NLIOpenAIScorer, NLIOpenAIConfig
import sys
from pathlib import Path

from efi_analyser.cache.llm_cache_manager import get_cache_manager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Analysis configuration
ANALYSIS_CONFIG = {
    "build_more_coal": {
        "queries": [
            "plans to build more coal power plants",
            "approving new coal capacity",
            "expanding coal generation",
            "constructing new coal plants",
            "commissioning coal-fired power units",
            "adding gigawatts of coal capacity",
            "building coal power plants"
        ],
        "hypotheses": {
            "pro": [
                "New coal power plants should be built.",
                "Coal capacity should be expanded."
            ],
            "anti": [
                "New coal power plants should not be built.",
                "Coal capacity should not be expanded."
            ]
        }
    },
    "coal_overcapacity": {
        "queries": [
            "problem of coal overcapacity",
            "too much coal capacity compared to demand",
            "coal plants sitting idle",
            "surplus coal capacity",
            "excess capacity in coal power sector",
            "stranded coal assets"
        ],
        "hypotheses": {
            "pro": [
                "Coal overcapacity should be reduced.",
                "Excess coal capacity should be cut.",
                "There is no need for new coal capacity."
            ],
            "anti": [
                "Coal overcapacity should not be reduced.",
                "Excess coal capacity should be maintained or increased.",
                "India needs more coal capacity."
            ]
        }
    }
}

def create_embedded_corpus(corpus_name: str) -> EmbeddedCorpus:
    """Create and return the embedded corpus."""
    corpus_path = project_root / "corpora" / corpus_name
    workspace_path = project_root / "workspace"
    
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")
    
    # Initialize chunker and embedder
    chunker = TextChunker()
    embedder = SentenceTransformerEmbedder()
    
    # Create embedded corpus
    embedded_corpus = EmbeddedCorpus(
        corpus_path=corpus_path,
        workspace_path=workspace_path,
        chunker=chunker,
        embedder=embedder
    )
    
    return embedded_corpus

def create_retriever(embedded_corpus: EmbeddedCorpus) -> RetrieverIndex:
    """Create the index retriever."""
    retriever = RetrieverIndex(
        embedded_data_source=embedded_corpus,
        workspace_path=project_root / "workspace",
        chunker_spec=embedded_corpus.chunker_spec,
        embedder_spec=embedded_corpus.embedder_spec,
        auto_rebuild=True
    )
    
    return retriever

def retrieve_statements_for_query(retriever: RetrieverIndex, query: str, top_k: int = 10) -> List[Candidate]:
    """Retrieve top-k statements for a given query."""
    logger.info(f"Retrieving {top_k} statements for query: '{query}'")
    
    try:
        candidates = retriever.query(query, top_k=top_k)
        logger.info(f"Successfully retrieved {len(candidates)} candidates")
        return candidates
    except Exception as e:
        logger.error(f"Error retrieving statements for query '{query}': {e}")
        return []

def collect_unique_candidates(candidates_list: List[List[Candidate]]) -> List[Candidate]:
    """Collect candidates from multiple queries and remove duplicates based on text content."""
    seen_texts = set()
    unique_candidates = []
    
    for candidates in candidates_list:
        for candidate in candidates:
            # Use text content as uniqueness key
            text_key = candidate.text.strip()
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_candidates.append(candidate)
    
    logger.info(f"Collected {len(unique_candidates)} unique candidates from {sum(len(c) for c in candidates_list)} total")
    return unique_candidates

def rescore_candidates_with_all_models(candidates: List[Candidate], hypotheses: Dict[str, List[str]], 
                                      scorers: Dict[str, Any]) -> Dict[str, Any]:
    """Rescore candidates with all available NLI models."""
    results = {}
    
    for i, candidate in enumerate(tqdm(candidates, desc="Multi-model NLI rescoring")):
        candidate_results = {
            "text": candidate.text,
            "item_id": candidate.item_id,
            "ann_score": candidate.ann_score,
            "rescorers": {},
            "aggregated_scores": {}
        }
        
        # Apply each rescorer
        for rescorer_name, scorer in scorers.items():
            logger.info(f"🔄 Starting rescoring candidate {i+1}/{len(candidates)} with {rescorer_name}")
            
            rescorer_results = {
                "hypothesis_scores": {}
            }

            # Score each hypothesis category (pro/anti)
            for stance, hypothesis_list in hypotheses.items():
                category_scores = []

                for hypothesis in hypothesis_list:
                    # Score the hypothesis against the candidate text
                    scores = scorer.batch_score([candidate.text], [hypothesis])[0]

                    category_scores.append({
                        "hypothesis": hypothesis,
                        "entails": scores.get("entails", 0.0),
                        "contradicts": scores.get("contradicts", 0.0),
                        "neutral": scores.get("neutral", 0.0)
                    })

                rescorer_results["hypothesis_scores"][stance] = category_scores
            
            # Calculate aggregate scores for this rescorer
            rescorer_results["aggregate_scores"] = calculate_aggregate_scores(rescorer_results["hypothesis_scores"])
            candidate_results["rescorers"][rescorer_name] = rescorer_results
            logger.info(f"✅ Completed rescoring with {rescorer_name}")
        
        # Calculate overall aggregated scores across all rescorers
        candidate_results["aggregated_scores"] = calculate_overall_aggregate_scores(candidate_results["rescorers"])

        results[candidate.item_id] = candidate_results

    return results

def select_top_candidates(candidates: List[Candidate], top_k: int = 20) -> List[Candidate]:
    """Select top candidates based on retrieval scores."""
    # Sort by ann_score (retrieval score) and take top_k
    sorted_candidates = sorted(candidates, key=lambda x: x.ann_score, reverse=True)
    top_candidates = sorted_candidates[:top_k]
    
    logger.info(f"Selected {len(top_candidates)} top candidates for rescoring")
    return top_candidates


def calculate_aggregate_scores(hypothesis_scores: Dict[str, List[Dict]]) -> Dict[str, float]:
    """Calculate aggregate entail/contradict/neutral scores across all hypotheses."""
    all_entails = []
    all_contradicts = []
    all_neutral = []
    
    for stance, hypotheses in hypothesis_scores.items():
        if stance == "anti":
            for hypothesis in hypotheses:
                all_entails.append(hypothesis["contradicts"])
                all_contradicts.append(hypothesis["entails"])
                all_neutral.append(hypothesis["neutral"])
        elif stance == "pro":
            for hypothesis in hypotheses:
                all_entails.append(hypothesis["entails"])
                all_contradicts.append(hypothesis["contradicts"])
                all_neutral.append(hypothesis["neutral"])
        else:
            raise ValueError(f"Invalid stance: {stance}")
    
    # Take max for entails and contradicts, and average for neutral
    return {
        "entails": max(all_entails) if all_entails else 0.0,
        "contradicts": max(all_contradicts) if all_contradicts else 0.0,
        "neutral": sum(all_neutral) / len(all_neutral) if all_neutral else 0.0
    }

def calculate_overall_aggregate_scores(rescorers: Dict[str, Dict]) -> Dict[str, float]:
    """Calculate overall aggregate scores across all rescorers."""
    all_entails = []
    all_contradicts = []
    all_neutral = []
    
    for rescorer_name, rescorer_results in rescorers.items():
        aggregate_scores = rescorer_results.get("aggregate_scores", {})
        all_entails.append(aggregate_scores.get("entails", 0.0))
        all_contradicts.append(aggregate_scores.get("contradicts", 0.0))
        all_neutral.append(aggregate_scores.get("neutral", 0.0))
    
    # Take max across all rescorers for entails and contradicts, average for neutral
    return {
        "entails": max(all_entails) if all_entails else 0.0,
        "contradicts": max(all_contradicts) if all_contradicts else 0.0,
        "neutral": sum(all_neutral) / len(all_neutral) if all_neutral else 0.0
    }

def analyze_target(target_name: str, target_config: Dict[str, Any], scorers: Dict[str, Any], retriever: RetrieverIndex) -> Dict[str, Any]:
    """Analyze a single target (build_more_coal or coal_overcapacity)."""
    logger.info(f"🔍 Analyzing target: {target_name}")
    
    # Collect candidates from all queries
    all_candidates = []
    for query in target_config["queries"]:
        candidates = retrieve_statements_for_query(retriever, query, top_k=20)
        all_candidates.append(candidates)
    
    # Get unique candidates
    unique_candidates = collect_unique_candidates(all_candidates)
    
    # Select top candidates for rescoring
    top_candidates = select_top_candidates(unique_candidates, top_k=20)
    
    # Apply all rescorers to top candidates
    logger.info(f"Applying all rescorers to top {len(top_candidates)} candidates for {target_name}")
    rescored_results = rescore_candidates_with_all_models(top_candidates, target_config["hypotheses"], scorers)

    # Calculate target-level summary
    summary = {
        "target_name": target_name,
        "total_queries": len(target_config["queries"]),
        "total_candidates_collected": sum(len(c) for c in all_candidates),
        "unique_candidates": len(unique_candidates),
        "top_candidates_rescored": len(top_candidates),
        "rescorers": list(scorers.keys()),
        "results": rescored_results
    }
    
    return summary

def save_results(results: Dict[str, Any], output_path: Path) -> None:
    """Save analysis results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add timestamp and metadata
    results["timestamp"] = datetime.now().isoformat()
    results["analysis_config"] = ANALYSIS_CONFIG

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to: {output_path}")

def generate_html_report(analysis_results: Dict[str, Any], output_path: Path) -> None:
    """Generate HTML report showing most pro/anti sentences for each target."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get cache statistics
    cache_manager = get_cache_manager()
    cache_stats = cache_manager.get_cache_stats()
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Coal Overcapacity Analysis Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        
        .target-section {{
            background: white;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .target-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .stance-section {{
            margin-bottom: 30px;
            border-left: 4px solid #007bff;
            padding-left: 15px;
        }}

        .stance-section.anti {{
            border-left-color: #dc3545;
        }}

        .sentence-card {{
            background: #f8f9fa;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 10px;
        }}
        
        .sentence-meta {{
            font-size: 12px;
            color: #6c757d;
            margin-bottom: 5px;
        }}
        
        .sentence-text {{
            font-size: 14px;
            line-height: 1.5;
        }}
        
        .score-badge {{
            background: #28a745;
            color: white;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: bold;
        }}

        .score-badge.anti {{
            background: #dc3545;
        }}
        
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        
        .metric-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #28a745;
        }}
        
        .metric-label {{
            font-size: 14px;
            color: #6c757d;
            margin-top: 5px;
        }}
        
        .section-title {{
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 20px;
            color: #495057;
            border-bottom: 2px solid #dee2e6;
            padding-bottom: 10px;
        }}
        
        .target-title {{
            font-size: 20px;
            margin: 0;
        }}
        
        .target-stats {{
            font-size: 14px;
            opacity: 0.9;
        }}
        
        .stance-title {{
            font-size: 18px;
            margin-bottom: 15px;
            color: #007bff;
        }}

        .stance-title.anti {{
            color: #dc3545;
        }}
        
        .footer {{
            text-align: center;
            color: #6c757d;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
        }}

        .scores-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
            font-size: 13px;
        }}

        .scores-table th, .scores-table td {{
            padding: 8px 12px;
            text-align: center;
            border-bottom: 1px solid #dee2e6;
            border-right: 1px solid #dee2e6;
            vertical-align: middle;
        }}

        .scores-table th {{
            background-color: #f8f9fa;
            font-weight: 600;
            font-size: 12px;
        }}

        .scores-table th:first-child,
        .scores-table td:first-child {{
            text-align: left;
            border-right: 2px solid #dee2e6;
        }}

        .hypothesis-cell {{
            max-width: 300px;
            text-align: left;
            font-size: 12px;
            line-height: 1.4;
        }}

        .pro-hypothesis {{
            background-color: #e8f5e8;
        }}

        .anti-hypothesis {{
            background-color: #fdeaea;
        }}

        .score-triple {{
            font-family: monospace;
            font-size: 11px;
            white-space: nowrap;
        }}

        .entails-score {{
            color: #28a745;
            font-weight: bold;
        }}

        .neutral-score {{
            color: #6c757d;
        }}

        .contradicts-score {{
            color: #dc3545;
            font-weight: bold;
        }}

        .score-cell {{
            font-family: monospace;
            font-size: 11px;
            text-align: center;
        }}

        .max-score-cell {{
            font-weight: bold;
            color: #007bff;
            background-color: #e3f2fd;
        }}

        .score-bar-container {{
            position: relative;
            width: 100%;
            height: 20px;
            background-color: #f0f0f0;
            border-radius: 10px;
            overflow: hidden;
            margin: 2px 0;
        }}

        .score-bar {{
            height: 100%;
            border-radius: 10px;
            transition: width 0.3s ease, box-shadow 0.2s ease;
            position: relative;
        }}

        .score-bar:hover {{
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            transform: translateY(-1px);
        }}

        .score-bar.entails {{
            background: linear-gradient(90deg, #28a745, #20c997);
        }}

        .score-bar.contradicts {{
            background: linear-gradient(90deg, #dc3545, #fd7e14);
        }}

        .score-bar.neutral {{
            background: linear-gradient(90deg, #6c757d, #adb5bd);
        }}

        .score-bar.max {{
            background: linear-gradient(90deg, #007bff, #0056b3);
            box-shadow: 0 0 8px rgba(0, 123, 255, 0.3);
        }}

        .score-text {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 10px;
            font-weight: bold;
            color: white;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
            z-index: 1;
        }}

        .score-cell-with-bar {{
            padding: 4px 8px;
            text-align: center;
            vertical-align: middle;
        }}

        /* Details/expanders */
        details {{
            margin-top: 8px;
        }}
        details > summary {{
            cursor: pointer;
            font-weight: 600;
            color: #495057;
            list-style: none;
        }}
        details > summary::-webkit-details-marker {{
            display: none;
        }}
        .details-section-title {{
            font-size: 13px;
            color: #6c757d;
            margin: 10px 0 6px 0;
        }}
        .stance-subtitle {{
            font-size: 13px;
            margin: 8px 0 6px 0;
            color: #007bff;
        }}
        .stance-subtitle.anti {{
            color: #dc3545;
        }}
        .hypothesis-col {{
            max-width: 680px;
            white-space: normal;
        }}
        .highlight-pro {{
            background-color: #e9f7ef;
        }}
        .highlight-anti {{
            background-color: #fdecea;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Coal Overcapacity Analysis Report</h1>
        <p>Multi-model NLI-based stance detection for coal overcapacity discourse</p>
        <p><strong>Models:</strong> HF (RoBERTa), LLM (llama3.2:3b), Phi (phi3:3.8b), OpenAI (gpt-3.5-turbo)</p>
        <p><strong>Generated:</strong> {timestamp}</p>
        <p><strong>Cache Stats:</strong> {cache_stats['total_entries']} entries ({cache_stats['total_size_mb']:.1f} MB) | 
           <strong>Models cached:</strong> {', '.join(cache_stats['models'].keys()) if cache_stats['models'] else 'None'}</p>
    </div>
    
    <div class="target-section">
        <h2 class="section-title">📊 Analysis Summary</h2>
        
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{len(ANALYSIS_CONFIG)}</div>
                <div class="metric-label">Targets Analyzed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{sum(results['unique_candidates'] for results in analysis_results.values() if 'unique_candidates' in results)}</div>
                <div class="metric-label">Total Unique Statements</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{sum(results['total_queries'] for results in analysis_results.values() if 'total_queries' in results)}</div>
                <div class="metric-label">Total Queries Used</div>
            </div>
        </div>
    </div>
"""

    # Add sections for each target
    for target_name, target_results in analysis_results.items():
        if target_name not in ANALYSIS_CONFIG:
            continue

        # Extract top pro and anti sentences
        candidates = target_results.get("results", {})

        # Get all sentences with their scores from all models
        all_sentences = []

        for item_id, candidate_data in candidates.items():
            # Collect scores from all available rescorers in the new format
            model_scores = {}
            rescorers = candidate_data.get("rescorers", {})
            
            for rescorer_name, rescorer_data in rescorers.items():
                aggregate_scores = rescorer_data.get("aggregate_scores", {})
                if aggregate_scores:
                    model_scores[rescorer_name] = aggregate_scores
            
            # Use aggregated scores for ranking
            aggregated_scores = candidate_data.get("aggregated_scores", {})
            primary_score = max(aggregated_scores.get("entails", 0.0), aggregated_scores.get("contradicts", 0.0))
            
            all_sentences.append((primary_score, candidate_data, model_scores))

        # Sort by primary score and get top 20
        all_sentences.sort(key=lambda x: x[0], reverse=True)
        top_sentences = all_sentences[:20]

        html_content += f"""
    <div class="target-section">
        <div class="target-header">
            <div>
                <h3 class="target-title">Target: {target_name.replace('_', ' ').title()}</h3>
                <div class="target-stats">
                    {target_results.get('total_queries', 0)} queries |
                    {target_results.get('unique_candidates', 0)} unique statements |
                    Top {len(top_sentences)} statements shown
                </div>
            </div>
            <div class="score-badge">Multi-Model NLI Analysis</div>
        </div>

        <div class="stance-section">
            <h4 class="stance-title">📊 Top Statements with All Rescorer Results</h4>
            <p style="font-size: 12px; color: #6c757d; margin-bottom: 15px;">
                <em>Ranked by highest score across all models. Shows entails, contradicts, and neutral scores for each model.</em>
            </p>
            
            <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 15px; font-size: 12px;">
                <strong>Model Coverage:</strong> 
                {f"HF: {sum(1 for _, _, scores in top_sentences if 'HF (RoBERTa)' in scores)}/{len(top_sentences)} | " if top_sentences else ""}
                {f"LLM: {sum(1 for _, _, scores in top_sentences if 'LLM (llama3.2:3b)' in scores)}/{len(top_sentences)} | " if top_sentences else ""}
                {f"Phi: {sum(1 for _, _, scores in top_sentences if 'Phi (phi3:3.8b)' in scores)}/{len(top_sentences)} | " if top_sentences else ""}
                {f"OpenAI: {sum(1 for _, _, scores in top_sentences if 'OpenAI (gpt-3.5-turbo)' in scores)}/{len(top_sentences)}" if top_sentences else "No statements"}
                </div>
            
            <div style="background: #e9ecef; padding: 8px; border-radius: 5px; margin-bottom: 15px; font-size: 11px;">
                <strong>Score Bar Legend:</strong> 
                <span style="color: #28a745;">🟢 Entails</span> | 
                <span style="color: #dc3545;">🔴 Contradicts</span> | 
                <span style="color: #6c757d;">⚪ Neutral</span> | 
                <span style="color: #007bff;">🔵 Max Score (highlighted)</span>
        </div>
"""

        if top_sentences:
            for i, (primary_score, candidate_data, model_scores) in enumerate(top_sentences):
                # Determine primary model for ranking
                primary_model = "HF (RoBERTa)"
                if "OpenAI (gpt-3.5-turbo)" in model_scores:
                    primary_model = "OpenAI (gpt-3.5-turbo)"
                elif "Phi (phi3:3.8b)" in model_scores:
                    primary_model = "Phi (phi3:3.8b)"
                elif "LLM (llama3.2:3b)" in model_scores:
                    primary_model = "LLM (llama3.2:3b)"

                html_content += f"""
            <div class="sentence-card">
                <div class="sentence-meta">
                    <strong>Rank {i+1}</strong> |
                    <span style="color: #007bff;">Primary Score: {primary_score:.3f}</span>
                    <span style='color: #6c757d; font-size: 12px;'>(Ranked by: {primary_model})</span>
            </div>
                <div class="sentence-text">
                    {candidate_data['text']}
            </div>
                <table class="scores-table">
                    <thead>
                        <tr>
                            <th>Hypothesis</th>"""

                # Add column headers for each available model - get from rescorers list in target data
                model_names = analysis_results[target_name].get("rescorers", [])
                for model_name in model_names:
                    html_content += f"""
                            <th>{model_name}</th>"""

                html_content += """
                        </tr>
                    </thead>
                    <tbody>"""

                # Get hypothesis scores from the candidate data
                # We need to reconstruct hypotheses from the target config and show individual scores
                target_config = ANALYSIS_CONFIG.get(target_name, {})
                target_hypotheses = target_config.get("hypotheses", {})

                # Create rows for each hypothesis
                for stance, hypotheses_list in [("pro", target_hypotheses.get("pro", [])), ("anti", target_hypotheses.get("anti", []))]:
                    for hypothesis_idx, hypothesis_text in enumerate(hypotheses_list):
                        stance_class = f"{stance}-hypothesis"
                    
                        html_content += f"""
                            <tr class="{stance_class}">
                                <td class="hypothesis-cell"><strong>[{stance.upper()}]</strong> {hypothesis_text}</td>"""

                        # Add scores for each model
                        for model_name in model_names:
                            # Get hypothesis scores for this model from the new rescorers structure
                            rescorers = candidate_data.get("rescorers", {})
                            model_hypothesis_scores = None
                            
                            if model_name in rescorers:
                                model_hypothesis_scores = rescorers[model_name].get("hypothesis_scores", {})

                            if model_hypothesis_scores and stance in model_hypothesis_scores:
                                stance_scores = model_hypothesis_scores[stance]
                                if hypothesis_idx < len(stance_scores):
                                    hypothesis_scores = stance_scores[hypothesis_idx]
                                    entails = hypothesis_scores.get("entails", 0.0)
                                    neutral = hypothesis_scores.get("neutral", 0.0) 
                                    contradicts = hypothesis_scores.get("contradicts", 0.0)
                                    
                                    # Create the formatted score triple
                                    score_triple = create_score_triple(entails, neutral, contradicts)
                                    html_content += f"""
                        <td class="score-triple">{score_triple}</td>"""
                                else:
                                    html_content += f"""
                        <td class="score-triple">-</td>"""
                            else:
                                html_content += f"""
                        <td class="score-triple">-</td>"""

                        html_content += """
                        </tr>"""

                html_content += """
                    </tbody>
                </table>
                """

                # Note: Detailed hypothesis scores section removed for simplicity
                # The main table above shows all model scores with horizontal bars

                html_content += """
        </div>
"""
        else:
            html_content += '<p style="color: #6c757d; font-style: italic;">No statements found</p>'
        
        html_content += """
        </div>
    </div>
"""
    
    html_content += f"""
    <div class="footer">
        <p>Report generated by Coal Overcapacity Analysis Script • {timestamp}</p>
        <p>Analysis method: Multi-model NLI-based stance detection with multiple hypotheses</p>
        <p>Models: HF (RoBERTa), LLM (llama3.2:3b), Phi (phi3:3.8b), OpenAI (gpt-3.5-turbo)</p>
        <p><strong>Cache Management:</strong> All LLM queries are cached for faster subsequent runs. Cache location: cache/llm/</p>
        <p><strong>Cache Details:</strong> {cache_stats['total_entries']} total entries, {cache_stats['total_size_mb']:.1f} MB total size</p>
        {f"<p><strong>Cache Range:</strong> {cache_stats['oldest_entry']} to {cache_stats['newest_entry']}</p>" if cache_stats['oldest_entry'] and cache_stats['newest_entry'] else ""}
    </div>
</body>
</html>
"""

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"HTML report generated: {output_path}")

def create_score_bar(score: float, score_type: str, is_max: bool = False) -> str:
    """Create HTML for a score bar with numerical value."""
    width_percent = min(100, max(2, score * 100))  # Ensure minimum 2% width for visibility
    bar_class = f"score-bar {score_type}"
    if is_max:
        bar_class += " max"
    
    return f"""
    <div class="score-bar-container">
        <div class="score-bar {bar_class}" style="width: {width_percent}%;">
            <div class="score-text">{score:.3f}</div>
        </div>
    </div>"""

def create_score_triple(entails: float, neutral: float, contradicts: float) -> str:
    """Create HTML for entails | neutral | contradicts score triple with color coding."""
    entails_html = f'<span class="entails-score">{entails:.3f}</span>'
    neutral_html = f'<span class="neutral-score">{neutral:.3f}</span>'
    contradicts_html = f'<span class="contradicts-score">{contradicts:.3f}</span>'
    
    return f"{entails_html} | {neutral_html} | {contradicts_html}"

def print_cache_stats():
    """Print cache statistics."""
    cache_manager = get_cache_manager()
    stats = cache_manager.get_cache_stats()
    
    logger.info("📊 Cache Statistics:")
    logger.info(f"  Total entries: {stats['total_entries']}")
    logger.info(f"  Total size: {stats['total_size_mb']:.1f} MB")
    
    if stats['models']:
        logger.info("  Models in cache:")
        for model, model_stats in stats['models'].items():
            logger.info(f"    {model}: {model_stats['entries']} entries ({model_stats['size_mb']:.1f} MB)")
    
    if stats['oldest_entry'] and stats['newest_entry']:
        logger.info(f"  Cache range: {stats['oldest_entry']} to {stats['newest_entry']}")

def main():
    """Main analysis function."""
    logger.info("🚀 Starting Coal Overcapacity Analysis")
    
    # Print initial cache stats
    print_cache_stats()
    
    # Create embedded corpus
    logger.info("Loading embedded corpus: mediacloud_india_coal")
    try:
        embedded_corpus = create_embedded_corpus("mediacloud_india_coal")
        logger.info("✅ Embedded corpus loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load embedded corpus: {e}")
        return
    
    # Create retriever
    logger.info("Setting up index retriever")
    try:
        retriever = create_retriever(embedded_corpus)
        logger.info("✅ Retriever initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize retriever: {e}")
        return
    
    # Create all NLI scorers
    logger.info("Setting up all NLI scorers")
    
    try:
        scorers = {
            "HF (RoBERTa)": NLIHFScorer(name="nli_hf"),
            "LLM (llama3.2:3b)": NLILLMScorer(name="nli_llm", config=LLMScorerConfig(verbose=True, model="llama3.2:3b")),
            "Phi (phi3:3.8b)": NLILLMScorer(name="nli_phi", config=LLMScorerConfig(verbose=True, model="phi3:3.8b")),
                "Mistral (7b-instruct)": NLILLMScorer(name="nli_mistral", config=LLMScorerConfig(verbose=True, model="mistral:7b")),
            "OpenAI (gpt-3.5-turbo)": NLIOpenAIScorer(name="nli_openai_35", config=NLIOpenAIConfig(verbose=True, model="gpt-3.5-turbo")),
            "OpenAI (gpt-4o-mini)": NLIOpenAIScorer(name="nli_openai_4mini", config=NLIOpenAIConfig(verbose=True, model="gpt-4o-mini"))
        }
        
        logger.info(f"✅ All {len(scorers)} NLI scorers initialized successfully: {list(scorers.keys())}")
    except Exception as e:
        logger.error(f"❌ Failed to initialize NLI scorers: {e}")
        return
    
    # Analyze each target
    analysis_results = {}
    for target_name, target_config in ANALYSIS_CONFIG.items():
        target_results = analyze_target(target_name, target_config, scorers, retriever)
        analysis_results[target_name] = target_results
    
    # Save results
    json_output_path = project_root / "results" / "coal_overcapacity_analysis" / "report.json"
    save_results(analysis_results, json_output_path)
    
    # Generate HTML report
    html_output_path = project_root / "results" / "coal_overcapacity_analysis" / "report.html"
    generate_html_report(analysis_results, html_output_path)
    
    # Print summary
    logger.info("📊 Analysis Summary:")
    for target_name in ANALYSIS_CONFIG.keys():
        if target_name in analysis_results:
            results = analysis_results[target_name]
            logger.info(f"  Target '{target_name}':")
            logger.info(f"    Queries: {results['total_queries']}")
            logger.info(f"    Candidates collected: {results['total_candidates_collected']}")
            logger.info(f"    Unique candidates: {results['unique_candidates']}")
            logger.info(f"    Top candidates rescored: {results['top_candidates_rescored']}")
            logger.info(f"    Rescorers used: {', '.join(results['rescorers'])}")

    # Print final cache stats
    logger.info("📊 Final Cache Statistics:")
    print_cache_stats()

    logger.info(f"🎉 Analysis completed!")
    logger.info(f"  JSON results: {json_output_path}")
    logger.info(f"  HTML report: {html_output_path}")

if __name__ == "__main__":
    main()
