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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from efi_corpus.embedded import EmbeddedCorpus
from efi_core.retrieval.retriever_index import RetrieverIndex
from efi_core.types import ChunkerSpec, EmbedderSpec, Candidate
from efi_analyser.chunkers import TextChunker
from efi_analyser.embedders import SentenceTransformerEmbedder
from efi_analyser.scorers import NLIHFScorer, NLILLMScorer, LLMScorerConfig

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

def rescore_candidates_with_nli(candidates: List[Candidate], hypotheses: Dict[str, List[str]], nli_scorer: NLIHFScorer) -> Dict[str, Any]:
    """Rescore candidates with NLI hypotheses to determine stance."""
    results = {}
    
    for i, candidate in enumerate(candidates):
        logger.info(f"Rescoring candidate {i+1}/{len(candidates)}")
        candidate_results = {
            "text": candidate.text,
            "item_id": candidate.item_id,
            "ann_score": candidate.ann_score,
            "hypothesis_scores": {}
        }
        
        # Score each hypothesis category (pro/anti)
        for stance, hypothesis_list in hypotheses.items():
            category_scores = []
            
            for hypothesis in hypothesis_list:
                # Score the hypothesis against the candidate text
                # NLI: premise (document hit) entails/contradicts hypothesis (config statement)
                scores = nli_scorer.batch_score([candidate.text], [hypothesis])[0]
                
                category_scores.append({
                    "hypothesis": hypothesis,
                    "entails": scores.get("entails", 0.0),
                    "contradicts": scores.get("contradicts", 0.0),
                    "neutral": scores.get("neutral", 0.0)
                })
            
            candidate_results["hypothesis_scores"][stance] = category_scores
        
        # Calculate aggregate scores for this candidate
        candidate_results["aggregate_scores"] = calculate_aggregate_scores(candidate_results["hypothesis_scores"])
        
        results[candidate.item_id] = candidate_results
    
    return results

def select_top_candidates_for_llm_rescoring(rescored_results: Dict[str, Any], top_k: int = 2) -> List[Candidate]:
    """Select top candidates for LLM rescoring based on aggregate scores."""
    # Create a list of (item_id, candidate_data) tuples
    candidates_with_scores = []

    for item_id, candidate_data in rescored_results.items():
        aggregate_scores = candidate_data.get("aggregate_scores", {})
        # Use the higher of entails or contradicts as the overall score
        overall_score = max(aggregate_scores.get("entails", 0.0), aggregate_scores.get("contradicts", 0.0))

        # Create a candidate object for rescoring
        from efi_core.types import Candidate
        candidate = Candidate(
            text=candidate_data["text"],
            item_id=item_id,
            ann_score=overall_score,  # Use overall score for ranking
            meta={}  # Empty meta since we don't need it for rescoring
        )
        candidates_with_scores.append((overall_score, candidate))

    # Sort by score (descending) and take top_k
    candidates_with_scores.sort(key=lambda x: x[0], reverse=True)
    top_candidates = [candidate for _, candidate in candidates_with_scores[:top_k]]

    logger.info(f"Selected {len(top_candidates)} candidates for LLM rescoring")
    return top_candidates

def rescore_candidates_with_llm(candidates: List[Candidate], hypotheses: Dict[str, List[str]], nli_llm_scorer: NLILLMScorer) -> Dict[str, Any]:
    """Rescore candidates with LLM NLI hypotheses."""
    results = {}

    for i, candidate in enumerate(candidates):
        logger.info(f"LLM rescoring candidate {i+1}/{len(candidates)}")
        candidate_results = {
            "text": candidate.text,
            "item_id": candidate.item_id,
            "ann_score": candidate.ann_score,
            "hypothesis_scores": {}
        }

        # Score each hypothesis category (pro/anti)
        for stance, hypothesis_list in hypotheses.items():
            category_scores = []

            for hypothesis in hypothesis_list:
                # Score the hypothesis against the candidate text using LLM
                # NLI: premise entails/is neutral to/contradicts hypothesis
                scores = nli_llm_scorer.batch_score([candidate.text], [hypothesis])[0]

                logger.info(f"LLM scores for hypothesis '{hypothesis[:50]}...': {scores}")

                category_scores.append({
                    "hypothesis": hypothesis,
                    "entails": scores.get("entails", 0.0),
                    "contradicts": scores.get("contradicts", 0.0),
                    "neutral": scores.get("neutral", 0.0)
                })

            candidate_results["hypothesis_scores"][stance] = category_scores

        # Calculate aggregate scores for this candidate (LLM version)
        candidate_results["aggregate_scores"] = calculate_aggregate_scores(candidate_results["hypothesis_scores"])

        results[candidate.item_id] = candidate_results

    return results

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

def analyze_target(target_name: str, target_config: Dict[str, Any], retriever: RetrieverIndex, nli_scorer: NLIHFScorer, nli_llm_scorer: NLILLMScorer) -> Dict[str, Any]:
    """Analyze a single target (build_more_coal or coal_overcapacity)."""
    logger.info(f"🔍 Analyzing target: {target_name}")
    
    # Collect candidates from all queries
    all_candidates = []
    for query in target_config["queries"]:
        candidates = retrieve_statements_for_query(retriever, query, top_k=10)
        all_candidates.append(candidates)
    
    # Get unique candidates
    unique_candidates = collect_unique_candidates(all_candidates)
    
    # Rescore with NLI hypotheses
    rescored_results = rescore_candidates_with_nli(unique_candidates, target_config["hypotheses"], nli_scorer)

    # Apply level 2 LLM rescoring to top 10 sentences
    logger.info(f"Applying LLM rescoring to top 10 sentences for {target_name}")
    top_candidates = select_top_candidates_for_llm_rescoring(rescored_results, top_k=10)
    llm_rescored_results = rescore_candidates_with_llm(top_candidates, target_config["hypotheses"], nli_llm_scorer)

    # Update rescored_results with LLM results for top candidates
    logger.info(f"Updating results with LLM scores for {len(llm_rescored_results)} candidates")
    for item_id, llm_scores in llm_rescored_results.items():
        logger.info(f"Processing LLM result for item_id: {item_id}")
        if item_id in rescored_results:
            logger.info(f"Found matching item_id in rescored_results: {item_id}")
            # Keep both HF and LLM aggregate scores
            rescored_results[item_id]["hf_aggregate_scores"] = rescored_results[item_id]["aggregate_scores"]
            rescored_results[item_id]["llm_aggregate_scores"] = llm_scores["aggregate_scores"]
            rescored_results[item_id]["llm_rescored"] = True
            rescored_results[item_id]["llm_hypothesis_scores"] = llm_scores["hypothesis_scores"]
            # Update the main aggregate_scores to LLM for ranking/reporting
            rescored_results[item_id]["aggregate_scores"] = llm_scores["aggregate_scores"]
            logger.info(f"Updated rescored_results for {item_id}")
        else:
            logger.info(f"item_id {item_id} not found in rescored_results")
            logger.info(f"Available item_ids: {list(rescored_results.keys())[:5]}...")  # Show first 5

    # Calculate target-level summary
    summary = {
        "target_name": target_name,
        "total_queries": len(target_config["queries"]),
        "total_candidates_collected": sum(len(c) for c in all_candidates),
        "unique_candidates": len(unique_candidates),
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
        }}

        .scores-table th, .scores-table td {{
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}

        .scores-table th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}

        .score-cell {{
            font-family: monospace;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Coal Overcapacity Analysis Report</h1>
        <p>NLI-based stance detection for coal overcapacity discourse</p>
        <p><strong>Generated:</strong> {timestamp}</p>
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

        # Get top pro and anti sentences based on proper NLI scoring
        pro_sentences = []
        anti_sentences = []

        for item_id, candidate_data in candidates.items():
            hypothesis_scores = candidate_data.get("hypothesis_scores", {})

            # For pro sentences: use the aggregate entails score (already calculated)
            aggregate_scores = candidate_data.get("aggregate_scores", {})
            pro_score = aggregate_scores.get("entails", 0.0)
            pro_sentences.append((pro_score, candidate_data))

            # For anti sentences: use the highest contradicts score (already inverted in aggregate_scores)
            aggregate_scores = candidate_data.get("aggregate_scores", {})
            anti_score = aggregate_scores.get("contradicts", 0.0)
            anti_sentences.append((anti_score, candidate_data))

        # Sort and get top 10 for each
        pro_sentences.sort(key=lambda x: x[0], reverse=True)
        anti_sentences.sort(key=lambda x: x[0], reverse=True)

        top_pro = pro_sentences[:10]
        top_anti = anti_sentences[:10]

        html_content += f"""
    <div class="target-section">
        <div class="target-header">
            <div>
                <h3 class="target-title">Target: {target_name.replace('_', ' ').title()}</h3>
                <div class="target-stats">
                    {target_results.get('total_queries', 0)} queries |
                    {target_results.get('unique_candidates', 0)} unique statements
                </div>
            </div>
            <div class="score-badge">NLI Analysis</div>
        </div>

        <div class="stance-section">
            <h4 class="stance-title">🟢 Most Pro Sentences (Top 10)</h4>
            <p style="font-size: 12px; color: #6c757d; margin-bottom: 15px;">
                <em>Scored by: Pro-entails + Anti-contradicts</em>
            </p>
"""

        if top_pro:
            for i, (score, candidate_data) in enumerate(top_pro):
                aggregate_scores = candidate_data.get("aggregate_scores", {})
                hf_scores = candidate_data.get("hf_aggregate_scores", {})
                llm_scores = candidate_data.get("llm_aggregate_scores", {})
                is_llm_rescored = candidate_data.get("llm_rescored", False)

                # Debug logging
                logger.info(f"Pro candidate {i+1}: llm_rescored={is_llm_rescored}, hf_scores_keys={list(hf_scores.keys())}, llm_scores_keys={list(llm_scores.keys())}")

                html_content += f"""
            <div class="sentence-card">
                <div class="sentence-meta">
                    <strong>Rank {i+1}</strong> |
                    <span style="color: #28a745;">Pro Score: {score:.3f}</span>
                    {"<span style='color: #007bff; font-size: 12px;'>(LLM Rescored)</span>" if is_llm_rescored else ""}
                </div>
                <div class="sentence-text">
                    {candidate_data['text']}
                </div>
                <table class="scores-table">
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Entails</th>
                            <th>Contradicts</th>
                            <th>Neutral</th>
                        </tr>
                    </thead>
                    <tbody>"""

                # Always show HF Aggregate scores
                html_content += f"""
                        <tr>
                            <td><strong>HF Aggregate</strong></td>
                            <td class="score-cell">{aggregate_scores.get('entails', 0):.3f}</td>
                            <td class="score-cell">{aggregate_scores.get('contradicts', 0):.3f}</td>
                            <td class="score-cell">{aggregate_scores.get('neutral', 0):.3f}</td>
                        </tr>"""

                # Show LLM scores if available
                if is_llm_rescored:
                    html_content += f"""
                        <tr>
                            <td><strong>LLM Aggregate</strong></td>
                            <td class="score-cell">{llm_scores.get('entails', 0):.3f}</td>
                            <td class="score-cell">{llm_scores.get('contradicts', 0):.3f}</td>
                            <td class="score-cell">{llm_scores.get('neutral', 0):.3f}</td>
                        </tr>"""

                html_content += """
                    </tbody>
                </table>
        </div>
"""
        else:
            html_content += '<p style="color: #6c757d; font-style: italic;">No pro sentences found</p>'

        html_content += """
        </div>

        <div class="stance-section anti">
            <h4 class="stance-title anti">🔴 Most Anti Sentences (Top 10)</h4>
            <p style="font-size: 12px; color: #6c757d; margin-bottom: 15px;">
                <em>Scored by: Aggregate contradicts score (inverted for anti hypotheses)</em>
            </p>
"""

        if top_anti:
            for i, (score, candidate_data) in enumerate(top_anti):
                aggregate_scores = candidate_data.get("aggregate_scores", {})
                hf_scores = candidate_data.get("hf_aggregate_scores", {})
                llm_scores = candidate_data.get("llm_aggregate_scores", {})
                is_llm_rescored = candidate_data.get("llm_rescored", False)

                # Debug logging
                logger.info(f"Anti candidate {i+1}: llm_rescored={is_llm_rescored}, hf_scores_keys={list(hf_scores.keys())}, llm_scores_keys={list(llm_scores.keys())}")

                html_content += f"""
            <div class="sentence-card">
                <div class="sentence-meta">
                    <strong>Rank {i+1}</strong> |
                    <span style="color: #dc3545;">Anti Score: {score:.3f}</span>
                    {"<span style='color: #007bff; font-size: 12px;'>(LLM Rescored)</span>" if is_llm_rescored else ""}
            </div>
                <div class="sentence-text">
                    {candidate_data['text']}
            </div>
                <table class="scores-table">
                    <thead>
                        <tr>
                            <th>Metric</th>
                            <th>Entails</th>
                            <th>Contradicts</th>
                            <th>Neutral</th>
                        </tr>
                    </thead>
                    <tbody>"""

                # Always show HF Aggregate scores
                html_content += f"""
                        <tr>
                            <td><strong>HF Aggregate</strong></td>
                            <td class="score-cell">{aggregate_scores.get('entails', 0):.3f}</td>
                            <td class="score-cell">{aggregate_scores.get('contradicts', 0):.3f}</td>
                            <td class="score-cell">{aggregate_scores.get('neutral', 0):.3f}</td>
                        </tr>"""

                # Show LLM scores if available
                if is_llm_rescored:
                    html_content += f"""
                        <tr>
                            <td><strong>LLM Aggregate</strong></td>
                            <td class="score-cell">{llm_scores.get('entails', 0):.3f}</td>
                            <td class="score-cell">{llm_scores.get('contradicts', 0):.3f}</td>
                            <td class="score-cell">{llm_scores.get('neutral', 0):.3f}</td>
                        </tr>"""

                html_content += """
                    </tbody>
                </table>
        </div>
"""
        else:
            html_content += '<p style="color: #6c757d; font-style: italic;">No anti sentences found</p>'
        
        html_content += """
        </div>
    </div>
"""
    
    html_content += f"""
    <div class="footer">
        <p>Report generated by Coal Overcapacity Analysis Script • {timestamp}</p>
        <p>Analysis method: NLI-based stance detection with multiple hypotheses</p>
    </div>
</body>
</html>
"""

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"HTML report generated: {output_path}")

def main():
    """Main analysis function."""
    logger.info("🚀 Starting Coal Overcapacity Analysis")
    
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
    
    # Create NLI scorers
    logger.info("Setting up NLI scorers")
    try:
        nli_scorer = NLIHFScorer(name="nli_hf")
        llm_config = LLMScorerConfig(verbose=True, model="llama3.2:3b")
        nli_llm_scorer = NLILLMScorer(name="nli_llm", config=llm_config)
        logger.info("✅ NLI scorers initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize NLI scorers: {e}")
        return
    
    # Analyze each target
    analysis_results = {}
    for target_name, target_config in ANALYSIS_CONFIG.items():
        target_results = analyze_target(target_name, target_config, retriever, nli_scorer, nli_llm_scorer)
        analysis_results[target_name] = target_results
    
    # Save results
    json_output_path = project_root / "results" / "coal_overcapacity_analysis.json"
    save_results(analysis_results, json_output_path)
    
    # Generate HTML report
    html_output_path = project_root / "results" / "coal_overcapacity_analysis.html"
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

    logger.info(f"🎉 Analysis completed!")
    logger.info(f"  JSON results: {json_output_path}")
    logger.info(f"  HTML report: {html_output_path}")

if __name__ == "__main__":
    main()
