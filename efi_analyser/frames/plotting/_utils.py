"""Shared utilities for plotting modules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import yaml


def load_corpus_index(results_dir: Path, corpus_name: str = None) -> Dict[str, dict]:
    """Load corpus index to map doc_id to metadata.
    
    Args:
        results_dir: Path to results directory
        corpus_name: Optional corpus name (will be inferred from config if not provided)
    
    Returns:
        Dict mapping doc_id to metadata
    """
    # Try to find the corpus from the config
    if not corpus_name:
        config_dir = results_dir / "configs"
        if config_dir.exists():
            for config_file in sorted(config_dir.glob("*.yaml"), reverse=True):
                try:
                    config = yaml.safe_load(config_file.read_text())
                    corpus_name = config.get("corpus")
                    if corpus_name:
                        break
                except Exception:
                    continue
    
    if not corpus_name:
        # Try to infer from results folder name
        corpus_name = results_dir.name.replace("_animalwelfare", "").replace("_v2", "").replace("_", "_")
    
    # Look for corpus index
    corpus_paths = [
        Path("corpora") / corpus_name / "index.jsonl",
        Path("corpora") / f"{corpus_name}" / "index.jsonl",
    ]
    
    for corpus_path in corpus_paths:
        if corpus_path.exists():
            index = {}
            for line in corpus_path.read_text().splitlines():
                if line.strip():
                    doc = json.loads(line)
                    index[doc["id"]] = doc
            return index
    
    return {}


def load_document_aggregates(results_dir: Path) -> list[dict]:
    """Load document aggregates from disk.
    
    Args:
        results_dir: Path to results directory
        
    Returns:
        List of document aggregate dicts
    """
    aggregates_path = results_dir / "aggregates" / "documents_weighted.json"
    if not aggregates_path.exists():
        # Fallback: try to aggregate from frame_assignments.json
        assignments_path = results_dir / "frame_assignments.json"
        if assignments_path.exists():
            assignments = json.loads(assignments_path.read_text())
            # Simple aggregation by document
            doc_scores = {}
            for a in assignments:
                passage_id = a.get("passage_id", "")
                doc_id = passage_id.split(":")[0] if ":" in passage_id else passage_id
                text = a.get("passage_text", "")
                weight = len(text)
                probs = a.get("probabilities", {})
                
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {"frame_scores": {}, "total_weight": 0}
                
                for frame, prob in probs.items():
                    if frame not in doc_scores[doc_id]["frame_scores"]:
                        doc_scores[doc_id]["frame_scores"][frame] = 0
                    doc_scores[doc_id]["frame_scores"][frame] += prob * weight
                doc_scores[doc_id]["total_weight"] += weight
            
            return [
                {"doc_id": doc_id, **data}
                for doc_id, data in doc_scores.items()
            ]
        return []
    
    return json.loads(aggregates_path.read_text())

