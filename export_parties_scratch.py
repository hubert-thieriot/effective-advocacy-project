#!/usr/bin/env python3
"""Scratch script to export parties with non-zero weights."""

import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple
import csv

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import json
import yaml

from efi_analyser.frames.types import FrameAssignments
from efi_analyser.frames.plotting.party_bars import _load_party_names, _load_party_families
from efi_corpus.corpus_handle import CorpusHandle



def main():
    # Default results directory - adjust as needed
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        # Try to find a recent results directory
        results_base = project_root / "results" / "narrative_framing"
        if results_base.exists():
            # Get most recent directory
            dirs = sorted(results_base.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
            if dirs:
                results_dir = dirs[0]
                print(f"Using results directory: {results_dir}")
            else:
                print("No results directory found. Please provide path as argument.")
                sys.exit(1)
        else:
            print("No results directory found. Please provide path as argument.")
            sys.exit(1)
    
    # Load assignments
    assignments_path = results_dir / "frame_assignments.json"
    if not assignments_path.exists():
        print(f"❌ Assignments file not found: {assignments_path}")
        sys.exit(1)
    
    print(f"📥 Loading assignments from {assignments_path}...")
    assignments = FrameAssignments.load(assignments_path)
    print(f"✅ Loaded {len(assignments)} assignments")
    
    # Load corpus to get metadata
    print("📥 Loading corpus...")
    # Get corpus name from config
    config_dir = results_dir / "configs"
    corpus_name = None
    corpora_root = project_root / "corpora"
    
    if config_dir.exists():
        for config_file in sorted(config_dir.glob("*.yaml"), reverse=True):
            try:
                config = yaml.safe_load(config_file.read_text())
                corpus_name = config.get("corpus")
                if isinstance(corpus_name, list):
                    corpus_name = corpus_name[0] if corpus_name else None
                if corpus_name:
                    break
            except Exception:
                continue
    
    if not corpus_name:
        print("⚠️ Could not determine corpus name from config")
        sys.exit(1)
    
    corpus_path = corpora_root / corpus_name
    if not corpus_path.exists():
        print(f"⚠️ Corpus not found at {corpus_path}")
        sys.exit(1)
    
    corpus = CorpusHandle(corpus_path=corpus_path, read_only=True)
    print(f"✅ Loaded corpus: {corpus_name}")
    
    # Calculate weights per party
    doc_scores: Dict[str, Tuple[float, float]] = {}
    doc_ids_seen: set = set()
    
    for a in assignments:
        doc_id = a.metadata.get("doc_id", "") or a.metadata.get("global_doc_id", "")
        if doc_id:
            doc_ids_seen.add(doc_id)
            weight = len(a.passage_text)
            probs = a.probabilities
            frame_sum = sum(probs.values()) * weight
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = (0.0, 0.0)
            current = doc_scores[doc_id]
            doc_scores[doc_id] = (current[0] + frame_sum, current[1] + weight)
    
    # Get metadata from corpus for all doc_ids
    print("📥 Loading metadata from corpus...")
    corpus_index = {}
    for doc_id in doc_ids_seen:
        # Parse doc_id - might be "corpus_name::local_id" or just "local_id"
        if "::" in doc_id:
            _, local_id = doc_id.split("::", 1)
        else:
            local_id = doc_id
        
        meta = corpus.get_metadata(local_id) or {}
        corpus_index[doc_id] = meta
    
    print(f"✅ Loaded metadata for {len(corpus_index)} documents")
    
    # Aggregate by country and party_id
    country_party_scores: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    party_ids_seen: set = set()
    party_id_to_country: Dict[str, str] = {}
    
    for doc_id, (frame_sum, total_weight) in doc_scores.items():
        meta = corpus_index.get(doc_id, {})
        extra = meta.get("extra", {})
        
        # Get country and party_id from extra field
        country = extra.get("country_name", "Unknown")
        party_id = extra.get("party_id", "Unknown")
        
        if party_id and party_id != "Unknown":
            score = frame_sum / total_weight if total_weight > 0 else 0.0
            # Include all parties with assignments, even if score is 0
            # (score of 0 might mean no frames matched, but party still has data)
            country_party_scores[country][party_id] += score
            party_ids_seen.add(party_id)
            if party_id not in party_id_to_country:
                party_id_to_country[party_id] = country
    
    print(f"✅ Found {len(party_ids_seen)} parties with non-zero weights")
    
    # Load party data
    print("📥 Loading party names and families...")
    party_names_data = _load_party_names()
    party_families = _load_party_families()
    
    # Prepare rows for CSV
    rows = []
    for party_id in sorted(party_ids_seen):
        country = party_id_to_country.get(party_id, "Unknown")
        
        # Get party name data
        party_data = party_names_data.get(party_id, {})
        name = party_data.get("name", "")
        name_english = party_data.get("name_english", "").strip()
        abbrev = party_data.get("abbrev", "")
        
        # If name_english is "NA" or empty, use original name
        if not name_english or name_english.upper() == "NA":
            name_english = name
        
        # Get family from families CSV
        family = party_families.get(party_id, "")
        
        rows.append({
            "party_id": party_id,
            "country": country,
            "abbrev": abbrev,
            "name": name,
            "name_english": name_english,
            "family": family,
        })
    
    # Write CSV
    output_path = project_root / "parties_nonzero_weight.csv"
    with output_path.open('w', encoding='utf-8', newline='') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=["party_id", "country", "abbrev", "name", "name_english", "family"])
            writer.writeheader()
            writer.writerows(rows)
            print(f"✅ Exported {len(rows)} parties to {output_path}")
        else:
            print(f"⚠️ No parties with non-zero weights found")


if __name__ == "__main__":
    main()

