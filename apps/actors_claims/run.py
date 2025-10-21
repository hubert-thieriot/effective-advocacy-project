#!/usr/bin/env python3
"""Run actors/claims/attributions extraction over a corpus directory."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from apps.actors_claims.config import AppConfig, load_config
from efi_corpus.corpus_handle import CorpusHandle
from efi_analyser.actors_claims import (
    Document as AC_Document,
    SimpleCombinedExtractor,
    SpacyCombinedExtractor,
    HfCombinedExtractor,
)
from apps.actors_claims.report import write_html_report


def _resolve_extractor(cfg: AppConfig):
    stack = cfg.extractor.stack.lower().strip()
    if stack == "simple":
        return SimpleCombinedExtractor()
    if stack == "spacy":
        return SpacyCombinedExtractor()
    if stack == "hf":
        # Device selection: None->auto (GPU if available), else explicit
        device = cfg.extractor.device
        return HfCombinedExtractor(model_name=cfg.extractor.hf_model, device=device)
    raise ValueError(f"Unknown extractor stack: {cfg.extractor.stack}")


def _safe_title(meta: Dict) -> str:
    t = meta.get("title") or meta.get("source_title") or ""
    return str(t)[:300]


def _load_existing_counts(results_dir: Path) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], int, int, str, str, str, str]:
    """Load actors/claims/atts counts and paths from existing files in results_dir.

    Returns: (actors_list, claims_list, atts_count, processed_docs, jsonl_path, actors_csv, claims_csv, atts_csv)
    """
    jsonl_path = str(results_dir / "extractions.jsonl")
    actors_csv = str(results_dir / "actors.csv")
    claims_csv = str(results_dir / "claims.csv")
    atts_csv = str(results_dir / "attributions.csv")

    actors: List[Tuple[str, int]] = []
    claims: List[Tuple[str, int]] = []
    atts_count = 0
    processed_docs = 0

    # Load actors
    p = Path(actors_csv)
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    try:
                        actors.append((row[0], int(row[1])))
                    except ValueError:
                        continue
    # Load claims
    p = Path(claims_csv)
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    try:
                        claims.append((row[0], int(row[1])))
                    except ValueError:
                        continue
    # Count attributions
    p = Path(atts_csv)
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            _ = next(reader, None)
            for _ in reader:
                atts_count += 1
    # Count processed docs from JSONL
    p = Path(jsonl_path)
    if p.exists():
        doc_ids = set()
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    did = str(rec.get("doc_id", ""))
                    if did:
                        doc_ids.add(did)
                except json.JSONDecodeError:
                    continue
        processed_docs = len(doc_ids)

    # Sort top items for report (limit there as well)
    actors.sort(key=lambda kv: kv[1], reverse=True)
    claims.sort(key=lambda kv: kv[1], reverse=True)
    return actors, claims, atts_count, processed_docs, jsonl_path, actors_csv, claims_csv, atts_csv


def run_from_config(cfg: AppConfig) -> Dict[str, Path]:
    corpus_dir = cfg.corpus_dir
    handle = CorpusHandle(corpus_dir, read_only=True)
    doc_ids = handle.list_ids()
    if not doc_ids:
        raise RuntimeError(f"No documents found under {corpus_dir}")

    # Sample documents
    rng = random.Random(cfg.seed)
    if cfg.random_sample:
        rng.shuffle(doc_ids)
    # else keep existing order
    if cfg.limit and cfg.limit > 0:
        doc_ids = doc_ids[: cfg.limit]

    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = results_dir / "extractions.jsonl"
    actors_csv_path = results_dir / "actors.csv"
    claims_csv_path = results_dir / "claims.csv"
    atts_csv_path = results_dir / "attributions.csv"
    html_path = results_dir / "report.html"

    # Report-only mode: rebuild report from existing outputs
    if cfg.report_only:
        results_dir = Path(cfg.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        html_path = results_dir / "report.html"
        actors_list, claims_list, atts_count, processed_docs, jsonl_path, actors_csv, claims_csv, atts_csv = _load_existing_counts(results_dir)
        write_html_report(
            html_path,
            corpus_dir=str(corpus_dir),
            total_docs=processed_docs,
            actors=actors_list[:50],
            claims=claims_list[:50],
            atts_count=atts_count,
            jsonl_path=jsonl_path,
            actors_csv=actors_csv,
            claims_csv=claims_csv,
            atts_csv=atts_csv,
        )
        return {
            "jsonl": results_dir / "extractions.jsonl",
            "actors_csv": results_dir / "actors.csv",
            "claims_csv": results_dir / "claims.csv",
            "atts_csv": results_dir / "attributions.csv",
            "html": html_path,
        }

    extractor = _resolve_extractor(cfg)

    # Collect aggregated counts
    actor_counts: Dict[str, int] = {}
    claim_counts: Dict[str, int] = {}
    atts_rows: List[List[str]] = []

    if cfg.save_jsonl and cfg.overwrite:
        jsonl_path.unlink(missing_ok=True)

    # Normalize keyword filter for doc-level inclusion
    normalized_keywords = None
    if cfg.filter_keywords:
        normalized = [str(k).strip().lower() for k in cfg.filter_keywords if str(k).strip()]
        normalized_keywords = normalized or None

    processed_count = 0
    with open(jsonl_path, "a" if cfg.save_jsonl else "w", encoding="utf-8") as jf:
        for doc_id in tqdm(doc_ids, desc="Extracting actors/claims"):
            try:
                meta = handle.get_metadata(doc_id) or {}
                text = handle.get_text(doc_id)
                # Apply optional keyword gate at document level
                if normalized_keywords is not None:
                    lt = text.lower()
                    if not any(kw in lt for kw in normalized_keywords):
                        continue
                # Default to configured language if missing
                language = meta.get("language") or cfg.language
                ac_doc = AC_Document(doc_id=doc_id, language=str(language), text=text)
                res = extractor.run(ac_doc)

                record = {
                    "doc_id": doc_id,
                    "title": _safe_title(meta),
                    "url": meta.get("uri") or meta.get("url"),
                    "published_at": meta.get("published_at"),
                    "mentions": [asdict(m) for m in res.mentions],
                    "claims": [asdict(c) for c in res.claims],
                    "attributions": [asdict(a) for a in res.attributions],
                }
                if cfg.save_jsonl:
                    jf.write(json.dumps(record, ensure_ascii=False) + "\n")

                # Aggregate
                for a in res.attributions:
                    actor = a.speaker_text.strip()
                    if actor:
                        actor_counts[actor] = actor_counts.get(actor, 0) + 1
                for c in res.claims:
                    ct = c.claim_text.strip()
                    if ct:
                        claim_counts[ct] = claim_counts.get(ct, 0) + 1
                # Flat attributions table rows
                for a in res.attributions:
                    # Try to find the claim text
                    claim_text = next((c.claim_text for c in res.claims if c.claim_id == a.claim_id), "")
                    atts_rows.append([
                        doc_id,
                        _safe_title(meta),
                        a.speaker_text,
                        claim_text,
                        a.mode.value,
                        str(a.confidence),
                    ])
                processed_count += 1
            except Exception as exc:  # log and continue
                print(f"Error processing {doc_id}: {exc}")
                continue

    # Write aggregations
    if cfg.save_csv:
        with open(actors_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["actor", "count"])
            for k, v in sorted(actor_counts.items(), key=lambda kv: kv[1], reverse=True):
                w.writerow([k, v])
        with open(claims_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["claim", "count"])
            for k, v in sorted(claim_counts.items(), key=lambda kv: kv[1], reverse=True):
                w.writerow([k, v])
        with open(atts_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["doc_id", "title", "actor", "claim", "mode", "confidence"])
            w.writerows(atts_rows)

    # Report
    write_html_report(
        html_path,
        corpus_dir=str(corpus_dir),
        total_docs=processed_count,
        actors=sorted(actor_counts.items(), key=lambda kv: kv[1], reverse=True)[:50],
        claims=sorted(claim_counts.items(), key=lambda kv: kv[1], reverse=True)[:50],
        atts_count=len(atts_rows),
        jsonl_path=str(jsonl_path) if cfg.save_jsonl else None,
        actors_csv=str(actors_csv_path) if cfg.save_csv else None,
        claims_csv=str(claims_csv_path) if cfg.save_csv else None,
        atts_csv=str(atts_csv_path) if cfg.save_csv else None,
    )

    return {
        "jsonl": jsonl_path,
        "actors_csv": actors_csv_path,
        "claims_csv": claims_csv_path,
        "atts_csv": atts_csv_path,
        "html": html_path,
    }


def main():
    ap = argparse.ArgumentParser(description="Run actors/claims/attributions extraction")
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config")
    ap.add_argument("--report-only", action="store_true", help="Only rebuild the HTML report from existing outputs")
    args = ap.parse_args()
    cfg = load_config(Path(args.config))
    if args.report_only:
        cfg.report_only = True
    run_from_config(cfg)


if __name__ == "__main__":
    main()
