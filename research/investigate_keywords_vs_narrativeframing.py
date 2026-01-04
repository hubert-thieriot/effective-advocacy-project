#!/usr/bin/env python3
"""
Investigate discrepancies between keyword-based word_count and narrative_framing results.

Compares a target frame (e.g., construction_dust) between:
- Keyword presence using the word_count config's theme keywords
- LLM-labeled passages in narrative_framing assignments

Outputs summary stats and suggested keyword gaps.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml


def load_word_count_keywords(config_path: Path, theme_id: str) -> List[str]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for t in data.get("themes", []):
        if str(t.get("id")) == theme_id:
            return [str(k).strip() for k in (t.get("keywords") or []) if str(k).strip()]
    return []


def compile_keyword_patterns(keywords: Sequence[str]) -> List[Tuple[str, re.Pattern]]:
    pats: List[Tuple[str, re.Pattern]] = []
    for kw in keywords:
        # Escape regex chars but keep spaces flexible (\s+)
        esc = re.escape(kw).replace("\\ ", "\\s+")
        pat = re.compile(r"(?i)" + esc)
        pats.append((kw, pat))
    return pats


def load_schema(schema_path: Path) -> Dict[str, Dict[str, object]]:
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    out: Dict[str, Dict[str, object]] = {}
    for item in payload.get("frames", []):
        out[str(item.get("frame_id"))] = {
            "name": item.get("name"),
            "short": item.get("short_name") or item.get("name") or item.get("frame_id"),
            "keywords": item.get("keywords", []),
        }
    return out


def load_assignments(assignments_path: Path) -> List[Dict[str, object]]:
    return json.loads(assignments_path.read_text(encoding="utf-8"))


def filter_assignments_for_frame(
    assignments: Sequence[Dict[str, object]],
    frame_id: str,
    *,
    prob_threshold: Optional[float] = None,
) -> List[Dict[str, object]]:
    selected: List[Dict[str, object]] = []
    for a in assignments:
        tops = set(a.get("top_frames") or [])
        if frame_id in tops:
            selected.append(a)
            continue
        if prob_threshold is not None:
            try:
                p = float((a.get("probabilities") or {}).get(frame_id, 0.0))
            except Exception:
                p = 0.0
            if p >= prob_threshold:
                selected.append(a)
    return selected


def evaluate_keyword_hits(
    items: Sequence[Dict[str, object]],
    patterns: Sequence[Tuple[str, re.Pattern]],
) -> Tuple[int, Dict[str, int], List[Tuple[str, str]]]:
    hit_any = 0
    kw_hits: Dict[str, int] = defaultdict(int)
    misses: List[Tuple[str, str]] = []
    for a in items:
        text = str(a.get("passage_text") or "")
        found = False
        for kw, pat in patterns:
            if pat.search(text):
                found = True
                kw_hits[kw] += 1
        if found:
            hit_any += 1
        else:
            if len(misses) < 8:
                title = (a.get("metadata") or {}).get("title") or ""
                misses.append((title, text[:240].replace("\n", " ")))
    return hit_any, dict(kw_hits), misses


def top_terms(items: Sequence[Dict[str, object]], *, top_n: int = 30) -> List[Tuple[str, int]]:
    words: List[str] = []
    stop = {
        # Indonesian + common fillers (very light list for quick signal)
        "dan","yang","untuk","dengan","di","ke","dalam","ini","itu","atau","pada","dari","oleh",
        "karena","adalah","akan","sebagai","juga","lebih","bisa","tidak","sudah","kami","kita","mereka",
        "para","saat","sebuah","dapat","pemerintah","menjadi","tersebut","seperti","tahun","hari","kata",
        # English
        "the","and","for","with","this","that","from","have","has","are","was","were","will","can",
    }
    for a in items:
        text = str(a.get("passage_text") or "").lower()
        tokens = re.findall(r"\b[\w\-]{3,}\b", text)
        words.extend([w for w in tokens if w not in stop])
    ctr = Counter(words)
    return ctr.most_common(top_n)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare word_count keywords vs narrative_framing labels for a target frame")
    ap.add_argument("word_count_config", type=Path)
    ap.add_argument("narrative_results_dir", type=Path)
    ap.add_argument("--frame-id", default="construction_dust", help="Target frame_id (default: construction_dust)")
    ap.add_argument("--prob-threshold", type=float, default=0.2, help="Min probability to include if not in top_frames")
    args = ap.parse_args()

    wc_cfg = args.word_count_config
    res_dir = args.narrative_results_dir
    schema_path = res_dir / "frame_schema.json"
    assignments_path = res_dir / "frame_assignments.json"

    if not schema_path.exists() or not assignments_path.exists():
        raise SystemExit(f"Missing schema or assignments in {res_dir}")

    wc_keywords = load_word_count_keywords(wc_cfg, args.frame_id)
    if not wc_keywords:
        print(f"No keywords found for theme '{args.frame_id}' in {wc_cfg}")
    pats = compile_keyword_patterns(wc_keywords)

    schema = load_schema(schema_path)
    label = schema.get(args.frame_id, {}).get("name") or args.frame_id
    llm_keywords = schema.get(args.frame_id, {}).get("keywords") or []

    assignments = load_assignments(assignments_path)
    selected = filter_assignments_for_frame(assignments, args.frame_id, prob_threshold=args.prob_threshold)

    print(f"Frame: {args.frame_id} – {label}")
    print(f"word_count keywords: {len(wc_keywords)}; narrative schema keywords: {len(llm_keywords)}")
    if llm_keywords:
        print(f"  schema keywords: {', '.join(llm_keywords)}")
    print(f"Selected LLM-labeled passages: {len(selected)} (top_frames or p>={args.prob_threshold})")

    hit_any, kw_hits, miss_samples = evaluate_keyword_hits(selected, pats)
    share = (hit_any / len(selected)) if selected else 0.0
    print(f"Passages containing any word_count keyword: {hit_any}/{len(selected)} ({share:.1%})")
    if kw_hits:
        top_kw = sorted(kw_hits.items(), key=lambda x: x[1], reverse=True)[:10]
        print("Top matching keywords:")
        for k, v in top_kw:
            print(f"  {k}: {v}")
    if miss_samples:
        print("Examples without keyword hits:")
        for title, preview in miss_samples:
            t = title or "(no title)"
            print(f"- {t}\n  {preview}")

    # Suggest additions by showing frequent terms in LLM-labeled passages
    terms = top_terms(selected, top_n=40)
    print("\nTop terms in LLM-labeled passages (quick signal):")
    for w, c in terms:
        print(f"  {w}: {c}")


if __name__ == "__main__":
    main()

