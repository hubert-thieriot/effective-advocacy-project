#!/usr/bin/env python3
"""Command-line entry point for the word count (theme keyword) workflow."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlparse

import pandas as pd

from apps.word_count.config import WordCountConfig, load_config
from apps.word_count.report import ThemeDocStat, generate_html_report
from apps.narrative_framing.aggregation_document import DocumentFrameAggregate
from efi_analyser.pipeline.word_occurrence import WordOccurrencePipeline
from efi_corpus import CorpusHandle


def _domain_from_url(url: str) -> str:
    try:
        netloc = urlparse(url or "").netloc
        return netloc.lower() or "unknown"
    except Exception:
        return "unknown"


def _date_only(s: str | None) -> str:
    if not s:
        return ""
    return str(s)[:10]


def main() -> None:
    parser = argparse.ArgumentParser(description="Word Count (themes) application")
    parser.add_argument("config", type=str, help="Path to YAML config")
    args = parser.parse_args()

    cfg: WordCountConfig = load_config(Path(args.config))

    corpus_path = Path("corpora") / cfg.corpus
    if not corpus_path.exists():
        raise SystemExit(f"Corpus not found: {corpus_path}")

    # Prepare pipeline with all unique keywords
    pipeline = WordOccurrencePipeline(
        keywords=cfg.all_keywords(),
        case_sensitive=cfg.case_sensitive,
        whole_word_only=cfg.whole_word_only,
        allow_hyphenation=cfg.allow_hyphenation,
        doc_limit=cfg.doc_limit,
        date_from=cfg.date_from,
        date_to=cfg.date_to,
        patterns=cfg.all_patterns(),
    )

    corpus_handle = CorpusHandle(corpus_path)
    pipe_result = pipeline.run(corpus_handle)
    word_results = pipeline.get_word_occurrence_results(pipe_result)

    # Build keyword -> theme mapping
    kw_to_theme: Dict[str, str] = {}
    theme_names: Dict[str, str] = {}
    theme_order: List[str] = []
    for t in cfg.themes:
        theme_names[t.id] = t.name
        theme_order.append(t.id)
        for k in t.keywords:
            kw_to_theme[k] = t.id
        # Map regex patterns as well so we can attribute counts
        for p in (t.patterns or []):
            kw_to_theme[p] = t.id

    # Aggregations
    theme_doc_counts: Dict[str, int] = {tid: 0 for tid in theme_order}
    daily_counts: Dict[str, Dict[str, int]] = {tid: defaultdict(int) for tid in theme_order}
    daily_total_docs: Dict[str, int] = defaultdict(int)
    domain_counts: Dict[str, int] = defaultdict(int)
    cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)
    top_docs: Dict[str, List[ThemeDocStat]] = {tid: [] for tid in theme_order}

    total_docs = 0
    document_aggregates: List[DocumentFrameAggregate] = []

    for r in word_results.results:
        total_docs += 1
        date = _date_only(r.date)
        url = r.url or r.metadata.get("uri") or ""
        domain = _domain_from_url(url)
        title = r.title or r.metadata.get("title") or ""

        # Compute per-theme hits
        doc_theme_hits: Dict[str, int] = defaultdict(int)
        for kw, cnt in (r.keyword_counts or {}).items():
            if cnt <= 0:
                continue
            theme_id = kw_to_theme.get(kw)
            if not theme_id:
                continue
            doc_theme_hits[theme_id] += int(cnt)

        # Track total documents per day (regardless of theme presence)
        if date:
            daily_total_docs[date] += 1

        # Only count as having a theme if hits >= cfg.min_words
        if doc_theme_hits:
            domain_counts[domain] += 1

        present_themes = [tid for tid, hits in doc_theme_hits.items() if hits >= (cfg.min_words or 1)]
        for tid in present_themes:
            theme_doc_counts[tid] += 1
            if date:
                daily_counts[tid][date] += 1
            top_docs[tid].append(
                ThemeDocStat(
                    document_id=r.document_id,
                    url=url,
                    title=title,
                    date=date,
                    domain=domain,
                    hits=doc_theme_hits[tid],
                )
            )
        # Co-occurrence
        for a in present_themes:
            for b in present_themes:
                cooccurrence[(a, b)] += 1

        # Build document-level aggregates (binary presence per theme)
        frame_scores = {tid: (1.0 if doc_theme_hits.get(tid, 0) >= (cfg.min_words or 1) else 0.0) for tid in theme_order}
        document_aggregates.append(
            DocumentFrameAggregate(
                doc_id=r.document_id,
                frame_scores=frame_scores,
                total_weight=1.0,
                published_at=r.date,
                title=title,
                url=url,
                top_frames=[tid for tid, v in frame_scores.items() if v > 0.0],
            )
        )

    # Sort top docs per theme by hits desc, then date desc
    for tid, items in top_docs.items():
        items.sort(key=lambda x: (x.hits, x.date), reverse=True)

    # Prepare results directory
    case_id = Path(args.config).stem
    results_dir = cfg.results_dir or (Path("results") / "word_count" / case_id)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Persist basic CSV/JSON
    # Summary CSV
    summary_path = results_dir / "summary.csv"
    pd.DataFrame({
        "theme_id": list(theme_doc_counts.keys()),
        "theme_name": [theme_names.get(t, t) for t in theme_doc_counts.keys()],
        "documents": list(theme_doc_counts.values()),
    }).to_csv(summary_path, index=False)

    # Daily counts CSV
    daily_rows = []
    for tid, series in daily_counts.items():
        for d, v in series.items():
            daily_rows.append({"theme_id": tid, "date": d, "documents": v})
    daily_df = pd.DataFrame(daily_rows)
    if not daily_df.empty:
        daily_df.sort_values(["date", "theme_id"]).to_csv(results_dir / "daily_counts.csv", index=False)

    # Domain counts CSV
    if domain_counts:
        pd.DataFrame({"domain": list(domain_counts.keys()), "documents": list(domain_counts.values())}) \
            .sort_values("documents", ascending=False) \
            .to_csv(results_dir / "domain_counts.csv", index=False)

    # Save raw metadata JSON for debugging
    raw_json = {
        "theme_counts": theme_doc_counts,
        "total_documents": total_docs,
        "date_from": cfg.date_from,
        "date_to": cfg.date_to,
    }
    import json as _json
    (results_dir / "report_data.json").write_text(_json.dumps(raw_json, indent=2), encoding="utf-8")

    # Generate HTML report
    html_path = results_dir / "frame_report.html"
    generate_html_report(
        html_path,
        case_title=case_id.replace("_", " ").title(),
        theme_counts=theme_doc_counts,
        daily_counts={k: dict(v) for k, v in daily_counts.items()},
        domain_counts=dict(domain_counts),
        cooccurrence=dict(cooccurrence),
        top_docs=top_docs,
        theme_names=theme_names,
        total_docs=total_docs,
        date_from=cfg.date_from,
        date_to=cfg.date_to,
        document_aggregates=document_aggregates,
        daily_total_docs=dict(daily_total_docs),
    )

    print(f"Processed documents: {total_docs}")
    print(f"Themes: {', '.join([theme_names[t] for t in theme_order])}")
    print(f"Report saved to: {html_path}")


if __name__ == "__main__":
    main()
