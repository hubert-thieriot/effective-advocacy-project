from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Dict
import json
import random

from efi_corpus.corpus_handle import CorpusHandle


def _html_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _render_table(headers: Sequence[str], rows: Sequence[Sequence[str]], max_rows: int = 50) -> str:
    cells = []
    head = "".join(f"<th>{_html_escape(h)}</th>" for h in headers)
    cells.append(f"<thead><tr>{head}</tr></thead>")
    body_rows = []
    for r in rows[:max_rows]:
        cols = "".join(f"<td>{_html_escape(str(c))}</td>" for c in r)
        body_rows.append(f"<tr>{cols}</tr>")
    cells.append(f"<tbody>{''.join(body_rows)}</tbody>")
    return f"<table class='tbl'>{''.join(cells)}</table>"


def write_html_report(
    out_path: Path,
    *,
    corpus_dir: str,
    total_docs: int,
    actors: List[Tuple[str, int]],
    claims: List[Tuple[str, int]],
    atts_count: int,
    jsonl_path: Optional[str],
    actors_csv: Optional[str],
    claims_csv: Optional[str],
    atts_csv: Optional[str],
):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = [
        "<html><head><meta charset='utf-8'><title>Actors & Claims Report</title>",
        "<style>body{font-family:system-ui, sans-serif; margin:20px;} h1,h2{margin:0.2em 0;} .tbl{border-collapse:collapse; width:100%;} .tbl th,.tbl td{border:1px solid #ddd; padding:6px;} .tbl th{background:#f7f7f7;}</style>",
        "</head><body>",
        f"<h1>Actors & Claims Report</h1>",
        f"<p><b>Corpus:</b> {_html_escape(corpus_dir)} &nbsp;&nbsp; <b>Docs processed:</b> {total_docs} &nbsp;&nbsp; <b>Attributions:</b> {atts_count}</p>",
    ]
    # Links
    links = []
    if jsonl_path:
        links.append(f"<a href='{_html_escape(jsonl_path)}'>extractions.jsonl</a>")
    if actors_csv:
        links.append(f"<a href='{_html_escape(actors_csv)}'>actors.csv</a>")
    if claims_csv:
        links.append(f"<a href='{_html_escape(claims_csv)}'>claims.csv</a>")
    if atts_csv:
        links.append(f"<a href='{_html_escape(atts_csv)}'>attributions.csv</a>")
    if links:
        html.append(f"<p>{' | '.join(links)}</p>")

    # Top actors
    html.append("<h2>Top Actors</h2>")
    html.append(_render_table(["Actor", "Attribution Count"], [(a, n) for a, n in actors]))
    # Top claims
    html.append("<h2>Top Claims</h2>")
    html.append(_render_table(["Claim", "Count"], [(c, n) for c, n in claims]))

    # Highlighted examples
    if jsonl_path:
        try:
            examples = _build_highlight_examples(corpus_dir, jsonl_path, max_examples=20, seed=17)
        except Exception as exc:
            examples = []
            html.append(f"<p style='color:#a00'>Error building examples: {_html_escape(str(exc))}</p>")
        if examples:
            html.append("<h2>Highlighted Examples</h2>")
            # Styles for highlights
            html.append(
                """
<style>
.snippet {border:1px solid #eee; padding:10px; margin:8px 0; background:#fafafa;}
.snippet .meta {font-size:12px; color:#555; margin-bottom:6px;}
.hl-actor {background:#dbeafe; padding:0 2px; border-radius:2px;}
.hl-claim {background:#fef08a; padding:0 2px; border-radius:2px;}
.hl-both {background:linear-gradient(90deg, #dbeafe, #fef08a); padding:0 2px; border-radius:2px;}
</style>
"""
            )
            for ex in examples:
                title = _html_escape(ex.get("title") or ex.get("doc_id", ""))
                url = ex.get("url", "")
                info = []
                speaker = ex.get("speaker", "")
                if speaker:
                    info.append(f"Speaker: <span class='hl-actor'>{_html_escape(speaker)}</span>")
                mode = ex.get("mode", "")
                if mode:
                    info.append(f"Mode: {_html_escape(mode)}")
                cconf = ex.get("claim_conf")
                if isinstance(cconf, (int, float)):
                    info.append(f"Claim conf: {cconf:.2f}")
                aconf = ex.get("att_conf")
                if isinstance(aconf, (int, float)):
                    info.append(f"Attribution conf: {aconf:.2f}")
                meta = (
                    f"<div class='meta'><b>{title}</b>"
                    + (f" — <a href='{_html_escape(url)}' target='_blank'>link</a>" if url else "")
                    + (" &nbsp; | &nbsp; " + " &nbsp; | &nbsp; ".join(info) if info else "")
                    + "</div>"
                )
                html.append(f"<div class='snippet'>{meta}{ex.get('snippet_html','')}</div>")

    html.append("</body></html>")
    Path(out_path).write_text("\n".join(html), encoding="utf-8")


def _build_highlight_examples(corpus_dir: str, jsonl_path: str, *, max_examples: int = 20, seed: int = 17) -> List[Dict[str, str]]:
    """Create highlighted snippets that show actor and claim spans.

    Picks up to `max_examples` attributions across documents. For each attribution,
    we build a context window around the union of the actor and claim spans and
    inject HTML spans with classes 'hl-actor' and 'hl-claim'. If the spans overlap,
    a combined 'hl-both' class is used.
    """
    handle = CorpusHandle(Path(corpus_dir), read_only=True)
    # Load records
    records: List[Dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    # Gather attributions across docs
    pairs: List[Tuple[Dict, Dict]] = []  # (record, attribution)
    for rec in records:
        atts = rec.get("attributions") or []
        if not isinstance(atts, list):
            continue
        for a in atts:
            if not isinstance(a, dict):
                continue
            pairs.append((rec, a))
    if not pairs:
        return []
    rng = random.Random(seed)
    rng.shuffle(pairs)
    examples: List[Dict[str, str]] = []
    for rec, att in pairs[: max_examples * 2]:  # over-sample and filter failures
        doc_id = rec.get("doc_id")
        if not doc_id:
            continue
        try:
            text = handle.get_text(str(doc_id))
        except Exception:
            continue
        # Find claim span
        claim_id = att.get("claim_id")
        claim = None
        for c in rec.get("claims") or []:
            if isinstance(c, dict) and c.get("claim_id") == claim_id:
                claim = c
                break
        if not claim:
            continue
        cs = int(claim.get("start_char", -1))
        ce = int(claim.get("end_char", -1))
        ss = int(att.get("speaker_start_char", -1))
        se = int(att.get("speaker_end_char", -1))
        if cs < 0 or ce <= cs:
            continue
        # Build snippet window covering both spans
        left = [p for p in [cs, ss] if p >= 0]
        right = [p for p in [ce, se] if p >= 0]
        start = max(0, (min(left) if left else cs) - 120)
        end = min(len(text), (max(right) if right else ce) + 120)
        snippet_html = _highlight_window(text, start, end, (ss, se) if ss >= 0 and se > ss else None, (cs, ce))
        # Pull extra info: confidences and mode
        claim_conf = claim.get("confidence")
        att_conf = att.get("confidence")
        mode = claim.get("mode") or att.get("mode")
        # Ensure simple str for mode
        mode_str = str(mode) if mode is not None else ""
        speaker_text = att.get("speaker_text") or ""
        examples.append(
            {
                "doc_id": str(doc_id),
                "title": rec.get("title", ""),
                "url": rec.get("url", ""),
                "snippet_html": snippet_html,
                "speaker": speaker_text,
                "mode": mode_str,
                "claim_conf": claim_conf,
                "att_conf": att_conf,
            }
        )
        if len(examples) >= max_examples:
            break
    return examples


def _highlight_window(text: str, win_s: int, win_e: int, actor_span: Optional[Tuple[int, int]], claim_span: Tuple[int, int]) -> str:
    """Return HTML for a text window with highlights for actor and claim.

    Spans are absolute char offsets; this function slices and escapes per segment,
    and wraps actor in <span class='hl-actor'>, claim in <span class='hl-claim'>,
    overlap in <span class='hl-both'>.
    """
    win_s = max(0, min(win_s, len(text)))
    win_e = max(win_s, min(win_e, len(text)))
    rel_actor = None
    if actor_span is not None:
        as_, ae_ = actor_span
        if ae_ > as_ and not (ae_ <= win_s or as_ >= win_e):
            rel_actor = (max(0, as_ - win_s), min(win_e - win_s, ae_ - win_s))
    cs, ce = claim_span
    rel_claim = (max(0, cs - win_s), min(win_e - win_s, ce - win_s))

    # Build boundary points
    boundaries = {0, win_e - win_s}
    for span in [rel_actor, rel_claim]:
        if span is None:
            continue
        s, e = span
        if e > s:
            boundaries.add(s)
            boundaries.add(e)
    ordered = sorted(boundaries)

    out_parts: List[str] = []
    for i in range(len(ordered) - 1):
        s = ordered[i]
        e = ordered[i + 1]
        seg = text[win_s + s : win_s + e]
        cls = None
        in_actor = rel_actor is not None and s < rel_actor[1] and e > rel_actor[0]
        in_claim = s < rel_claim[1] and e > rel_claim[0]
        if in_actor and in_claim:
            cls = "hl-both"
        elif in_actor:
            cls = "hl-actor"
        elif in_claim:
            cls = "hl-claim"
        esc = _html_escape(seg)
        if cls:
            out_parts.append(f"<span class='{cls}'>{esc}</span>")
        else:
            out_parts.append(esc)

    # Add ellipses at edges if we truncated
    prefix = "… " if win_s > 0 else ""
    suffix = " …" if win_e < len(text) else ""
    return prefix + "".join(out_parts) + suffix
