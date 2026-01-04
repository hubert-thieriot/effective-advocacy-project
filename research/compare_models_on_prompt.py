#!/usr/bin/env python3
"""
Compare GPT models on the same resolved narrative-framing application prompt.

- Inputs: one or more resolved application USER prompt file paths
  (e.g., results/.../prompts/application/resolved/application_batch_011_user.txt)
  Optionally provide an explicit SYSTEM prompt path; otherwise the script
  will try to infer a sibling "_system.txt" next to each user prompt; if not
  found, it will fall back to prompts/application/system.jinja (raw content).

- For each model, the script sends [system, user] messages to OpenAI via the
  shared OpenAIInterface with centralized caching. It expects a JSON array
  response where each item contains:
    {"passage_id": ..., "probs": {frame_id: prob, ...}, ...}

- It aggregates normalized probabilities across all passages from all input
  prompts and renders an HTML table with one column per model and one row per
  frame, showing bar charts of average probabilities (share) per frame.

Output: results/research/frame_model_comparison.html by default.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv

from efi_analyser.scorers.openai_interface import OpenAIConfig, OpenAIInterface

_PALETTE = [
    "#1E3D58",
    "#057D9F",
    "#F18F01",
    "#A23B72",
    "#6C63FF",
    "#3A7D44",
    "#F45B69",
    "#0E7C7B",
    "#F2A541",
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def infer_system_for_user(user_path: Path, fallback: Path | None) -> str:
    # Try sibling "_system.txt" next to given user prompt
    name = user_path.name
    if name.endswith("_user.txt"):
        sibling = user_path.parent / name.replace("_user.txt", "_system.txt")
        if sibling.exists():
            return read_text(sibling)
    # Fallback if provided
    if fallback and fallback.exists():
        return read_text(fallback)
    # Last resort: empty system prompt
    return ""


def build_messages(system_content: str, user_content: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_content or "You are a helpful framing classifier."},
        {"role": "user", "content": user_content},
    ]


def _clean_json_fence(s: str) -> str:
    cleaned = s.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 3 and lines[-1].strip().startswith("```"):
            cleaned = "\n".join(lines[1:-1]).strip()
    return cleaned


def parse_batch_response(raw: str) -> List[Dict[str, object]]:
    cleaned = _clean_json_fence(raw)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return []
    # Only keep entries with a dict 'probs'
    out: List[Dict[str, object]] = []
    for item in payload:
        if isinstance(item, dict) and isinstance(item.get("probs"), dict):
            out.append(item)
    return out


def aggregate_frame_shares(entries: List[Dict[str, object]]) -> Tuple[List[str], Dict[str, float]]:
    """Return (frame_order, avg_share_by_frame) from parsed entries.

    Normalizes per entry so probabilities sum to 1.0 if possible, then averages.
    """
    if not entries:
        return [], {}
    # Collect union of frame ids
    frame_ids: List[str] = []
    seen = set()
    for e in entries:
        for fid in (e.get("probs") or {}).keys():
            if fid not in seen:
                seen.add(fid)
                frame_ids.append(str(fid))
    # Compute mean normalized probability per frame
    sums = {fid: 0.0 for fid in frame_ids}
    n = 0
    for e in entries:
        probs_raw: Dict[str, float] = {str(k): float(v) for k, v in (e.get("probs") or {}).items()}
        total = sum(probs_raw.values())
        if total <= 0:
            continue
        n += 1
        for fid in frame_ids:
            sums[fid] += probs_raw.get(fid, 0.0) / total
    if n == 0:
        return frame_ids, {fid: 0.0 for fid in frame_ids}
    return frame_ids, {fid: sums[fid] / n for fid in frame_ids}


def render_html(frame_ids: List[str], model_to_shares: Dict[str, Dict[str, float]], out_path: Path) -> None:
    models = list(model_to_shares.keys())
    # Build table rows per frame
    def _bar_cell(model: str, fid: str) -> str:
        v = float(model_to_shares.get(model, {}).get(fid, 0.0))
        pct = max(0.0, min(v, 1.0)) * 100.0
        return (
            f'<div class="barcell"><div class="bar" style="width:{pct:.2f}%"></div>'
            f'<div class="barlabel">{pct:.1f}%</div></div>'
        )

    rows = []
    for fid in frame_ids:
        cells = [f"<td class=\"frameid\">{fid}</td>"]
        for m in models:
            cells.append(f"<td>{_bar_cell(m, fid)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")

    head_cells = ["<th>Frame</th>"] + [f"<th>{m}</th>" for m in models]
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Frame Model Comparison</title>
  <style>
    body {{ font-family: Inter, system-ui, -apple-system, Segoe UI, sans-serif; margin: 24px; color: #1e293b; }}
    h1 {{ margin-bottom: 6px; }}
    p.note {{ color:#475569; margin-top: 4px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
    th, td {{ border-bottom: 1px solid #e2e8f0; padding: 8px 10px; text-align: left; }}
    th {{ background: #f8fafc; font-weight: 600; }}
    td.frameid {{ font-weight: 600; white-space: nowrap; }}
    .barcell {{ position: relative; height: 18px; background: #f1f5f9; border-radius: 4px; overflow: hidden; }}
    .bar {{ position: absolute; left: 0; top: 0; bottom: 0; background: #0ea5e9; }}
    .barlabel {{ position: absolute; right: 6px; top: 0; bottom: 0; display: flex; align-items: center; font-size: 12px; color: #0f172a; font-weight: 600; }}
    .meta {{ margin-top: 8px; color: #64748b; }}
  </style>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
  </head>
<body>
  <h1>Frame Model Comparison</h1>
  <p class="note">Average normalized frame probabilities across all passages for the given prompts.</p>
  <table>
    <thead><tr>{''.join(head_cells)}</tr></thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
  <div class="meta">Generated by research/compare_models_on_prompt.py</div>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def parse_frames_from_user(user_content: str) -> Tuple[List[str], Dict[str, str]]:
    ids: List[str] = []
    names: Dict[str, str] = {}
    lines = user_content.splitlines()
    in_frames = False
    for line in lines:
        if line.strip().startswith("Frames "):
            in_frames = True
            continue
        if in_frames and line.strip().startswith("TASK"):
            break
        if in_frames and line.strip().startswith("- ") and ":" in line:
            try:
                left = line.strip()[2:]
                fid, name = left.split(":", 1)
                fid = fid.strip()
                name = name.strip()
                ids.append(fid)
                names[fid] = name
            except Exception:
                continue
    return ids, names


def parse_passages_from_user(user_content: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    lines = user_content.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("- passage_id:"):
            try:
                pid = line.split(":", 1)[1].strip()
            except Exception:
                continue
            txt = ""
            if i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                if nxt.startswith("TEXT:"):
                    txt = nxt.split(":", 1)[1].strip()
                elif nxt.startswith("- text:"):
                    txt = nxt.split(":", 1)[1].strip()
            out.append((pid, txt))
    return out


def _color_for_index(i: int) -> str:
    return _PALETTE[i % len(_PALETTE)]


def _bars_strip(probs: Dict[str, float], frame_order: List[str], frame_names: Dict[str, str]) -> str:
    total = sum(max(0.0, float(probs.get(fid, 0.0))) for fid in frame_order)
    rows: List[str] = []
    for idx, fid in enumerate(frame_order):
        val = float(probs.get(fid, 0.0))
        frac = 0.0 if total <= 0 else max(0.0, min(val / total, 1.0))
        pct = frac * 100.0
        label = frame_names.get(fid, fid)
        color = _color_for_index(idx)
        rows.append(
            f'<div class="frow"><div class="fname">{label}</div>'
            f'<div class="fbar"><div class="bar" style="width:{pct:.1f}%;background:{color}"></div>'
            f'<div class="fval">{pct:.0f}%</div></div></div>'
        )
    return "".join(rows)


def render_html_per_passage(
    passages: List[Tuple[str, str]],
    frame_order: List[str],
    frame_names: Dict[str, str],
    models: List[str],
    model_to_probs: Dict[str, Dict[str, Dict[str, float]]],
    out_path: Path,
) -> None:
    def _bars_strip_no_names(probs: Dict[str, float], frame_order: List[str]) -> str:
        total = sum(max(0.0, float(probs.get(fid, 0.0))) for fid in frame_order)
        rows: List[str] = []
        for idx, fid in enumerate(frame_order):
            val = float(probs.get(fid, 0.0))
            frac = 0.0 if total <= 0 else max(0.0, min(val / total, 1.0))
            pct = frac * 100.0
            color = _color_for_index(idx)
            rows.append(
                f'<div class="frow2"><div class="fbar"><div class="bar" style="width:{pct:.1f}%;background:{color}"></div>'
                f'<div class="fval">{pct:.0f}%</div></div></div>'
            )
        return "".join(rows)

    def _cell_for_model(m: str, pid: str, include_names: bool) -> str:
        probs = model_to_probs.get(m, {}).get(pid, {})
        inner = _bars_strip(probs, frame_order, frame_names) if include_names else _bars_strip_no_names(probs, frame_order)
        cls = "modelcell first" if include_names else "modelcell narrow"
        return f'<td class="{cls}">{inner}</td>'

    head_cells = ["<th>Text</th>"] + [f"<th>{m}</th>" for m in models]
    rows: List[str] = []
    for pid, text in passages:
        cells = [f"<td class=\"ptext\">{text}</td>"]
        for i, m in enumerate(models):
            cells.append(_cell_for_model(m, pid, include_names=(i == 0)))
        rows.append("<tr>" + "".join(cells) + "</tr>")

    html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\"/>
  <title>Frame Model Comparison – Passages</title>
  <style>
    body {{ font-family: Inter, system-ui, -apple-system, Segoe UI, sans-serif; margin: 24px; color: #1e293b; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #e2e8f0; padding: 10px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f8fafc; font-weight: 600; position: sticky; top: 0; }}
    td.ptext {{ max-width: 520px; }}
    .modelcell {{ min-width: 180px; }}
    .modelcell.first {{ min-width: 380px; }}
    .modelcell.narrow {{ width: 180px; max-width: 220px; }}
    .frow {{ display: grid; grid-template-columns: 160px 1fr 48px; align-items: center; gap: 8px; margin: 3px 0; }}
    .frow2 {{ display: grid; grid-template-columns: 1fr 48px; align-items: center; gap: 8px; margin: 3px 0; }}
    .fname {{ font-size: 12px; color: #334155; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .fbar {{ position: relative; height: 12px; background: #f1f5f9; border-radius: 4px; overflow: hidden; }}
    .fbar .bar {{ position: absolute; left: 0; top: 0; bottom: 0; }}
    .fval {{ font-size: 12px; text-align: right; color: #0f172a; font-weight: 600; }}
  </style>
</head>
<body>
  <h1>Frame Model Comparison – Passages</h1>
  <p class=\"note\">Each row is a passage. Cells show per-frame bars (normalized per passage) for each model.</p>
  <table>
    <thead><tr>{''.join(head_cells)}</tr></thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    load_dotenv()

    ap = argparse.ArgumentParser(description="Compare GPT models on the same resolved narrative-framing prompt")
    ap.add_argument("user_prompts", nargs="+", help="Paths to resolved application USER prompts (*.txt)")
    ap.add_argument("--system", dest="system_prompt", default=None, help="Path to SYSTEM prompt (resolved). If omitted, tries sibling _system.txt; then falls back to prompts/application/system.jinja.")
    ap.add_argument(
        "--models",
        default="gpt-4.1,gpt-4.1-mini,gpt-4o,gpt-4o-mini,gpt-5,gpt-5-nano",
        help="Comma-separated model ids (e.g., gpt-4.1,gpt-4.1-mini,gpt-4o,gpt-4o-mini,gpt-5,gpt-5-nano)",
    )
    ap.add_argument("--temperature", type=float, default=None, help="Override temperature (default per model)")
    ap.add_argument("--out", default="results/research/frame_model_comparison.html", help="Output HTML path")
    ap.add_argument("--per-passage", action="store_true", help="Render one row per passage with per-frame bars per model")
    args = ap.parse_args()

    user_paths = [Path(p).resolve() for p in args.user_prompts]
    sys_path = Path(args.system_prompt).resolve() if args.system_prompt else None
    default_system_fallback = Path("prompts/application/system.jinja")
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    # Read all user prompt contents (concatenate multiple batches into one logical run)
    user_contents: List[str] = [read_text(p) for p in user_paths]
    # Determine a system prompt content; prefer sibling per user prompt if present; fall back to provided/global
    system_content: str | None = None
    # Use the first user prompt to infer sibling; if not found, fallback to global or empty
    system_content = infer_system_for_user(user_paths[0], sys_path or default_system_fallback)

    # Build messages for each user content and concatenate entries
    # We'll run each user content separately and aggregate entries across them
    model_to_entries: Dict[str, List[Dict[str, object]]] = {m: [] for m in models}
    model_to_counts: Dict[str, int] = {m: 0 for m in models}
    # Parse frames + passages from first prompt and union of passages across prompts
    ref_frames_order, ref_frame_names = parse_frames_from_user(user_contents[0])
    seen_pids = set()
    passages_all: List[Tuple[str, str]] = []
    for uc in user_contents:
        for pid, txt in parse_passages_from_user(uc):
            if pid not in seen_pids:
                seen_pids.add(pid)
                passages_all.append((pid, txt))

    for m in models:
        temp = OpenAIConfig.get_default_temperature(m) if args.temperature is None else float(args.temperature)
        client = OpenAIInterface(name=f"model_{m}", config=OpenAIConfig(model=m, temperature=temp, verbose=False))
        all_entries: List[Dict[str, object]] = []
        for user_content in user_contents:
            messages = build_messages(system_content=system_content or "", user_content=user_content)
            raw = client.infer(messages)
            entries = parse_batch_response(raw)
            all_entries.extend(entries)
            model_to_counts[m] += len(entries)
        model_to_entries[m] = all_entries

    # Build frame universe and shares
    all_frame_ids: List[str] = []
    seen = set()
    per_model_shares: Dict[str, Dict[str, float]] = {}
    per_model_probs: Dict[str, Dict[str, Dict[str, float]]] = {m: {} for m in models}
    for m, entries in model_to_entries.items():
        # Build passage->probs map and collect frame ids
        for e in entries:
            pid = e.get("passage_id")
            probs = e.get("probs") or {}
            if isinstance(pid, str) and isinstance(probs, dict):
                per_model_probs[m][pid] = {str(k): float(v) for k, v in probs.items()}
                for fid in probs.keys():
                    if fid not in seen:
                        seen.add(str(fid))
                        all_frame_ids.append(str(fid))
        frame_ids, shares = aggregate_frame_shares(entries)
        per_model_shares[m] = shares
        for fid in frame_ids:
            if fid not in seen:
                seen.add(fid)
                all_frame_ids.append(fid)

    # Choose frame order from prompt if available
    frame_order = ref_frames_order if ref_frames_order else all_frame_ids
    frame_names = ref_frame_names if ref_frame_names else {fid: fid for fid in frame_order}

    # Render HTML table
    out_path = Path(args.out)
    # Render either per-passage detail or aggregated shares
    if args.per_passage:
        render_html_per_passage(passages_all, frame_order, frame_names, models, per_model_probs, out_path)
    else:
        render_html(frame_order, per_model_shares, out_path)
    # Append meta-information below the table
    meta_lines = ["\n<!-- META -->\n<div class=\"meta\">"]
    meta_lines.append(f"Models: {', '.join(models)}")
    meta_lines.append(f"User prompts: {len(user_paths)}")
    for m in models:
        meta_lines.append(f"{m}: {model_to_counts.get(m, 0)} entries")
    meta_lines.append("<ul>")
    for p in user_paths:
        meta_lines.append(f"<li>{p}</li>")
    meta_lines.append("</ul></div>\n")
    try:
        with out_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(meta_lines))
    except Exception:
        pass
    print(f"Wrote comparison to: {out_path}")


if __name__ == "__main__":
    main()
