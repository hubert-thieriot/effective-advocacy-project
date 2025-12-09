"""Plotting functions for narrative framing reports."""

from __future__ import annotations

import base64
import copy
import html
import json
import math
import textwrap
from collections import defaultdict
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil import parser
from sklearn.metrics import roc_auc_score, roc_curve

import kaleido  # type: ignore  # noqa: F401
import plotly.graph_objects as go  # type: ignore
import plotly.io as pio  # type: ignore

from apps.narrative_framing.aggregation_document import DocumentFrameAggregate
from efi_analyser.frames import FrameAssignment
from efi_core.utils import normalize_date

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


def build_color_map(frames) -> Dict[str, str]:
    """Build a color map for frames."""
    color_map: Dict[str, str] = {}
    for idx, frame in enumerate(frames):
        color_map[frame.frame_id] = _PALETTE[idx % len(_PALETTE)]
    return color_map


def build_corpus_color_map(corpora: Sequence[str]) -> Dict[str, str]:
    """Assign distinct colors to each corpus name using the global palette."""
    color_map: Dict[str, str] = {}
    for idx, name in enumerate(corpora):
        color_map[str(name)] = _PALETTE[idx % len(_PALETTE)]
    return color_map


def sanitize_filename(name: str, *, max_len: int = 120, fallback: Optional[str] = None) -> str:
    """Return a filesystem-safe filename (no path) limited in length."""
    if not name:
        return fallback or "chart.png"
    # Split extension
    dot = name.rfind(".")
    if 0 < dot < len(name) - 1:
        base, ext = name[:dot], name[dot:]
    else:
        base, ext = name, ".png"
    # Filter base chars
    safe = []
    for ch in base:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    base_safe = "".join(safe).strip("._") or (fallback or "chart").rsplit(".", 1)[0]
    # Truncate
    keep = max_len - len(ext)
    if keep < 8:
        keep = max(8, keep)
    if len(base_safe) > keep:
        base_safe = base_safe[:keep]
    return base_safe + ext


def safe_export_path(export_png_path: Optional[Path], *, div_id: str) -> Optional[Path]:
    """Sanitize and shorten the output path to avoid OS filename limits."""
    if export_png_path is None:
        return None
    try:
        p = Path(export_png_path)
    except Exception:
        p = Path(str(export_png_path))
    # Ensure we only sanitize the final filename, not the directory
    name = p.name or f"{div_id}.png"
    # If the name looks like an HTML-ish title dump, replace with div_id
    suspicious_tokens = ("br", "span", "style", "font", "color")
    if any(tok in name for tok in suspicious_tokens) or len(name) > 120:
        name = f"{div_id}.png"
    safe_name = sanitize_filename(name, fallback=f"{div_id}.png")
    if safe_name != name:
        print(f"ℹ️  Sanitized export filename for {div_id}: '{name}' → '{safe_name}'")
    return p.with_name(safe_name)


def compute_classifier_metrics(
    assignments: Sequence[FrameAssignment],
    frame_ids: List[str],
    threshold: float = 0.5,
    classifier_lookup: Optional[Dict[str, Dict[str, object]]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute precision, recall, F1, and AUC for each frame."""
    metrics = {}
    
    for frame_id in frame_ids:
        # Get true labels and predictions for this frame
        y_true = []
        y_scores = []
        
        for assignment in assignments:
            # True label: 1 if this frame is in top_frames, 0 otherwise
            true_label = 1 if frame_id in assignment.top_frames else 0
            y_true.append(true_label)
            
            # Prediction score: prefer classifier probabilities if available
            if classifier_lookup is not None:
                pred_entry = classifier_lookup.get(assignment.passage_id)
                if pred_entry and isinstance(pred_entry.get("probabilities"), dict):
                    score = float(pred_entry["probabilities"].get(frame_id, 0.0))  # type: ignore[index]
                else:
                    score = 0.0
            else:
                # Fallback to LLM probabilities if no classifier provided
                score = assignment.probabilities.get(frame_id, 0.0)
            y_scores.append(score)
        
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        y_pred = (y_scores >= threshold).astype(int)
        
        # Compute metrics
        if len(np.unique(y_true)) > 1:  # Only compute if both classes present
            precision = np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_pred == 1) if np.sum(y_pred == 1) > 0 else 0.0
            recall = np.sum((y_pred == 1) & (y_true == 1)) / np.sum(y_true == 1) if np.sum(y_true == 1) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            auc = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0
        else:
            precision = recall = f1 = auc = 0.0
        
        metrics[frame_id] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'support': int(np.sum(y_true))
        }
    
    return metrics


def plot_precision_recall_bars(metrics: Dict[str, Dict[str, float]], frame_names: Dict[str, str], color_map: Dict[str, str]) -> str:
    """Create a horizontal bar chart showing precision and recall for each frame."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    frames = list(metrics.keys())
    precisions = [metrics[f]['precision'] for f in frames]
    recalls = [metrics[f]['recall'] for f in frames]
    colors = [color_map.get(f, '#4F8EF7') for f in frames]
    labels = [frame_names.get(f, f) for f in frames]
    wrapped_labels = [textwrap.fill(label, 18) for label in labels]
    
    # Precision bars
    y_pos = np.arange(len(frames))
    bars1 = ax1.barh(y_pos, precisions, color=colors, alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(wrapped_labels, fontsize=10)
    ax1.set_xlabel('Precision', fontsize=12)
    ax1.set_title('Precision by Frame', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 1)
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, precisions)):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.2f}', 
                va='center', fontsize=9, fontweight='bold')
    
    # Recall bars
    bars2 = ax2.barh(y_pos, recalls, color=colors, alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(wrapped_labels, fontsize=10)
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_title('Recall by Frame', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars2, recalls)):
        ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.2f}', 
                va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    return fig_to_base64(fig)


def plot_domain_counts_bar(domain_counts: Sequence[Tuple[str, int]]) -> str:
    """Create a horizontal bar chart of domain counts."""
    if not domain_counts:
        return ""
    domains = [item[0] for item in domain_counts]
    counts = [item[1] for item in domain_counts]
    indices = np.arange(len(domains))

    fig, ax = plt.subplots(figsize=(10, max(5, len(domains) * 0.35)))
    bars = ax.barh(indices, counts, color="#4F8EF7", alpha=0.85)
    ax.set_yticks(indices)
    ax.set_yticklabels(domains, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Document count", fontsize=12)
    ax.set_title("Top domains by document count", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    for bar, value in zip(bars, counts):
        ax.text(
            value + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            str(value),
            va="center",
            fontsize=9,
        )
    plt.tight_layout()
    return fig_to_base64(fig)


def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string."""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    return image_base64


def wrap_label_html(text: str, max_len: int = 16) -> str:
    """Insert <br> breaks into text so that each line is at most max_len characters."""
    words = str(text).split()
    if not words:
        return text
    lines: List[str] = []
    current: List[str] = []
    current_len = 0
    for w in words:
        wlen = len(w)
        # If adding this word exceeds max_len, start a new line
        if current and (current_len + 1 + wlen) > max_len:
            lines.append(" ".join(current))
            current = [w]
            current_len = wlen
        else:
            if current:
                current_len += 1 + wlen
                current.append(w)
            else:
                current = [w]
                current_len = wlen
    if current:
        lines.append(" ".join(current))
    return "<br>".join(lines)


def wrap_text_html(text: str, max_len: int = 72) -> str:
    """Soft-wrap long text for Plotly titles by inserting <br> at word boundaries."""
    if not text:
        return text
    words = str(text).split()
    if not words:
        return text
    lines: List[str] = []
    current: List[str] = []
    current_len = 0
    for w in words:
        wlen = len(w)
        if current and (current_len + 1 + wlen) > max_len:
            lines.append(" ".join(current))
            current = [w]
            current_len = wlen
        else:
            if current:
                current.append(w)
                current_len += 1 + wlen
            else:
                current = [w]
                current_len = wlen
    if current:
        lines.append(" ".join(current))
    return "<br>".join(lines)


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert hex color to rgba string."""
    value = hex_color.lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    try:
        r = int(value[0:2], 16)
        g = int(value[2:4], 16)
        b = int(value[4:6], 16)
    except ValueError:
        r = g = b = 0
    alpha = max(0.0, min(alpha, 1.0))
    return f"rgba({r}, {g}, {b}, {alpha})"


def render_probability_bars(
    probabilities: Dict[str, float],
    frame_lookup: Dict[str, Dict[str, str]],
    color_map: Dict[str, str],
) -> str:
    """Render probability bars as HTML."""
    if not probabilities:
        return "—"
    prob_bars = []
    sorted_items = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    for frame_id, score in sorted_items:
        width = max(2, int(score * 100))
        color = color_map.get(frame_id, "#4F8EF7")
        label = frame_lookup.get(frame_id, {}).get("short", frame_id)
        prob_bars.append(
            "<div class=\"bar\">"
            f"<div class=\"fill\" style=\"width:{width}%; background:{color};\"></div>"
            f"<span class=\"bar-label\">{html.escape(label)} ({score:.0%})</span>"
            "</div>"
        )
    return "".join(prob_bars) if prob_bars else "—"


def render_plotly_fragment(
    div_id: str,
    data: Sequence[Dict[str, object]],
    layout: Dict[str, object],
    *,
    config: Optional[Dict[str, object]] = None,
    export_png_path: Optional[Path] = None,
    export_svg: bool = False,
) -> str:
    """Render a Plotly chart as an HTML fragment."""
    if not data:
        return ""
    # Disable interactivity by default to prevent scroll trapping
    config = config or {"displayModeBar": False, "responsive": True, "staticPlot": True, "scrollZoom": False, "doubleClick": False}

    # Best-effort PNG export when requested
    if export_png_path is not None:
        export_png_path = safe_export_path(export_png_path, div_id=div_id)
        try:
            fig = go.Figure(data=list(data))
            # Use a title-safe layout for static export to avoid filename inference issues
            try:
                safe_layout = copy.deepcopy(layout)
            except Exception:
                safe_layout = dict(layout)
            title_entry = safe_layout.get("title")
            # Move title text into an annotation instead of layout.title
            if isinstance(title_entry, dict) and title_entry.get("text"):
                title_text = title_entry.get("text")
                # Remove title from layout for export
                safe_layout.pop("title", None)
                # Ensure annotations list exists
                anns = list(safe_layout.get("annotations", []))
                anns.append({
                    "text": title_text,
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0,
                    "y": 1.10,
                    "xanchor": "left",
                    "yanchor": "bottom",
                    "showarrow": False,
                    "align": "left",
                    "font": {"size": 18, "color": "#333"},
                })
                safe_layout["annotations"] = anns
            # Update layout for PNG export: transparent background, Inter font, higher resolution
            export_layout = copy.deepcopy(safe_layout) if isinstance(safe_layout, dict) else dict(safe_layout)
            export_layout.update({
                "plot_bgcolor": "rgba(0,0,0,0)",
                "paper_bgcolor": "rgba(0,0,0,0)",
                "font": {
                    "family": "Inter, 'Segoe UI', sans-serif",
                    "size": 12,
                    "color": "#1e293b"
                }
            })
            # Update axis fonts
            if "xaxis" in export_layout:
                if isinstance(export_layout["xaxis"], dict):
                    if "title" in export_layout["xaxis"]:
                        if isinstance(export_layout["xaxis"]["title"], dict):
                            export_layout["xaxis"]["title"].setdefault("font", {}).update({
                                "family": "Inter, 'Segoe UI', sans-serif", "size": 12
                            })
            if "yaxis" in export_layout:
                if isinstance(export_layout["yaxis"], dict):
                    if "title" in export_layout["yaxis"]:
                        if isinstance(export_layout["yaxis"]["title"], dict):
                            export_layout["yaxis"]["title"].setdefault("font", {}).update({
                                "family": "Inter, 'Segoe UI', sans-serif", "size": 12
                            })
            fig.update_layout(**export_layout)
            export_png_path.parent.mkdir(parents=True, exist_ok=True)
            # Use to_image() + write with higher resolution (scale=3 for better quality)
            img_bytes = pio.to_image(fig, format="png", scale=3, width=None, height=None)
            export_png_path.write_bytes(img_bytes)
            
            # Export SVG if requested
            if export_svg:
                try:
                    export_svg_path = export_png_path.with_suffix('.svg')
                    svg_bytes = pio.to_image(fig, format="svg", width=None, height=None)
                    export_svg_path.write_bytes(svg_bytes)
                except Exception as exc:
                    print(f"⚠️ Plotly SVG export failed for {div_id}: {exc}")
            
            # Also export as self-supporting HTML
            html_path = export_png_path.with_suffix('.html')
            fig_html = go.Figure(data=list(data))
            fig_html.update_layout(**export_layout)
            # Create self-supporting HTML with embedded Plotly
            html_content = fig_html.to_html(
                include_plotlyjs='cdn',
                div_id=div_id,
                config={"displayModeBar": False, "responsive": True}
            )
            # Inject CSS for font styling to match report
            css_injection = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        body {
            font-family: 'Inter', 'Segoe UI', sans-serif;
            margin: 0;
            padding: 0;
        }
        .plotly {
            font-family: 'Inter', 'Segoe UI', sans-serif !important;
        }
    </style>
"""
            # Insert CSS before closing </head> tag
            if '</head>' in html_content:
                html_content = html_content.replace('</head>', css_injection + '</head>')
            else:
                # If no </head> tag, insert before <body> or at the start
                if '<body' in html_content:
                    html_content = html_content.replace('<body', css_injection + '<body')
                else:
                    html_content = css_injection + html_content
            html_path.write_text(html_content)
        except Exception as exc:
            print(f"⚠️ Plotly PNG export failed for {div_id}: {exc}")
    data_json = json.dumps(data, ensure_ascii=False)
    layout_json = json.dumps(layout, ensure_ascii=False)
    config_json = json.dumps(config, ensure_ascii=False)
    return (
        f'<div id="{div_id}" class="plotly-chart"></div>'
        "<script>(function(){"
        f"var data = {data_json};"
        f"var layout = {layout_json};"
        f"var config = {config_json};"
        f"Plotly.newPlot('{div_id}', data, layout, config);"
        "})();</script>"
    )


def render_plotly_llm_coverage(
    assignments: Sequence[FrameAssignment],
    frames: Sequence[dict],
    color_map: Dict[str, str],
) -> str:
    """Render LLM coverage chart."""
    if not assignments or not frames:
        return ""
    # Count occurrences per frame_id across LLM assignments (based on top_frames)
    counts: Dict[str, int] = {str(f["frame_id"]): 0 for f in frames}
    for a in assignments:
        for fid in a.top_frames:
            if fid in counts:
                counts[fid] += 1
    # Sort by count desc
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    frame_ids = [fid for fid, _ in ordered]
    values = [int(v) for _, v in ordered]
    labels = []
    for fid in frame_ids:
        meta = next((f for f in frames if str(f["frame_id"]) == fid), None)
        label = (meta.get("short") or meta.get("name") or fid) if meta else fid
        labels.append(label)
    colors = [color_map.get(fid, "#057d9f") for fid in frame_ids]

    traces = [
        {
            "type": "bar",
            "x": labels,
            "y": values,
            "marker": {"color": colors},
            "hovertemplate": "%{x}<br>%{y} passages<extra></extra>",
        }
    ]
    layout = {
        "margin": {"l": 40, "r": 20, "t": 20, "b": 80},
        "xaxis": {"title": "Frame", "tickangle": -30},
        "yaxis": {"title": "Passages (LLM top_k)"},
        "height": 500,
    }
    return render_plotly_fragment("llm-coverage-chart", traces, layout)


def render_plotly_llm_binned_distribution(
    assignments: Sequence[FrameAssignment],
    frames: Sequence[dict],
    *,
    bins: Optional[Sequence[float]] = None,
) -> str:
    """Render a stacked bar chart of LLM probabilities binned per frame."""
    if not assignments or not frames:
        return ""

    frame_ids = [str(f["frame_id"]) for f in frames]
    labels = [str((f.get("short") or f.get("name") or f["frame_id"])) for f in frames]

    # Default bins: [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    if bins is None:
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001]
    # Build bin labels
    bin_labels = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        lo_s = f"{lo:.1f}".rstrip("0").rstrip(".")
        hi_s = f"{min(hi, 1.0):.1f}".rstrip("0").rstrip(".")
        bin_labels.append(f"{lo_s}–{hi_s}")

    # Initialize counts: dict[bin_index][frame_index] -> count
    counts = [[0 for _ in frame_ids] for _ in range(len(bins) - 1)]

    # Iterate assignments and tally
    for a in assignments:
        probs = a.probabilities or {}
        for fx, fid in enumerate(frame_ids):
            p = float(probs.get(fid, 0.0))
            # Find bin index
            bi = None
            for i in range(len(bins) - 1):
                if bins[i] <= p < bins[i + 1] or (math.isclose(p, 1.0) and i == len(bins) - 2):
                    bi = i
                    break
            if bi is not None:
                counts[bi][fx] += 1

    # Choose a fixed set of colors per bin
    bin_colors = [
        "#e2e8f0",  # light
        "#cbd5e1",
        "#94a3b8",
        "#64748b",
        "#334155",  # dark
    ]
    while len(bin_colors) < len(counts):
        bin_colors.append("#4b5563")

    traces: List[Dict[str, object]] = []
    for bi, label in enumerate(bin_labels):
        traces.append(
            {
                "type": "bar",
                "name": label,
                "x": labels,
                "y": counts[bi],
                "marker": {"color": bin_colors[bi % len(bin_colors)]},
                "hovertemplate": "%{x}<br>Bin: " + label + "<br>%{y} passages<extra></extra>",
            }
        )

    layout = {
        "barmode": "stack",
        "margin": {"l": 40, "r": 20, "t": 20, "b": 80},
        "xaxis": {"title": "Frame", "tickangle": -30},
        "yaxis": {"title": "Passages (by probability bin)"},
        "height": 480,
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    }
    return render_plotly_fragment("llm-binned-chart", traces, layout)


def render_plotly_classifier_percentage_bars(
    assignments: Sequence[FrameAssignment],
    frames: Sequence[dict],
    color_map: Dict[str, str],
    classifier_lookup: Optional[Dict[str, Dict[str, object]]] = None,
    threshold: float = 0.5,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
) -> str:
    """Render a bar chart showing normalized percentage per frame using classifier predictions."""
    if not assignments or not frames or not classifier_lookup:
        return ""
    
    # Count passages per frame using classifier predictions (at threshold)
    counts: Dict[str, int] = {str(f["frame_id"]): 0 for f in frames}
    for assignment in assignments:
        pred_entry = classifier_lookup.get(assignment.passage_id)
        if not pred_entry or not isinstance(pred_entry.get("probabilities"), dict):
            continue
        probs = pred_entry["probabilities"]  # type: ignore[index]
        for frame_id in counts.keys():
            prob = float(probs.get(frame_id, 0.0))
            if prob >= threshold:
                counts[frame_id] += 1
    
    # Calculate total and percentages
    total = sum(counts.values())
    if total == 0:
        return ""
    
    # Sort by count desc
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    frame_ids = [fid for fid, _ in ordered]
    values = [int(v) for _, v in ordered]
    percentages = [(v / total * 100) if total > 0 else 0.0 for v in values]
    
    labels = []
    for fid in frame_ids:
        meta = next((f for f in frames if str(f["frame_id"]) == fid), None)
        label = (meta.get("short") or meta.get("name") or fid) if meta else fid
        # Smart wrapping based on max line length
        wrapped_label = wrap_label_html(label, max_len=16)
        labels.append(wrapped_label)
    colors = [color_map.get(fid, "#057d9f") for fid in frame_ids]
    
    traces = [
        {
            "type": "bar",
            "x": labels,
            "y": percentages,
            "marker": {"color": colors},
            "hovertemplate": "%{x}<br>%{y:.1f}%<br>(%{customdata} passages)<extra></extra>",
            "customdata": values,
        }
    ]
    
    layout = {
        "margin": {"l": 40, "r": 20, "t": 20, "b": 0},
        "xaxis": {"title": "", "tickmode": "linear", "tickangle": 0, "automargin": True},
        "yaxis": {"title": ""},
        "height": 500,
        "legend": {"orientation": "h", "yanchor": "top", "y": 1.02, "x": 0, "xanchor": "left"},
    }
    
    return render_plotly_fragment("classifier-percentage-chart", traces, layout)


def render_plotly_classifier_by_year(
    assignments: Sequence[FrameAssignment],
    frames: Sequence[dict],
    color_map: Dict[str, str],
    classifier_lookup: Optional[Dict[str, Dict[str, object]]] = None,
    threshold: float = 0.5,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
) -> str:
    """Render a grouped bar chart with one column per year, using frame colors with transparency."""
    if not assignments or not frames or not classifier_lookup:
        return ""
    
    # Collect counts per frame per year
    frame_year_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    year_totals: Dict[int, int] = defaultdict(int)
    years_set = set()
    today = date.today()
    
    for assignment in assignments:
        pred_entry = classifier_lookup.get(assignment.passage_id)
        if not pred_entry or not isinstance(pred_entry.get("probabilities"), dict):
            continue
        
        # Extract year from metadata with proper date parsing
        published_at = assignment.metadata.get("published_at", "")
        if not published_at:
            continue
        try:
            # Parse the date string properly
            parsed_date = parser.parse(str(published_at))
            # Skip future dates
            if parsed_date.date() > today:
                continue
            # Skip dates before 2020 (for air pollution study focus period)
            if parsed_date.year < 2020:
                continue
            # Skip unreasonably future dates
            if parsed_date.year > today.year:
                continue
            year = parsed_date.year
            years_set.add(year)
        except (ValueError, TypeError, parser.ParserError):
            continue
        
        probs = pred_entry["probabilities"]  # type: ignore[index]
        for frame in frames:
            frame_id = str(frame["frame_id"])
            prob = float(probs.get(frame_id, 0.0))
            if prob >= threshold:
                frame_year_counts[frame_id][year] += 1
                year_totals[year] += 1
    
    if not years_set:
        return ""
    
    years = sorted(years_set)
    base_frame_ids = [str(f["frame_id"]) for f in frames]
    # Order frames by overall importance (total counts across years, desc)
    totals_by_frame: Dict[str, int] = {fid: sum(frame_year_counts[fid].values()) for fid in base_frame_ids}
    frame_ids = sorted(base_frame_ids, key=lambda fid: totals_by_frame.get(fid, 0), reverse=True)
    labels = []
    for fid in frame_ids:
        meta = next((f for f in frames if str(f["frame_id"]) == fid), None)
        label = (meta.get("short") or meta.get("name") or fid) if meta else fid
        labels.append(wrap_label_html(label, max_len=16))
    
    # Create traces - one per year
    traces: List[Dict[str, object]] = []
    num_years = len(years)
    for idx, year in enumerate(years):
        # Calculate alpha: ensure minimum opacity is not too light
        # Older years more transparent, recent years more opaque
        # With ascending year order, higher idx = newer year = higher opacity
        alpha = 0.6 + (idx / max(num_years - 1, 1)) * 0.4  # Range from 0.6 to 1.0
        
        # Get counts for this year and normalize to percentages
        year_total = year_totals[year]
        if year_total > 0:
            percentages = [(frame_year_counts[fid][year] / year_total * 100) for fid in frame_ids]
            counts = [frame_year_counts[fid][year] for fid in frame_ids]
        else:
            percentages = [0.0] * len(frame_ids)
            counts = [0] * len(frame_ids)
        
        # Create colors with transparency
        colors_with_alpha = [hex_to_rgba(color_map.get(fid, "#057d9f"), alpha) for fid in frame_ids]
        
        traces.append({
            "type": "bar",
            "name": str(year),
            "x": labels,
            "y": percentages,
            "customdata": counts,
            "marker": {"color": colors_with_alpha},
            "hovertemplate": "%{x}<br>Year: " + str(year) + "<br>%{y:.1f}%<br>(%{customdata} passages)<extra></extra>",
            # Hide legends for bar traces; we'll add grey-scale legend entries below
            "showlegend": False,
        })

    # Add grey-scale legend entries for years (legend-only traces)
    # Use progressively darker greys for more recent years to mirror alpha progression
    for idx, year in enumerate(years):
        alpha = 0.6 + (idx / max(num_years - 1, 1)) * 0.4
        # Map alpha to grey intensity (lighter for older years)
        intensity = int(min(255, max(60, round(255 * (0.5 + 0.5 * alpha)))))
        grey_rgba = f"rgba({intensity}, {intensity}, {intensity}, 1.0)"
        traces.append({
            "type": "scatter",
            "mode": "markers",
            "x": [None],
            "y": [None],
            "marker": {"size": 10, "color": grey_rgba},
            "name": str(year),
            "showlegend": True,
            "hoverinfo": "skip",
        })
    
    layout = {
        "barmode": "group",
        "margin": {"l": 40, "r": 20, "t": 20, "b": 0},
        "xaxis": {"title": "", "tickmode": "linear", "tickangle": 0, "automargin": True},
        "yaxis": {"title": ""},
        "height": 500,
        "legend": {"orientation": "h", "yanchor": "top", "y": 1.08, "x": 0.5, "xanchor": "center"},
    }
    
    return render_plotly_fragment("classifier-by-year-chart", traces, layout)


def render_occurrence_percentage_bars(
    documents: Sequence[DocumentFrameAggregate],
    frames: Sequence[dict],
    color_map: Dict[str, str],
    *,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    caption: Optional[str] = None,
    export_png_path: Optional[Path] = None,
) -> str:
    """Render occurrence percentage bars chart."""
    if not documents or not frames:
        return ""
    total_docs = len(documents)
    frame_ids = [str(f["frame_id"]) for f in frames]
    counts: Dict[str, int] = {fid: 0 for fid in frame_ids}
    for doc in documents:
        for fid in frame_ids:
            try:
                val = float(doc.frame_scores.get(fid, 0.0))
            except Exception:
                val = 0.0
            if val > 0.0:
                counts[fid] += 1
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    frame_ids = [fid for fid, _ in ordered]
    shares = [((counts[fid] / total_docs)) if total_docs > 0 else 0.0 for fid in frame_ids]
    labels = []
    for fid in frame_ids:
        meta = next((f for f in frames if str(f["frame_id"]) == fid), None)
        label = (meta.get("short") or meta.get("name") or fid) if meta else fid
        labels.append(wrap_label_html(label, max_len=16))
    colors = [color_map.get(fid, "#057d9f") for fid in frame_ids]
    traces = [{
        "type": "bar",
        "x": labels,
        "y": shares,
        "marker": {"color": colors},
        "hovertemplate": "%{x}<br>%{y:.1f}%<extra></extra>",
    }]
    layout = {
        "margin": {"l": 40, "r": 20, "t": 20, "b": 70},
        "xaxis": {"title": "", "tickmode": "linear", "tickangle": 0, "automargin": True},
        "yaxis": {"title": "", "tickformat": ".0%"},
        "height": 420,
        "legend": {"orientation": "h", "yanchor": "top", "y": 1.02, "x": 0, "xanchor": "left"},
    }
    return render_plotly_fragment("occurrence-percentage-chart", traces, layout, export_png_path=export_png_path)


def render_occurrence_by_year(
    documents: Sequence[DocumentFrameAggregate],
    frames: Sequence[dict],
    color_map: Dict[str, str],
    *,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    caption: Optional[str] = None,
    export_png_path: Optional[Path] = None,
) -> str:
    """Render occurrence by year chart."""
    if not documents or not frames:
        return ""
    frame_ids = [str(f["frame_id"]) for f in frames]
    frame_year_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    year_totals: Dict[int, int] = defaultdict(int)
    years_set = set()
    today = date.today()
    for doc in documents:
        published_at = doc.published_at or ""
        try:
            parsed_date = parser.parse(str(published_at))
            if parsed_date.date() > today:
                continue
            year = parsed_date.year
            years_set.add(year)
        except Exception:
            continue
        year_totals[year] += 1
        for fid in frame_ids:
            try:
                val = float(doc.frame_scores.get(fid, 0.0))
            except Exception:
                val = 0.0
            if val > 0.0:
                frame_year_counts[fid][year] += 1
    if not years_set:
        return ""
    years = sorted(years_set)
    totals_by_frame: Dict[str, int] = {fid: sum(frame_year_counts[fid].values()) for fid in frame_ids}
    frame_ids = sorted(frame_ids, key=lambda fid: totals_by_frame.get(fid, 0), reverse=True)
    labels = []
    for fid in frame_ids:
        meta = next((f for f in frames if str(f["frame_id"]) == fid), None)
        label = (meta.get("short") or meta.get("name") or fid) if meta else fid
        labels.append(wrap_label_html(label, max_len=16))

    traces: List[Dict[str, object]] = []
    num_years = len(years)
    
    # Create a more gradual alpha progression
    for idx, year in enumerate(years):
        # More gradual alpha range: 0.3 to 1.0
        # With ascending year order, higher idx = newer year = higher opacity
        alpha = 0.3 + (idx / max(num_years - 1, 1)) * 0.7
        year_total = year_totals[year]
        if year_total > 0:
            shares = [(frame_year_counts[fid][year] / year_total) for fid in frame_ids]
            counts = [frame_year_counts[fid][year] for fid in frame_ids]
        else:
            shares = [0.0] * len(frame_ids)
            counts = [0] * len(frame_ids)
        # Apply transparency to frame colors
        colors_with_alpha = [hex_to_rgba(color_map.get(fid, "#057d9f"), alpha) for fid in frame_ids]
        traces.append({
            "type": "bar",
            "name": str(year),
            "x": labels,
            "y": shares,
            "customdata": counts,
            "marker": {"color": colors_with_alpha},
            "hovertemplate": "%{x}<br>Year: " + str(year) + "<br>%{y:.1f}%<extra></extra>",
            "showlegend": True,
        })
    layout = {
        "barmode": "group",
        "margin": {"l": 40, "r": 20, "t": 20, "b": 40},
        "xaxis": {"title": "", "tickmode": "linear", "tickangle": 0, "automargin": True},
        "yaxis": {"title": "", "tickformat": ".0%"},
        "height": 500,
        "legend": {"orientation": "h", "yanchor": "top", "y": 1.06, "x": 0.5, "xanchor": "center"},
    }
    return render_plotly_fragment("occurrence-by-year-chart", traces, layout, export_png_path=export_png_path)


def render_plotly_timeseries(
    records: Optional[Sequence[Dict[str, object]]],
    frame_lookup: Dict[str, Dict[str, str]],
    color_map: Dict[str, str],
    *,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    caption: Optional[str] = None,
    export_png_path: Optional[Path] = None,
    export_svg: bool = False,
) -> str:
    """Render plotly timeseries chart."""
    if not records:
        print(f"⚠️ render_plotly_timeseries: No records provided (records={records})")
        return ""
    
    print(f"📊 render_plotly_timeseries: Processing {len(records)} records")

    series: Dict[str, List[Tuple[str, float]]] = {}
    items_processed = 0
    items_skipped = 0
    for item in records:
        frame_id = str(item.get("frame_id"))
        date_value = item.get("date")
        if not frame_id or not date_value:
            items_skipped += 1
            continue
        share_value = item.get("value")
        series.setdefault(frame_id, []).append((str(date_value), share_value))
        items_processed += 1
    
    print(f"   Processed {items_processed} items, skipped {items_skipped} items")

    if not series:
        print(f"⚠️ render_plotly_timeseries: No series extracted from records")
        return ""
    
    print(f"📊 render_plotly_timeseries: Extracted {len(series)} frame series")

    traces: List[Dict[str, object]] = []
    for frame_id, points in series.items():
        points.sort(key=lambda entry: entry[0])
        dates = [entry[0] for entry in points]
        values = [entry[1] for entry in points]
        if len(points) > 1:
            df = pd.DataFrame({"date": dates, "share": values})
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date")
            if df.empty:
                continue
            # Data from time_series_30day is already smoothed, so use share directly
            x_vals = df["date"].dt.strftime("%Y-%m-%d").tolist()
            y_vals = df["share"].clip(0, 1).round(5).tolist()
        else:
            x_vals = dates
            y_vals = [round(max(min(v, 1.0), 0.0), 5) for v in values]
        label = frame_lookup.get(frame_id, {}).get("short") or frame_lookup.get(frame_id, {}).get("name") or frame_id
        base_color = color_map.get(frame_id, "#1E3D58")
        traces.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": label,
                "x": x_vals,
                "y": y_vals,
                "stackgroup": "one",
                "line": {"color": hex_to_rgba(base_color, 0.05), "width": 0.0001},
                "fillcolor": hex_to_rgba(base_color, 0.6),
                "hovertemplate": "%{x}<br>%{y:.2%}<extra>" + label + "</extra>",
            }
        )

    layout = {
        "margin": {"l": 60, "r": 30, "t": 30, "b": 60},
        "yaxis": {"tickformat": ".0%", "title": "Share", "range": [0, 1]},
        "xaxis": {"title": "Date"},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        "hovermode": "x unified",
        "height": 520,
    }

    return render_plotly_fragment("time-series-chart", traces, layout, export_png_path=export_png_path, export_svg=export_svg)


def render_plotly_timeseries_lines(
    records: Optional[Sequence[Dict[str, object]]],
    frame_lookup: Dict[str, Dict[str, str]],
    color_map: Dict[str, str],
) -> str:
    """Render plotly timeseries lines chart."""
    if not records:
        return ""

    series: Dict[str, List[Tuple[str, float]]] = {}
    for item in records:
        frame_id = str(item.get("frame_id"))
        date_value = item.get("date")
        value = item.get("value")
        if not frame_id or not date_value or value is None:
            continue
        series.setdefault(frame_id, []).append((str(date_value), value))

    if not series:
        return ""

    traces: List[Dict[str, object]] = []
    for frame_id, points in series.items():
        points.sort(key=lambda entry: entry[0])
        dates = [entry[0] for entry in points]
        values = [entry[1] for entry in points]
        if len(points) > 1:
            df = pd.DataFrame({"date": dates, "share": values})
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date")
            if df.empty:
                continue
            # Data from time_series_30day is already smoothed, so use share directly
            x_vals = df["date"].dt.strftime("%Y-%m-%d").tolist()
            y_vals = df["share"].clip(0, 1).round(5).tolist()
        else:
            x_vals = dates
            y_vals = [round(max(min(v, 1.0), 0.0), 5) for v in values]

        label = frame_lookup.get(frame_id, {}).get("short") or frame_lookup.get(frame_id, {}).get("name") or frame_id
        color = color_map.get(frame_id, "#1E3D58")
        traces.append(
            {
                "type": "scatter",
                "mode": "lines",
                "name": label,
                "x": x_vals,
                "y": y_vals,
                "line": {"color": color, "width": 2},
                "hovertemplate": "%{x}<br>%{y:.2%}<extra>" + label + "</extra>",
            }
        )

    layout = {
        "margin": {"l": 60, "r": 30, "t": 30, "b": 60},
        "yaxis": {"tickformat": ".0%", "title": "Share", "range": [0, 1]},
        "xaxis": {"title": "Date"},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        "hovermode": "x unified",
        "height": 520,
    }

    return render_plotly_fragment("time-series-lines-chart", traces, layout)


def render_plotly_total_docs_timeseries(
    aggregates: Optional[Sequence[DocumentFrameAggregate]],
    *,
    export_png_path: Optional[Path] = None,
    export_svg: bool = False,
) -> str:
    """Render total documents per day with 30-day rolling average."""
    if not aggregates:
        return ""
    
    # Build daily counts from aggregates
    daily_counts: Dict[str, int] = defaultdict(int)
    today = date.today()
    
    for agg in aggregates:
        if not agg.published_at:
            continue
        normalized = normalize_date(agg.published_at)
        if not normalized:
            continue
        day_value = date(normalized.year, normalized.month, normalized.day)
        # Skip future-dated documents
        if day_value > today:
            continue
        daily_counts[day_value.isoformat()] += 1
    
    if not daily_counts:
        return ""
    
    # Build DataFrame with 30-day rolling average
    df = pd.DataFrame([{"date": d, "count": c} for d, c in daily_counts.items()])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    if df.empty:
        return ""
    
    df = df.set_index("date").asfreq("D", fill_value=0)
    df["smooth"] = df["count"].rolling(window=30, min_periods=1).mean()
    
    data = [{
        "type": "scatter",
        "mode": "lines",
        "name": "Articles per day",
        "x": df.index.strftime("%Y-%m-%d").tolist(),
        "y": df["smooth"].tolist(),
        "line": {"color": "#1E3D58", "width": 2.5},
        "hovertemplate": "Date: %{x}<br>Articles (30-day avg): %{y}<extra></extra>"
    }]
    
    layout = {
        "margin": {"l": 60, "r": 30, "t": 30, "b": 60},
        "yaxis": {"title": ""},
        "xaxis": {"title": ""},
        "height": 420,
    }
    
    return render_plotly_fragment("total-docs-timeseries-chart", data=data, layout=layout, export_png_path=export_png_path, export_svg=export_svg)


def render_plotly_domain_counts(
    domain_counts: Optional[Sequence[Tuple[str, int]]],
    total_documents: int = 0,
) -> str:
    """Render plotly domain counts chart."""
    if not domain_counts:
        return ""
    
    # Sort by total count descending and take top 20
    sorted_entries = sorted(domain_counts, key=lambda x: x[1], reverse=True)[:20]
    if not sorted_entries:
        return ""

    domains = [name for name, _ in sorted_entries]
    values = [count for _, count in sorted_entries]
    
    traces = [
        {
            "type": "bar",
            "orientation": "h",
            "y": domains,
            "x": values,
            "marker": {"color": "#057d9f"},
            "hovertemplate": "%{y}<br>%{x} documents<extra></extra>",
        }
    ]

    layout = {
        "margin": {"l": 120, "r": 30, "t": 20, "b": 40},
        "xaxis": {"title": "Number of documents"},
        "yaxis": {"title": "", "autorange": "reversed"},
        "height": max(400, 32 * len(sorted_entries)),
    }

    return render_plotly_fragment("domain-counts-chart", traces, layout)


def extract_domain_from_url(url: Optional[str]) -> str:
    """Extract base domain from URL, ignoring subdomains."""
    if not url:
        return ""
    parsed = urlparse(url)
    netloc = parsed.netloc or parsed.path
    if netloc.startswith("www."):
        netloc = netloc[4:]
    domain = netloc.lower()
    
    # Extract base domain (ignore subdomains)
    # e.g., kota.tribunnews.com -> tribunnews.com
    parts = domain.split('.')
    if len(parts) >= 2:
        # Handle common two-part TLDs like .co.uk, .co.id, .com.au
        if len(parts) >= 3 and parts[-2] in ('co', 'com', 'org', 'net', 'ac', 'gov'):
            return '.'.join(parts[-3:])
        else:
            return '.'.join(parts[-2:])
    
    return domain


def format_date_label(raw: Optional[str]) -> str:
    """Format date label."""
    if not raw:
        return ""
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return raw
    return dt.strftime("%d %b %Y")


def render_global_bar_chart(
    aggregates_data: object,
    frame_lookup: Dict[str, Dict[str, str]],
    color_map: Dict[str, str],
    *,
    export_png_path: Optional[Path] = None,
    chart_id: str = "global-bar-chart",
    export_svg: bool = False,
) -> str:
    """Render a bar chart from global aggregate data."""
    if not aggregates_data:
        print(f"⚠️ render_global_bar_chart: No aggregates_data provided for {chart_id}")
        return ""
    
    # Handle both list and dict formats
    if isinstance(aggregates_data, list) and len(aggregates_data) > 0:
        # If list, extract first PeriodAggregate
        from apps.narrative_framing.aggregation_temporal import PeriodAggregate
        if isinstance(aggregates_data[0], PeriodAggregate):
            doc_count = aggregates_data[0].document_count
            print(f"📊 render_global_bar_chart ({chart_id}): Processing PeriodAggregate with {doc_count} documents")
            aggregates_data = {
                "frame_scores": aggregates_data[0].frame_scores,
                "period_id": aggregates_data[0].period_id,
                "document_count": aggregates_data[0].document_count,
            }
        elif isinstance(aggregates_data[0], dict):
            doc_count = aggregates_data[0].get("document_count", "unknown")
            print(f"📊 render_global_bar_chart ({chart_id}): Processing dict with {doc_count} documents")
            aggregates_data = aggregates_data[0]
    
    if not isinstance(aggregates_data, dict) or "frame_scores" not in aggregates_data:
        print(f"⚠️ render_global_bar_chart ({chart_id}): Invalid aggregates_data format")
        return ""
    
    frame_scores = aggregates_data["frame_scores"]
    doc_count = aggregates_data.get("document_count", "unknown")
    print(f"📊 render_global_bar_chart ({chart_id}): Rendering {len(frame_scores)} frames, doc_count={doc_count}")
    
    if not isinstance(frame_scores, dict):
        print(f"⚠️ render_global_bar_chart ({chart_id}): frame_scores is not a dict")
        return ""
    
    # Sort by score descending
    ordered = sorted(frame_scores.items(), key=lambda kv: float(kv[1] or 0), reverse=True)
    frame_ids = [fid for fid, _ in ordered]
    values = [float(val) for _, val in ordered]
    
    # Debug: print top 3 values
    if len(values) > 0:
        print(f"   Top 3 frame scores: {dict(zip(frame_ids[:3], values[:3]))}")
    
    labels = []
    for fid in frame_ids:
        meta = frame_lookup.get(fid, {})
        label = meta.get("short") or meta.get("name") or fid
        labels.append(wrap_label_html(label, max_len=16))
    
    colors = [color_map.get(fid, "#057d9f") for fid in frame_ids]
    
    traces = [
        {
            "type": "bar",
            "x": labels,
            "y": values,
            "marker": {"color": colors},
            "hovertemplate": "%{x}<br>%{y:.2f}<extra></extra>",
        }
    ]
    
    layout = {
        "margin": {"l": 40, "r": 20, "t": 20, "b": 0},
        "xaxis": {"title": "", "tickmode": "linear", "tickangle": 0, "automargin": True},
        "yaxis": {"title": "", "tickformat": ".0%"},
        "height": 500,
    }
    
    return render_plotly_fragment(chart_id, traces, layout, export_png_path=export_png_path, export_svg=export_svg)


def render_yearly_bar_chart(
    aggregates_data: object,
    frame_lookup: Dict[str, Dict[str, str]],
    color_map: Dict[str, str],
    *,
    export_png_path: Optional[Path] = None,
    chart_id: str = "yearly-bar-chart",
    export_svg: bool = False,
) -> str:
    """Render a grouped bar chart from yearly aggregate data."""
    if not aggregates_data:
        return ""
    
    from apps.narrative_framing.aggregation_temporal import PeriodAggregate
    
    # Convert PeriodAggregate objects to dicts if needed
    if isinstance(aggregates_data, list) and aggregates_data:
        if isinstance(aggregates_data[0], PeriodAggregate):
            aggregates_data = [
                {
                    "period_id": p.period_id,
                    "frame_scores": p.frame_scores,
                    "document_count": p.document_count,
                }
                for p in aggregates_data
            ]
    
    if not isinstance(aggregates_data, list):
        return ""
    
    # Extract years and frame scores
    frame_year_scores: Dict[str, Dict[int, float]] = defaultdict(dict)
    years_set = set()
    
    for entry in aggregates_data:
        if not isinstance(entry, dict):
            continue
        period_id = entry.get("period_id", "")
        frame_scores = entry.get("frame_scores", {})
        if not isinstance(frame_scores, dict):
            continue
        try:
            year = int(period_id)
        except (ValueError, TypeError):
            continue
        years_set.add(year)
        for frame_id, score in frame_scores.items():
            try:
                frame_year_scores[frame_id][year] = float(score)
            except (ValueError, TypeError):
                pass
    
    if not years_set:
        return ""
    
    years = sorted(years_set)
    
    # Sort frames by average score across all years (descending)
    frame_avg_scores = {
        fid: sum(scores.values()) / len(scores) if scores else 0.0
        for fid, scores in frame_year_scores.items()
    }
    frame_ids = sorted(frame_avg_scores.keys(), key=lambda fid: frame_avg_scores[fid], reverse=True)
    
    labels = []
    for fid in frame_ids:
        meta = frame_lookup.get(fid, {})
        label = meta.get("short") or meta.get("name") or fid
        labels.append(wrap_label_html(label, max_len=10))
    
    traces: List[Dict[str, object]] = []
    num_years = len(years)
    
    # Use transparency variation to distinguish years (older = more transparent)
    for idx, year in enumerate(years):
        # With ascending year order, higher idx = newer year = higher opacity
        alpha = 0.6 + (idx / max(num_years - 1, 1)) * 0.4  # Range from 0.6 to 1.0
        scores = [frame_year_scores.get(fid, {}).get(year, 0.0) for fid in frame_ids]
        # Apply transparency to frame colors
        colors_with_alpha = [hex_to_rgba(color_map.get(fid, "#057d9f"), alpha) for fid in frame_ids]
        
        traces.append({
            "type": "bar",
            "name": str(year),
            "x": labels,
            "y": scores,
            "marker": {"color": colors_with_alpha},
            "hovertemplate": f"%{{x}}<br>Year: {year}<br>%{{y:.2f}}<extra></extra>",
            "showlegend": False,  # Hide legend for bars; use grey-scale legend below
        })
    
    # Add grey-scale legend entries for first and last year only to avoid clutter
    legend_years: List[Tuple[int, int]] = []
    if num_years == 1:
        legend_years = [(0, years[0])]
    elif num_years > 1:
        legend_years = [(0, years[0]), (num_years - 1, years[-1])]
    for idx, year in legend_years:
        alpha = 0.6 + (idx / max(num_years - 1, 1)) * 0.4
        color_with_alpha = hex_to_rgba("#1E3D58", alpha)
        traces.append({
            "type": "scatter",
            "mode": "markers",
            "x": [None],
            "y": [None],
            "marker": {"size": 10, "color": color_with_alpha},
            "name": str(year),
            "showlegend": True,
            "hoverinfo": "skip",
        })
    
    layout = {
        "barmode": "group",
        "margin": {"l": 40, "r": 20, "t": 20, "b": 0},
        "xaxis": {"title": "", "tickmode": "linear", "tickangle": 0, "automargin": True},
        "yaxis": {"title": "", "tickformat": ".0%"},
        "height": 500,
        # Legend positioned on top
        "legend": {"orientation": "h", "yanchor": "top", "y": 1.08, "x": 0.5, "xanchor": "center"},
    }
    
    return render_plotly_fragment(chart_id, traces, layout, export_png_path=export_png_path, export_svg=export_svg)


def render_corpus_bar_chart(
    aggregates_data: object,
    frame_lookup: Dict[str, Dict[str, str]],
    *,
    corpus_aliases: Optional[Dict[str, str]] = None,
    export_png_path: Optional[Path] = None,
    chart_id: str = "corpus-bar-chart",
    export_svg: bool = False,
) -> str:
    """Render a grouped bar chart from per-corpus aggregate data."""
    if not isinstance(aggregates_data, list) or not aggregates_data:
        return ""

    # Collect unique corpora and frame scores
    corpora: List[str] = []
    frame_scores_by_corpus: Dict[str, Dict[str, float]] = {}
    for entry in aggregates_data:
        if not isinstance(entry, dict):
            continue
        corpus_name = str(entry.get('corpus', ''))
        scores = entry.get('frame_scores', {})
        if not corpus_name or not isinstance(scores, dict):
            continue
        corpora.append(corpus_name)
        frame_scores_by_corpus[corpus_name] = {str(fid): float(val) for fid, val in scores.items()}

    if not frame_scores_by_corpus:
        return ""

    # Deduplicate corpora in original order
    seen = set()
    corpora = [c for c in corpora if not (c in seen or seen.add(c))]

    # Determine frame order by overall average across corpora
    all_frame_ids: List[str] = []
    for scores in frame_scores_by_corpus.values():
        all_frame_ids.extend(list(scores.keys()))
    frame_ids = sorted(set(all_frame_ids))
    frame_avg_scores: Dict[str, float] = {}
    for fid in frame_ids:
        vals = [frame_scores_by_corpus[c].get(fid, 0.0) for c in corpora]
        frame_avg_scores[fid] = sum(vals) / len(vals) if vals else 0.0
    frame_ids = sorted(frame_ids, key=lambda f: frame_avg_scores.get(f, 0.0), reverse=True)

    # Labels from frame_lookup
    labels = []
    for fid in frame_ids:
        meta = frame_lookup.get(fid, {})
        label = meta.get("short") or meta.get("name") or fid
        labels.append(wrap_label_html(label, max_len=16))

    # Colors per corpus
    corpus_color_map = build_corpus_color_map(corpora)

    # Create traces: one per corpus
    traces: List[Dict[str, object]] = []
    for corpus_name in corpora:
        series = [frame_scores_by_corpus.get(corpus_name, {}).get(fid, 0.0) for fid in frame_ids]
        legend_name = corpus_aliases.get(corpus_name, corpus_name) if corpus_aliases else corpus_name
        traces.append({
            "type": "bar",
            "name": legend_name,
            "x": labels,
            "y": series,
            "marker": {"color": corpus_color_map.get(corpus_name, "#057d9f")},
            "hovertemplate": f"%{{x}}<br>{html.escape(legend_name)}<br>%{{y:.2f}}<extra></extra>",
        })

    layout = {
        "barmode": "group",
        "margin": {"l": 40, "r": 20, "t": 20, "b": 0},
        "xaxis": {"title": "", "tickmode": "linear", "tickangle": 0, "automargin": True},
        "yaxis": {"title": "", "tickformat": ".0%"},
        "height": 500,
        "legend": {"orientation": "h", "yanchor": "top", "y": 1.02, "x": 0, "xanchor": "left"},
    }

    return render_plotly_fragment(chart_id, traces, layout, export_png_path=export_png_path, export_svg=export_svg)


def render_domain_frame_distribution(
    domain_frame_summaries: Sequence[Dict[str, object]],
    frame_lookup: Dict[str, Dict[str, str]],
    color_map: Dict[str, str],
    *,
    export_png_path: Optional[Path] = None,
    export_html_path: Optional[Path] = None,
    n_min_per_media: Optional[int] = None,
    max_domains: int = 20,
    export_svg: bool = False,
) -> str:
    """Render frame distribution across top domains as a Plotly chart with subplots."""
    if not domain_frame_summaries:
        return ""
    
    if not isinstance(domain_frame_summaries, list):
        return ""
    
    # Extract domain frame scores and counts
    domain_frame_scores: Dict[str, Dict[str, float]] = {}
    domain_counts: Dict[str, int] = {}
    
    for entry in domain_frame_summaries:
        if not isinstance(entry, dict):
            continue
        domain = entry.get("domain", "")
        shares = entry.get("shares", {})
        count = entry.get("count", 0)
        if not isinstance(shares, dict):
            continue
        if domain:
            domain_frame_scores[domain] = shares
            domain_counts[domain] = count if isinstance(count, int) else 0
    
    if not domain_frame_scores:
        return ""
    
    # Filter domains by n_min_per_media if specified
    if n_min_per_media is not None and n_min_per_media > 0:
        domain_frame_scores = {
            domain: scores 
            for domain, scores in domain_frame_scores.items()
            if domain_counts.get(domain, 0) >= n_min_per_media
        }
        domain_counts = {
            domain: count
            for domain, count in domain_counts.items()
            if count >= n_min_per_media
        }
    
    if not domain_frame_scores:
        return ""
    
    # Sort domains by count (descending) and take top N
    top_domains = sorted(domain_counts.items(), key=lambda kv: kv[1], reverse=True)[:max_domains]
    
    if not top_domains:
        return ""
    
    domains = [domain for domain, _ in top_domains]
    all_frame_ids = sorted(set().union(*[scores.keys() for scores in domain_frame_scores.values()]))
    
    # Calculate total score per frame across all domains to sort frames
    frame_totals = {}
    for frame_id in all_frame_ids:
        frame_totals[frame_id] = sum(
            domain_frame_scores.get(domain, {}).get(frame_id, 0.0)
            for domain in domains
        )
    
    # Sort frames by total score (descending) - most represented on the left
    frame_ids = sorted(all_frame_ids, key=lambda fid: frame_totals.get(fid, 0.0), reverse=True)
    
    # Create subplots - one per domain
    from plotly.subplots import make_subplots
    
    # Create a grid layout: calculate rows and cols for better faceting
    n_domains = len(domains)
    cols = min(4, max(2, int((n_domains ** 0.5) * 1.2)))
    rows = (n_domains + cols - 1) // cols
    
    # Create subplot titles with bold domain names and (n=x)
    subplot_titles = []
    for domain in domains:
        count = domain_counts.get(domain, 0)
        title = f"<b>{domain}</b> (n={count})"
        subplot_titles.append(title)
    
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
        shared_xaxes=True,
        shared_yaxes=True,
    )
    
    # Prepare frame labels for x-axis (all frames shown in each subplot)
    frame_labels = []
    frame_colors = []
    for frame_id in frame_ids:
        meta = frame_lookup.get(frame_id, {})
        label = meta.get("short") or meta.get("name") or frame_id
        frame_labels.append(label)
        frame_colors.append(color_map.get(frame_id, "#057d9f"))
    
    # Add one trace per frame (for legend) - create invisible traces for legend
    legend_traces = []
    for frame_id, label, color in zip(frame_ids, frame_labels, frame_colors):
        legend_traces.append(
            go.Bar(
                name=label,
                x=[None],
                y=[None],
                marker_color=color,
                showlegend=True,
                legendgroup=label,
            )
        )
    
    # Add one trace per domain subplot - each trace shows all frames as bars
    for idx, domain in enumerate(domains):
        row = (idx // cols) + 1
        col = (idx % cols) + 1
        
        # Collect all frame values for this domain
        y_values = []
        for frame_id in frame_ids:
            value = domain_frame_scores.get(domain, {}).get(frame_id, 0.0)
            y_values.append(value)
        
        # Add one trace with all frames as bars for this domain
        fig.add_trace(
            go.Bar(
                name="",
                x=frame_labels,
                y=y_values,
                marker_color=frame_colors,
                showlegend=False,
                legendgroup="",
            ),
            row=row,
            col=col,
        )
    
    # Add legend traces (invisible, just for legend)
    for trace in legend_traces:
        fig.add_trace(trace)
    
    # Update layout
    fig.update_layout(
        height=max(500, rows * 160),
        margin={"t": 20, "b": 40, "l": 40, "r": 20}, 
        showlegend=True,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": 1.15,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 9},
            "itemwidth": 30,
            "tracegroupgap": 5,
        },
        font={"size": 10},
    )
    
    # Update subplot titles
    fig.update_annotations(font_size=9, yshift=-10)
    
    # Add y=0 line (x-axis) to all subplots and update axes
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if (i - 1) * cols + j <= n_domains:
                fig.update_yaxes(
                    title_text="",
                    tickformat=".0%",
                    row=i,
                    col=j,
                    showgrid=True,
                    gridcolor="#eef2f9",
                    zeroline=True,
                    zerolinecolor="#d3dce7",
                    zerolinewidth=0.5,
                    range=[-0.05, None],
                )
                fig.update_xaxes(
                    title_text="",
                    showticklabels=False,
                    row=i,
                    col=j,
                )
    
    # Export HTML for Jekyll includes if requested
    if export_html_path:
        try:
            export_html_path.parent.mkdir(parents=True, exist_ok=True)
            parent_dir = export_html_path.parent.name
            unique_div_id = f"domain-frame-distribution-{parent_dir.replace('_', '-')}"
            html_content = fig.to_html(
                include_plotlyjs='cdn',
                div_id=unique_div_id,
                config={"displayModeBar": False, "responsive": True}
            )
            css_injection = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        body {
            font-family: 'Inter', 'Segoe UI', sans-serif;
            margin: 0;
            padding: 0;
        }
        .plotly {
            font-family: 'Inter', 'Segoe UI', sans-serif !important;
        }
    </style>
"""
            if '</head>' in html_content:
                html_content = html_content.replace('</head>', css_injection + '</head>')
            else:
                if '<body' in html_content:
                    html_content = html_content.replace('<body', css_injection + '<body')
                else:
                    html_content = css_injection + html_content
            export_html_path.write_text(html_content, encoding="utf-8")
        except Exception as exc:
            print(f"⚠️ Plotly HTML export failed for domain-frame-distribution: {exc}")
    
    # Convert to HTML fragment
    fragment_div_id = "domain-frame-distribution"
    if export_html_path:
        parent_dir = export_html_path.parent.name
        fragment_div_id = f"domain-frame-distribution-{parent_dir.replace('_', '-')}"
    return render_plotly_fragment_from_figure(fig, fragment_div_id, export_png_path=export_png_path, export_svg=export_svg)


def render_plotly_fragment_from_figure(fig, div_id: str, export_png_path: Optional[Path] = None, export_svg: bool = False) -> str:
    """Render a Plotly figure as an HTML fragment."""
    # Export PNG if requested
    if export_png_path:
        try:
            export_png_path = safe_export_path(export_png_path, div_id=div_id)
            export_png_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Update layout for PNG export
            export_layout = {
                "plot_bgcolor": "rgba(0,0,0,0)",
                "paper_bgcolor": "rgba(0,0,0,0)",
                "font": {
                    "family": "Inter, 'Segoe UI', sans-serif",
                    "size": 12,
                    "color": "#1e293b"
                }
            }
            fig.update_layout(**export_layout)
            
            # Export PNG with higher resolution
            fig.write_image(str(export_png_path), scale=3, width=None, height=None)
            
            # Export SVG if requested
            if export_svg:
                try:
                    export_svg_path = export_png_path.with_suffix('.svg')
                    fig.write_image(str(export_svg_path), format='svg', width=None, height=None)
                except Exception as exc:
                    print(f"⚠️ Plotly SVG export failed for {div_id}: {exc}")
            
            # Also export as self-supporting HTML
            html_path = export_png_path.with_suffix('.html')
            html_content = fig.to_html(
                include_plotlyjs='cdn',
                div_id=div_id,
                config={"displayModeBar": False, "responsive": True}
            )
            css_injection = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        body {
            font-family: 'Inter', 'Segoe UI', sans-serif;
            margin: 0;
            padding: 0;
        }
        .plotly {
            font-family: 'Inter', 'Segoe UI', sans-serif !important;
        }
    </style>
"""
            if '</head>' in html_content:
                html_content = html_content.replace('</head>', css_injection + '</head>')
            else:
                if '<body' in html_content:
                    html_content = html_content.replace('<body', css_injection + '<body')
                else:
                    html_content = css_injection + html_content
            html_path.write_text(html_content)
        except Exception as exc:
            print(f"⚠️ Plotly PNG export failed for {div_id}: {exc}")
    
    # Generate Plotly HTML
    plotly_html = fig.to_html(include_plotlyjs=False, div_id=div_id, config={"displayModeBar": False, "responsive": True})
    
    # Extract just the div and script parts
    if '<div id="' in plotly_html and '</script>' in plotly_html:
        div_start = plotly_html.find('<div id="')
        div_end = plotly_html.find('</div>', div_start) + 6
        script_start = plotly_html.find('<script type="text/javascript">', div_end)
        script_end = plotly_html.find('</script>', script_start) + 9
        
        div_part = plotly_html[div_start:div_end]
        script_part = plotly_html[script_start:script_end]
        
        return div_part + script_part
    
    return f'<div id="{div_id}" class="plotly-chart"></div>'


def collect_top_stories_by_frame(
    document_aggregates_weighted: Optional[Sequence[DocumentFrameAggregate]],
    *,
    top_n: int = 3,
    corpus_index: Optional[Dict[str, dict]] = None,
) -> Dict[str, List[Dict[str, object]]]:
    """Collect top stories by frame.
    
    Args:
        document_aggregates_weighted: Document aggregates with frame scores
        top_n: Number of top stories to return per frame
        corpus_index: Optional corpus index mapping doc_id to metadata (for looking up titles)
    """
    top_stories: Dict[str, List[Dict[str, object]]] = {}
    if not document_aggregates_weighted:
        return top_stories

    for aggregate in document_aggregates_weighted:
        for frame_id, score in aggregate.frame_scores.items():
            value = float(score)
            if value <= 0:
                continue
            
            # Try to get title from aggregate, then from corpus index, then fall back to doc_id
            title = aggregate.title
            if not title and corpus_index:
                # Try direct lookup first
                doc_meta = corpus_index.get(aggregate.doc_id, {})
                title = doc_meta.get("title") or doc_meta.get("name")
                
                # If not found and doc_id is in global format, try local_id
                if not title:
                    local_id = aggregate.doc_id
                    # Handle global doc_id formats: corpus@@local_id or corpus::local_id
                    if "@@" in aggregate.doc_id:
                        from efi_analyser.frames.identifiers import split_global_doc_id
                        _, local_id = split_global_doc_id(aggregate.doc_id)
                    elif "::" in aggregate.doc_id:
                        local_id = aggregate.doc_id.split("::", 1)[1]
                    
                    if local_id != aggregate.doc_id:
                        doc_meta = corpus_index.get(local_id, {})
                        title = doc_meta.get("title") or doc_meta.get("name")
            
            stories = top_stories.setdefault(frame_id, [])
            stories.append(
                {
                    "score": value,
                    "title": title or aggregate.doc_id,
                    "url": aggregate.url or "",
                    "published_at": aggregate.published_at or "",
                    "doc_id": aggregate.doc_id,
                    "domain": extract_domain_from_url(aggregate.url),
                }
            )

    for frame_id, stories in top_stories.items():
        stories.sort(key=lambda item: item["score"], reverse=True)
        top_stories[frame_id] = stories[: max(1, top_n)]

    return top_stories


__all__ = [
    "build_color_map",
    "build_corpus_color_map",
    "sanitize_filename",
    "safe_export_path",
    "compute_classifier_metrics",
    "plot_precision_recall_bars",
    "plot_domain_counts_bar",
    "fig_to_base64",
    "wrap_label_html",
    "wrap_text_html",
    "hex_to_rgba",
    "render_probability_bars",
    "render_plotly_fragment",
    "render_plotly_llm_coverage",
    "render_plotly_llm_binned_distribution",
    "render_plotly_classifier_percentage_bars",
    "render_plotly_classifier_by_year",
    "render_occurrence_percentage_bars",
    "render_occurrence_by_year",
    "render_plotly_timeseries",
    "render_plotly_timeseries_lines",
    "render_plotly_total_docs_timeseries",
    "render_plotly_domain_counts",
    "extract_domain_from_url",
    "format_date_label",
    "render_global_bar_chart",
    "render_yearly_bar_chart",
    "render_corpus_bar_chart",
    "render_domain_frame_distribution",
    "render_plotly_fragment_from_figure",
    "collect_top_stories_by_frame",
]
