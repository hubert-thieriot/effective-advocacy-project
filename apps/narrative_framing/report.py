"""HTML report generation for the narrative framing workflow."""

from __future__ import annotations

import base64
import html
import json
import textwrap
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from apps.narrative_framing.aggregation import DocumentFrameAggregate
from efi_analyser.frames import FrameAssignment, FrameSchema

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


def _build_color_map(frames) -> Dict[str, str]:
    color_map: Dict[str, str] = {}
    for idx, frame in enumerate(frames):
        color_map[frame.frame_id] = _PALETTE[idx % len(_PALETTE)]
    return color_map


def _compute_classifier_metrics(
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


def _plot_precision_recall_bars(metrics: Dict[str, Dict[str, float]], frame_names: Dict[str, str], color_map: Dict[str, str]) -> str:
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
    return _fig_to_base64(fig)


def _plot_auc_bars(metrics: Dict[str, Dict[str, float]], frame_names: Dict[str, str], color_map: Dict[str, str]) -> str:
    """Create a single bar chart highlighting AUC scores per frame."""
    frames = list(metrics.keys())
    auc_scores = [metrics[f]["auc"] for f in frames]
    labels = [frame_names.get(f, f) for f in frames]
    colors = [color_map.get(f, "#1E3D58") for f in frames]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(frames)), auc_scores, color=colors, alpha=0.85)
    ax.set_xticks(range(len(frames)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUC", fontsize=12)
    ax.set_title("AUC by Frame", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, auc_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    return _fig_to_base64(fig)


def _render_plotly_llm_coverage(
    assignments: Sequence[FrameAssignment],
    frames: Sequence[dict],
    color_map: Dict[str, str],
) -> str:
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
        "height": 420,
    }
    return _render_plotly_fragment("llm-coverage-chart", traces, layout)


def _render_plotly_llm_binned_distribution(
    assignments: Sequence[FrameAssignment],
    frames: Sequence[dict],
    *,
    bins: Optional[Sequence[float]] = None,
) -> str:
    """Render a stacked bar chart of LLM probabilities binned per frame.

    - X axis: frames (short label)
    - Y axis: count of passages in each probability bin for that frame
    - Stacks: probability bins (e.g., 0–0.2, 0.2–0.4, ...)
    """
    if not assignments or not frames:
        return ""
    import math

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
    return _render_plotly_fragment("llm-binned-chart", traces, layout)


def _plot_confusion_matrix(
    assignments: Sequence[FrameAssignment],
    frame_ids: List[str],
    frame_names: Dict[str, str],
    threshold: float = 0.5,
    classifier_lookup: Optional[Dict[str, Dict[str, object]]] = None,
) -> str:
    """Create a confusion matrix heatmap for frame predictions."""
    # Create binary labels for each frame
    y_true_all = []
    y_pred_all = []
    
    for frame_id in frame_ids:
        y_true = []
        y_pred = []
        
        for assignment in assignments:
            true_label = 1 if frame_id in assignment.top_frames else 0
            if classifier_lookup is not None:
                pred_entry = classifier_lookup.get(assignment.passage_id)
                prob = 0.0
                if pred_entry and isinstance(pred_entry.get("probabilities"), dict):
                    prob = float(pred_entry["probabilities"].get(frame_id, 0.0))  # type: ignore[index]
            else:
                prob = assignment.probabilities.get(frame_id, 0.0)
            pred_label = 1 if prob >= threshold else 0
            y_true.append(true_label)
            y_pred.append(pred_label)
        
        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)
    
    # Create confusion matrix
    cm = confusion_matrix(y_true_all, y_pred_all)
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    ax.set_title('Confusion Matrix (All Frames Combined)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def _plot_roc_curves(
    assignments: Sequence[FrameAssignment],
    frame_ids: List[str],
    frame_names: Dict[str, str],
    color_map: Dict[str, str],
    classifier_lookup: Optional[Dict[str, Dict[str, object]]] = None,
) -> str:
    """Create ROC curves for each frame."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for frame_id in frame_ids:
        y_true = []
        y_scores = []
        
        for assignment in assignments:
            true_label = 1 if frame_id in assignment.top_frames else 0
            if classifier_lookup is not None:
                pred_entry = classifier_lookup.get(assignment.passage_id)
                if pred_entry and isinstance(pred_entry.get("probabilities"), dict):
                    score = float(pred_entry["probabilities"].get(frame_id, 0.0))  # type: ignore[index]
                else:
                    score = 0.0
            else:
                score = assignment.probabilities.get(frame_id, 0.0)
            y_true.append(true_label)
            y_scores.append(score)
        
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        
        if len(np.unique(y_true)) > 1:  # Only plot if both classes present
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            auc = roc_auc_score(y_true, y_scores)
            
            color = color_map.get(frame_id, '#4F8EF7')
            label = frame_names.get(frame_id, frame_id)
            ax.plot(fpr, tpr, color=color, linewidth=2, 
                   label=f'{label} (AUC = {auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves by Frame', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def _plot_performance_dashboard(metrics: Dict[str, Dict[str, float]], frame_names: Dict[str, str], color_map: Dict[str, str]) -> str:
    """Create a comprehensive performance dashboard."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    frames = list(metrics.keys())
    labels = [frame_names.get(f, f) for f in frames]
    colors = [color_map.get(f, '#4F8EF7') for f in frames]
    
    # F1 scores
    f1_scores = [metrics[f]['f1'] for f in frames]
    bars1 = ax1.bar(range(len(frames)), f1_scores, color=colors, alpha=0.8)
    ax1.set_xticks(range(len(frames)))
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_ylabel('F1 Score', fontsize=12)
    ax1.set_title('F1 Scores by Frame', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars1, f1_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # AUC scores
    auc_scores = [metrics[f]['auc'] for f in frames]
    bars2 = ax2.bar(range(len(frames)), auc_scores, color=colors, alpha=0.8)
    ax2.set_xticks(range(len(frames)))
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.set_ylabel('AUC Score', fontsize=12)
    ax2.set_title('AUC Scores by Frame', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars2, auc_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Precision vs Recall scatter
    precisions = [metrics[f]['precision'] for f in frames]
    recalls = [metrics[f]['recall'] for f in frames]
    ax3.scatter(recalls, precisions, c=colors, s=100, alpha=0.8, edgecolors='black', linewidth=1)
    
    # Add frame labels
    for i, label in enumerate(labels):
        ax3.annotate(label, (recalls[i], precisions[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax3.set_xlabel('Recall', fontsize=12)
    ax3.set_ylabel('Precision', fontsize=12)
    ax3.set_title('Precision vs Recall', fontsize=14, fontweight='bold')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.grid(alpha=0.3)
    
    # Support (number of positive examples)
    supports = [metrics[f]['support'] for f in frames]
    bars4 = ax4.bar(range(len(frames)), supports, color=colors, alpha=0.8)
    ax4.set_xticks(range(len(frames)))
    ax4.set_xticklabels(labels, rotation=45, ha='right')
    ax4.set_ylabel('Support (Positive Examples)', fontsize=12)
    ax4.set_title('Support by Frame', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars4, supports):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{int(val)}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def _plot_domain_counts_bar(domain_counts: Sequence[Tuple[str, int]]) -> str:
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
    return _fig_to_base64(fig)


def _plot_domain_frame_facets(
    domain_frame_summaries: Sequence[Dict[str, object]],
    frame_lookup: Dict[str, Dict[str, str]],
    color_map: Dict[str, str],
) -> str:
    if not domain_frame_summaries:
        return ""

    frame_ids = list(frame_lookup.keys())
    frame_labels = [frame_lookup[fid]["short"] for fid in frame_ids]
    colors = [color_map.get(fid, "#4F8EF7") for fid in frame_ids]

    total_domains = len(domain_frame_summaries)
    cols = min(5, max(1, int(math.ceil(math.sqrt(total_domains)))))
    rows = int(math.ceil(total_domains / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 2.8))
    axes_array = np.atleast_1d(axes).flatten()

    for ax in axes_array[total_domains:]:
        ax.axis("off")

    for idx, summary in enumerate(domain_frame_summaries):
        ax = axes_array[idx]
        shares = summary.get("shares", {})
        values = [float(shares.get(fid, 0.0)) for fid in frame_ids]
        ax.bar(range(len(frame_ids)), values, color=colors, alpha=0.9)
        ymax = max(values) if values else 0.0
        if ymax <= 0:
            upper = 1.0
        else:
            upper = min(1.0, ymax * 1.15) if ymax < 1.0 else ymax * 1.05
        upper = max(0.1, upper)
        ax.set_ylim(0, upper)
        ax.set_xticks(range(len(frame_ids)))
        ax.set_xticklabels(frame_labels, rotation=45, ha="right", fontsize=7)
        count = summary.get("count")
        subtitle = summary.get("domain", "")
        if count is not None:
            subtitle = f"{subtitle}\n(n={count})"
        ax.set_title(subtitle, fontsize=9)
        ax.grid(axis="y", alpha=0.2)

    fig.suptitle("Frame distribution across top domains", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _fig_to_base64(fig)


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string."""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    return image_base64


def _generate_metrics_table_rows(
    metrics: Dict[str, Dict[str, float]],
    frame_names: Dict[str, str],
    *,
    include_f1: bool = True,
) -> str:
    """Generate HTML table rows for metrics display."""
    rows = []
    for frame_id, frame_metrics in metrics.items():
        frame_name = frame_names.get(frame_id, frame_id)
        cells = [
            "<tr>",
            f"<td>{html.escape(frame_name)}</td>",
            f"<td>{frame_metrics['precision']:.3f}</td>",
            f"<td>{frame_metrics['recall']:.3f}</td>",
        ]
        if include_f1:
            cells.append(f"<td>{frame_metrics['f1']:.3f}</td>")
        cells.extend(
            [
                f"<td>{frame_metrics['auc']:.3f}</td>",
                f"<td>{int(frame_metrics['support'])}</td>",
                "</tr>",
            ]
        )
        rows.append("".join(cells))
    return "".join(rows)


def _render_probability_bars(
    probabilities: Dict[str, float],
    frame_lookup: Dict[str, Dict[str, str]],
    color_map: Dict[str, str],
) -> str:
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


def _render_plotly_fragment(
    div_id: str,
    data: Sequence[Dict[str, object]],
    layout: Dict[str, object],
    *,
    config: Optional[Dict[str, object]] = None,
) -> str:
    if not data:
        return ""
    config = config or {"displayModeBar": False, "responsive": True}
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


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
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


def _render_plotly_timeseries(
    records: Optional[Sequence[Dict[str, object]]],
    frame_lookup: Dict[str, Dict[str, str]],
    color_map: Dict[str, str],
) -> str:
    if not records:
        return ""

    series: Dict[str, List[Tuple[str, float]]] = {}
    for item in records:
        frame_id = str(item.get("frame_id"))
        date_value = item.get("date")
        if not frame_id or not date_value:
            continue
        share_value = item.get("share")
        if share_value is None:
            share_value = item.get("avg_score", 0.0)
        try:
            share = float(share_value)
        except (TypeError, ValueError):
            share = 0.0
        series.setdefault(frame_id, []).append((str(date_value), share))

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
            df["smooth"] = df["share"].rolling(window=30, min_periods=1).mean()
            x_vals = df["date"].dt.strftime("%Y-%m-%d").tolist()
            y_vals = df["smooth"].clip(0, 1).round(5).tolist()
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
                "line": {"color": _hex_to_rgba(base_color, 0.05), "width": 0.0001},
                "fillcolor": _hex_to_rgba(base_color, 0.6),
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

    return _render_plotly_fragment("time-series-chart", traces, layout)


def _render_plotly_timeseries_lines(
    records: Optional[Sequence[Dict[str, object]]],
    frame_lookup: Dict[str, Dict[str, str]],
    color_map: Dict[str, str],
) -> str:
    if not records:
        return ""

    series: Dict[str, List[Tuple[str, float]]] = {}
    for item in records:
        frame_id = str(item.get("frame_id"))
        date_value = item.get("date")
        if not frame_id or not date_value:
            continue
        share_value = item.get("share")
        if share_value is None:
            share_value = item.get("avg_score", 0.0)
        try:
            share = float(share_value)
        except (TypeError, ValueError):
            share = 0.0
        series.setdefault(frame_id, []).append((str(date_value), share))

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
            df["smooth"] = df["share"].rolling(window=30, min_periods=1).mean()
            x_vals = df["date"].dt.strftime("%Y-%m-%d").tolist()
            y_vals = df["smooth"].clip(0, 1).round(5).tolist()
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

    return _render_plotly_fragment("time-series-lines-chart", traces, layout)


def _render_plotly_timeseries_abs_lines(
    records: Optional[Sequence[Dict[str, object]]],
    frame_lookup: Dict[str, Dict[str, str]],
    color_map: Dict[str, str],
) -> str:
    """Line chart using absolute (average) frame scores per day, smoothed 30-day."""
    if not records:
        return ""

    series: Dict[str, List[Tuple[str, float]]] = {}
    for item in records:
        frame_id = str(item.get("frame_id"))
        date_value = item.get("date")
        if not frame_id or not date_value:
            continue
        try:
            value = float(item.get("avg_score", 0.0))
        except (TypeError, ValueError):
            value = 0.0
        series.setdefault(frame_id, []).append((str(date_value), value))

    if not series:
        return ""

    traces: List[Dict[str, object]] = []
    for frame_id, points in series.items():
        points.sort(key=lambda entry: entry[0])
        dates = [entry[0] for entry in points]
        values = [entry[1] for entry in points]
        if len(points) > 1:
            df = pd.DataFrame({"date": dates, "value": values})
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date")
            if df.empty:
                continue
            df["smooth"] = df["value"].rolling(window=30, min_periods=1).mean()
            x_vals = df["date"].dt.strftime("%Y-%m-%d").tolist()
            y_vals = df["smooth"].clip(0, 1).round(4).tolist()
        else:
            x_vals = dates
            y_vals = [round(max(min(v, 1.0), 0.0), 4) for v in values]

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
                "hovertemplate": "%{x}<br>%{y:.3f}<extra>" + label + "</extra>",
            }
        )

    layout = {
        "margin": {"l": 60, "r": 30, "t": 30, "b": 60},
        "yaxis": {"title": "Average Score", "range": [0, 1]},
        "xaxis": {"title": "Date"},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        "hovermode": "x unified",
        "height": 520,
    }

    return _render_plotly_fragment("time-series-abs-lines-chart", traces, layout)

def _render_plotly_domain_counts(
    domain_counts: Optional[Sequence[Tuple[str, int]]],
) -> str:
    if not domain_counts:
        return ""
    top_entries = domain_counts[:20]
    if not top_entries:
        return ""

    domains = [name for name, _ in top_entries]
    values = [int(count) for _, count in top_entries]

    traces = [
        {
            "type": "bar",
            "orientation": "h",
            "y": domains[::-1],
            "x": values[::-1],
            "marker": {"color": "#057d9f"},
            "hovertemplate": "%{y}<br>%{x} documents<extra></extra>",
        }
    ]

    layout = {
        "margin": {"l": 120, "r": 30, "t": 20, "b": 40},
        "xaxis": {"title": "Documents"},
        "yaxis": {"title": "Domain"},
        "height": max(320, 32 * len(top_entries)),
    }
    return _render_plotly_fragment("domain-counts-chart", traces, layout)


def _extract_domain_from_url(url: Optional[str]) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    netloc = parsed.netloc or parsed.path
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc.lower()


def _format_date_label(raw: Optional[str]) -> str:
    if not raw:
        return ""
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return raw
    return dt.strftime("%d %b %Y")


def _collect_top_stories_by_frame(
    document_aggregates: Optional[Sequence[DocumentFrameAggregate]],
    *,
    top_n: int = 3,
) -> Dict[str, List[Dict[str, object]]]:
    top_stories: Dict[str, List[Dict[str, object]]] = {}
    if not document_aggregates:
        return top_stories

    for aggregate in document_aggregates:
        for frame_id, score in aggregate.frame_scores.items():
            value = float(score)
            if value <= 0:
                continue
            stories = top_stories.setdefault(frame_id, [])
            stories.append(
                {
                    "score": value,
                    "title": aggregate.title or aggregate.doc_id,
                    "url": aggregate.url or "",
                    "published_at": aggregate.published_at or "",
                    "doc_id": aggregate.doc_id,
                    "domain": _extract_domain_from_url(aggregate.url),
                }
            )

    for frame_id, stories in top_stories.items():
        stories.sort(key=lambda item: item["score"], reverse=True)
        top_stories[frame_id] = stories[: max(1, top_n)]

    return top_stories


def write_html_report(
    schema: FrameSchema,
    assignments: Sequence[FrameAssignment],
    output_path: Path,
    classifier_lookup: Optional[Dict[str, Dict[str, object]]] = None,
    *,
    global_frame_share: Optional[Dict[str, float]] = None,
    timeseries_records: Optional[Sequence[Dict[str, object]]] = None,
    classified_documents: int = 0,
    classifier_sample_limit: Optional[int] = None,
    area_chart_b64: Optional[str] = None,
    include_classifier_plots: bool = True,
    domain_counts: Optional[Sequence[Tuple[str, int]]] = None,
    domain_frame_summaries: Optional[Sequence[Dict[str, object]]] = None,
    document_aggregates: Optional[Sequence[DocumentFrameAggregate]] = None,
    corpus_frame_summaries: Optional[Sequence[Dict[str, object]]] = None,
    metrics_threshold: float = 0.3,
) -> None:
    """Render a compact HTML report for frame assignments."""

    color_map = _build_color_map(schema.frames)
    frame_lookup = {
        frame.frame_id: {
            "name": frame.name,
            "short": frame.short_name or (frame.name.split()[0] if frame.name else frame.frame_id),
            "description": frame.description,
            "keywords": ", ".join(frame.keywords),
        }
        for frame in schema.frames
    }
    # Generate classifier performance plots if requested
    classifier_plots_html = ""
    if include_classifier_plots and assignments:
        try:
            frame_ids = [frame.frame_id for frame in schema.frames]
            frame_names = {frame.frame_id: frame.name for frame in schema.frames}

            metrics = _compute_classifier_metrics(
                assignments,
                frame_ids,
                threshold=metrics_threshold,
                classifier_lookup=classifier_lookup,
            )
            precision_recall_b64 = _plot_precision_recall_bars(metrics, frame_names, color_map)

            classifier_plots_html = (
                "<h3>Classifying Results</h3>"
                "<div class=\"card chart-card\">"
                "<h4>Precision &amp; Recall</h4>"
                f"<img src=\"data:image/png;base64,{precision_recall_b64}\" alt=\"Precision and Recall by Frame\" />"
                "</div>"
            )
        except Exception as exc:
            classifier_plots_html = f"""
            <h3>Classifying Results</h3>
            <div class="card chart-card">
                <p class="error-note">Error generating classifier performance plots: {html.escape(str(exc))}</p>
            </div>
            """

    frame_cards: List[str] = []
    for frame in schema.frames:
        color = color_map.get(frame.frame_id, "#1E3D58")
        short_label = frame.short_name or (frame.name.split()[0] if frame.name else frame.frame_id)
        share_badge = ""
        if global_frame_share and frame.frame_id in global_frame_share:
            share_value = max(0.0, float(global_frame_share[frame.frame_id]))
            share_badge = f"<span class=\"share-badge\">{share_value * 100:.0f}%</span>"
        card_parts = [
            f"<article class=\"frame-card\" style=\"--accent-color:{color};\">",
            share_badge,
            f"<h3>{html.escape(short_label)}</h3>",
            f"<p class=\"frame-card-title\">{html.escape(frame.name)}</p>",
        ]
        if frame.description:
            card_parts.append(
                f"<p class=\"frame-card-text\">{html.escape(frame.description)}</p>"
            )
        if frame.keywords:
            card_parts.append(
                f"<p class=\"frame-card-meta\"><strong>Keywords:</strong> {html.escape(', '.join(frame.keywords))}</p>"
            )
        if frame.examples:
            card_parts.append(
                f"<p class=\"frame-card-meta\"><strong>Examples:</strong> {html.escape('; '.join(frame.examples[:2]))}</p>"
            )
        card_parts.append("</article>")
        frame_cards.append("".join(card_parts))

    coverage_text = "No documents were classified."
    if classified_documents > 0:
        coverage_text = f"Classifier applied to {classified_documents} documents."

    timeseries_note = ""
    if timeseries_records:
        date_values = [str(item.get("date", "")) for item in timeseries_records if item.get("date")]
        if date_values:
            start = min(date_values)
            end = max(date_values)
            start_label = _format_date_label(start)
            end_label = _format_date_label(end)
            if start_label and end_label:
                timeseries_note = f"{start_label} – {end_label}"
            else:
                timeseries_note = f"Data covers {html.escape(start)} to {html.escape(end)}."

    timeseries_chart_html = _render_plotly_timeseries(timeseries_records, frame_lookup, color_map)
    timeseries_lines_html = _render_plotly_timeseries_lines(timeseries_records, frame_lookup, color_map)
    timeseries_abs_lines_html = _render_plotly_timeseries_abs_lines(timeseries_records, frame_lookup, color_map)
    if not timeseries_chart_html and area_chart_b64:
        timeseries_chart_html = (
            "<figure class=\"chart\">"
            f"<img src=\"data:image/png;base64,{area_chart_b64}\" alt=\"Frame share over time\" />"
            "<figcaption>Frame share over time (stacked area chart).</figcaption>"
            "</figure>"
        )

    domain_counts_chart_html = _render_plotly_domain_counts(domain_counts)
    if not domain_counts_chart_html and domain_counts:
        domain_counts_b64 = _plot_domain_counts_bar(domain_counts[:20])
        if domain_counts_b64:
            domain_counts_chart_html = (
                "<figure class=\"chart\">"
                f"<img src=\"data:image/png;base64,{domain_counts_b64}\" alt=\"Top domains by document count\" />"
                "<figcaption>Top domains ranked by number of classified documents.</figcaption>"
                "</figure>"
            )

    domain_frame_chart_html = ""
    if domain_frame_summaries:
        ordered_domain_frames = [entry for entry in domain_frame_summaries if entry.get("shares")]
        if ordered_domain_frames:
            domain_frame_b64 = _plot_domain_frame_facets(ordered_domain_frames, frame_lookup, color_map)
            if domain_frame_b64:
                domain_frame_chart_html = (
                    "<figure class=\"chart\">"
                    f"<img src=\"data:image/png;base64,{domain_frame_b64}\" alt=\"Frame distribution by domain\" />"
                    "<figcaption>Frame score distribution for the top domains. Each subplot shows average frame scores across documents from that domain.</figcaption>"
                    "</figure>"
                )

    rows = []
    for assignment in assignments:
        llm_probs_html = _render_probability_bars(assignment.probabilities, frame_lookup, color_map)

        classifier_html = "—"
        if classifier_lookup and assignment.passage_id in classifier_lookup:
            entry = classifier_lookup[assignment.passage_id]
            probs = entry.get("probabilities", {})
            if isinstance(probs, dict) and probs:
                classifier_html = _render_probability_bars(
                    {fid: float(score) for fid, score in probs.items()},
                    frame_lookup,
                    color_map,
        )

        rationale = html.escape(assignment.rationale) if assignment.rationale else "—"
        evidence_text = (
            "<br/>".join(html.escape(span) for span in assignment.evidence_spans)
            if assignment.evidence_spans
            else "—"
        )

        metadata = assignment.metadata if isinstance(assignment.metadata, dict) else {}
        url = metadata.get("url") or ""
        doc_folder_path = metadata.get("doc_folder_path") or ""
        
        url_icon = (
            f"<a class=\"link-icon\" href=\"{html.escape(url)}\" target=\"_blank\" rel=\"noopener noreferrer\" title=\"Open original URL\">🔗</a>"
            if url
            else ""
        )
        folder_icon = (
            f"<a class=\"link-icon\" href=\"file://{html.escape(doc_folder_path)}\" title=\"Open document folder\">📁</a>"
            if doc_folder_path
            else ""
        )
        
        link_icons = f"{url_icon}{folder_icon}"
        passage_html = f"{link_icons}<span class=\"passage-text\">{html.escape(assignment.passage_text)}</span>"

        rows.append(
            "<tr>"
            f"<td class=\"passage\">{passage_html}</td>"
            f"<td>{llm_probs_html}</td>"
            f"<td>{classifier_html}</td>"
            f"<td>{rationale}</td>"
            f"<td>{evidence_text}</td>"
            "</tr>"
        )

    table_html = "\n".join(rows) if rows else "<tr><td colspan=5>No assignments available</td></tr>"

    frame_cards_html = (
        f"<div class=\"frame-card-grid\">{''.join(frame_cards)}</div>"
        if frame_cards
        else "<p class=\"empty-note\">Schema does not define any frames.</p>"
    )

    frames_section_html = f"""
        <section class=\"report-section\" id=\"frames\">
            <header class=\"section-heading\">
                <h2>Frames</h2>
                <p>Schema definition and guidance.</p>
            </header>
            <div class=\"section-body frames-body\">
                {frame_cards_html}
            </div>
        </section>
    """

    # LLM application charts
    llm_coverage_section_html = ""
    llm_binned_section_html = ""
    if assignments:
        frames_as_dicts = [
            {"frame_id": f.frame_id, "name": f.name, "short": f.short_name}
            for f in schema.frames
        ]
        llm_cov_html = _render_plotly_llm_coverage(assignments, frames_as_dicts, color_map)
        if llm_cov_html:
            llm_coverage_section_html = f"""
        <section class=\"report-section\" id=\"llm-coverage\">
            <header class=\"section-heading\">
                <h2>LLM Frame Coverage</h2>
                <p>Passages per frame based on LLM assignments (top_k).</p>
            </header>
            <div class=\"section-body\">
                {llm_cov_html}
                <p class=\"chart-note\">Each bar counts sampled passages where the frame appears in LLM top_k.</p>
            </div>
        </section>
        """

        llm_binned_html = _render_plotly_llm_binned_distribution(assignments, frames_as_dicts)
        if llm_binned_html:
            llm_binned_section_html = f"""
        <section class=\"report-section\" id=\"llm-bins\">
            <header class=\"section-heading\">
                <h2>LLM Probability Distribution</h2>
                <p>Distribution of LLM probabilities per frame (stacked bins).</p>
            </header>
            <div class=\"section-body\">
                {llm_binned_html}
                <p class=\"chart-note\">Frames on X axis; stacks represent probability bins (0–0.2, 0.2–0.4, …).</p>
            </div>
        </section>
        """

    time_series_section_html = ""
    if timeseries_note or timeseries_chart_html:
        note_parts: List[str] = []
        if timeseries_note:
            note_parts.append(html.escape(timeseries_note))
        note_parts.append("30-day rolling average of frame share.")
        note_html = f"<p class=\"section-note\">{' • '.join(note_parts)}</p>"
        chart_inner = ""
        if timeseries_chart_html:
            chart_inner += timeseries_chart_html
        if timeseries_lines_html:
            chart_inner += timeseries_lines_html
        if timeseries_abs_lines_html:
            chart_inner += timeseries_abs_lines_html
        chart_block = (
            f"<div class=\"card chart-card\">{chart_inner}</div>"
            if chart_inner
            else "<div class=\"card chart-card\"><p class=\"empty-note\">Not enough data to show the time series.</p></div>"
        )
        time_series_section_html = f"""
        <section class=\"report-section\" id=\"time-series\">
            <header class=\"section-heading\">
                <h2>Time Series</h2>
                <p>Frame share momentum across the observation window.</p>
            </header>
            <div class=\"section-body\">
                {note_html}
                {chart_block}
            </div>
        </section>
        """

    domain_counts_card = ""
    if domain_counts_chart_html:
        domain_counts_card = (
            "<div class=\"card chart-card\">"
            "<h3>Top Media Sources</h3>"
            f"{domain_counts_chart_html}"
            "<p class=\"chart-note\">Classified document counts for the leading domains.</p>"
            "</div>"
        )

    domain_distribution_card = ""
    if domain_frame_chart_html:
        domain_distribution_card = (
            "<div class=\"card chart-card\">"
            "<h3>Frame Mix by Source</h3>"
            f"{domain_frame_chart_html}"
            "<p class=\"chart-note\">Average frame shares across the top domains.</p>"
            "</div>"
        )

    media_tiles = [tile for tile in [domain_counts_card, domain_distribution_card] if tile]
    top_media_section_html = ""
    if media_tiles:
        top_media_section_html = f"""
        <section class=\"report-section\" id=\"top-media\">
            <header class=\"section-heading\">
                <h2>Top Media &amp; Their Frames</h2>
                <p>Publishing concentration and frame emphasis across leading outlets.</p>
            </header>
            <div class=\"section-body media-body\">
                {''.join(media_tiles)}
            </div>
        </section>
        """

    top_stories_by_frame = _collect_top_stories_by_frame(document_aggregates)
    story_cards: List[str] = []
    for frame in schema.frames:
        stories = top_stories_by_frame.get(frame.frame_id)
        if not stories:
            continue
        color = color_map.get(frame.frame_id, "#1E3D58")
        items: List[str] = []
        for idx, story in enumerate(stories, start=1):
            title_value = str(story.get("title") or f"Story {idx}")
            title = html.escape(title_value)
            url_value = str(story.get("url") or "").strip()
            title_html = (
                f"<a href=\"{html.escape(url_value)}\" target=\"_blank\" rel=\"noopener noreferrer\">{title}</a>"
                if url_value
                else title
            )
            meta_parts: List[str] = []
            domain_label = str(story.get("domain") or "").strip()
            if domain_label:
                meta_parts.append(html.escape(domain_label))
            date_label = _format_date_label(str(story.get("published_at") or ""))
            if date_label:
                meta_parts.append(html.escape(date_label))
            meta_html = f"<div class=\"story-meta\">{' • '.join(meta_parts)}</div>" if meta_parts else ""
            score_pct = f"{float(story.get('score', 0.0)) * 100:.0f}%"
            items.append(
                "<li>"
                f"<div class=\"story-rank\">{idx}</div>"
                "<div class=\"story-content\">"
                f"<div class=\"story-title\">{title_html}</div>"
                f"{meta_html}"
                f"<div class=\"story-score\">{score_pct} frame weight</div>"
                "</div>"
                "</li>"
            )
        short_label = frame.short_name or (frame.name.split()[0] if frame.name else frame.frame_id)
        story_cards.append(
            "<article class=\"story-card\" style=\"--accent-color:" + color + ";\">"
            f"<header><h3>{html.escape(short_label)}</h3><p>{html.escape(frame.name)}</p></header>"
            f"<ol class=\"story-list\">{''.join(items)}</ol>"
            "</article>"
        )

    top_stories_section_html = ""
    if story_cards:
        top_stories_section_html = f"""
        <section class=\"report-section\" id=\"top-stories\">
            <header class=\"section-heading\">
                <h2>Top Stories per Frame</h2>
                <p>Leading documents aligned to each frame based on length-weighted shares.</p>
            </header>
            <div class=\"section-body story-grid\">
                {''.join(story_cards)}
            </div>
        </section>
        """

    applications_block = f"""
    <div class=\"developer-block\">
        <h3>Applications</h3>
        <div class=\"card table-card\">
            <div class=\"table-wrapper\">
                <table>
                    <thead>
                        <tr>
                            <th>Passage Text<div class=\"resizer\"></div></th>
                            <th>LLM Probabilities<div class=\"resizer\"></div></th>
                            <th>Classifier Probabilities<div class=\"resizer\"></div></th>
                            <th>Rationale<div class=\"resizer\"></div></th>
                            <th>Evidence<div class=\"resizer\"></div></th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_html}
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    """

    developer_components: List[str] = []
    if classifier_plots_html:
        developer_components.append(f"<div class=\"developer-block\">{classifier_plots_html}</div>")
    developer_components.append(applications_block)

    developer_section_html = ""
    if developer_components:
        developer_section_html = f"""
        <section class=\"report-section developer\" id=\"developer\">
            <header class=\"section-heading\">
                <h2>Developer</h2>
                <p>Diagnostics and passage-level review for future iteration.</p>
            </header>
            {''.join(developer_components)}
        </section>
        """

    header_metrics = (
        "<div class=\"header-metrics\">"
        f"<div class=\"metric\"><span class=\"metric-value\">{len(schema.frames)}</span><span class=\"metric-label\">Frames</span></div>"
        f"<div class=\"metric\"><span class=\"metric-value\">{classified_documents}</span><span class=\"metric-label\">Documents</span></div>"
        f"<div class=\"metric\"><span class=\"metric-value\">{len(assignments)}</span><span class=\"metric-label\">Passages</span></div>"
        "</div>"
    )
    timeline_html = f"<p class=\"timeline-note\">{html.escape(timeseries_note)}</p>" if timeseries_note else ""
    header_html = f"""
    <header class=\"report-header\">
        <div class=\"heading-text\">
            <span class=\"eyebrow\">Narrative Framing Report</span>
            <h1>{html.escape(schema.domain)}</h1>
            <p>{html.escape(coverage_text)}</p>
            {timeline_html}
        </div>
        {header_metrics}
    </header>
    """

    html_content = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Narrative Framing Report - {html.escape(schema.domain)}</title>
  <script src=\"https://cdn.plot.ly/plotly-2.27.0.min.js\"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=IBM+Plex+Sans:wght@400;600&display=swap');
    :root {{
      --ink-900: #0f172a;
      --ink-800: #1e293b;
      --ink-600: #334155;
      --ink-500: #475569;
      --ink-300: #94a3b8;
      --slate-50: #f7f9fc;
      --slate-100: #eef2f9;
      --border: #d3dce7;
      --accent-1: #1e3d58;
      --accent-2: #057d9f;
      --accent-3: #f18f01;
      --accent-4: #6c63ff;
      --success: #3a7d44;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 48px;
      background: linear-gradient(140deg, var(--slate-50) 0%, #e4edf7 100%);
      font-family: 'Inter', 'Segoe UI', sans-serif;
      color: var(--ink-800);
    }}
    a {{ color: var(--accent-2); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .report-page {{
      max-width: 1320px;
      margin: 0 auto;
      background: #ffffff;
      border-radius: 18px;
      box-shadow: 0 28px 68px rgba(15, 23, 42, 0.12);
      padding: 48px 64px 72px;
    }}
    .report-header {{
      display: flex;
      justify-content: space-between;
      gap: 32px;
      padding: 36px 40px;
      border-radius: 18px;
      background: linear-gradient(135deg, var(--accent-1) 0%, #0e7c7b 55%, var(--accent-4) 100%);
      color: #ffffff;
      margin-bottom: 48px;
    }}
    .heading-text h1 {{
      margin: 8px 0 12px;
      font-size: 2.25rem;
      letter-spacing: -0.015em;
    }}
    .heading-text p {{ margin: 6px 0 0 0; font-size: 1rem; }}
    .eyebrow {{
      text-transform: uppercase;
      letter-spacing: 0.18em;
      font-size: 0.75rem;
      opacity: 0.7;
    }}
    .timeline-note {{
      margin-top: 10px;
      font-size: 0.95rem;
      opacity: 0.85;
    }}
    .header-metrics {{
      display: flex;
      align-items: center;
      gap: 28px;
    }}
    .metric {{ text-align: right; }}
    .metric-value {{
      display: block;
      font-size: 1.9rem;
      font-weight: 600;
    }}
    .metric-label {{
      font-size: 0.85rem;
      opacity: 0.85;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .report-section {{ margin-bottom: 52px; }}
    .report-section:last-of-type {{ margin-bottom: 0; }}
    .section-heading h2 {{
      margin: 0;
      font-size: 1.7rem;
      color: var(--ink-900);
    }}
    .section-heading p {{
      margin: 6px 0 0 0;
      color: var(--ink-500);
      font-size: 1rem;
    }}
    .section-body {{ margin-top: 26px; }}
    .section-note {{
      margin: 0 0 18px 0;
      font-size: 0.95rem;
      color: var(--ink-500);
    }}
    .frames-body {{
      display: flex;
      flex-direction: column;
      gap: 24px;
    }}
    .frame-card-grid {{
      display: grid;
      gap: 18px;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    }}
    .card {{
      background: linear-gradient(140deg, #ffffff 0%, #f8fbff 100%);
      border-radius: 16px;
      border: 1px solid var(--border);
      padding: 22px 24px;
      box-shadow: 0 16px 38px rgba(15, 23, 42, 0.08);
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}
    .card h4 {{
      margin: 0;
      font-size: 1rem;
      color: var(--ink-600);
    }}
    .frame-card {{
      position: relative;
      border-radius: 18px;
      border: 1px solid rgba(15, 23, 42, 0.08);
      padding: 26px 24px;
      background: #ffffff;
      box-shadow: 0 20px 44px rgba(15, 23, 42, 0.1);
      overflow: hidden;
    }}
    .frame-card .share-badge {{
      position: absolute;
      top: 16px;
      right: 18px;
      background: var(--accent-color, var(--accent-2));
      color: #ffffff;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.85rem;
      font-weight: 600;
      letter-spacing: 0.02em;
    }}
    .frame-card::before {{
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(135deg, rgba(30, 61, 88, 0.08), rgba(5, 125, 159, 0.06));
      opacity: 0.8;
      pointer-events: none;
    }}
    .frame-card::after {{
      content: "";
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 6px;
      background: var(--accent-color, var(--accent-1));
      opacity: 0.9;
    }}
    .frame-card h3 {{
      position: relative;
      margin: 0;
      font-size: 1.15rem;
      color: var(--accent-1);
      font-weight: 600;
    }}
    .frame-card-title {{
      position: relative;
      margin: 6px 0 14px;
      font-size: 0.95rem;
      color: var(--ink-600);
    }}
    .frame-card-text {{
      position: relative;
      margin: 0 0 12px 0;
      color: var(--ink-600);
      line-height: 1.5;
      font-size: 0.95rem;
    }}
    .frame-card-meta {{
      position: relative;
      margin: 0 0 8px 0;
      font-size: 0.9rem;
      color: var(--ink-500);
    }}
    .frame-card-meta strong {{ color: var(--ink-600); font-weight: 600; margin-right: 6px; }}
    table {{
      border-collapse: collapse;
      width: 100%;
      table-layout: auto;
      font-size: 0.92rem;
    }}
    thead th {{
      position: sticky;
      top: 0;
      background: var(--slate-100);
      z-index: 2;
      text-align: left;
      font-weight: 600;
      color: var(--ink-600);
      padding: 10px 12px;
    }}
    th, td {{
      border: 1px solid rgba(148, 163, 184, 0.35);
      padding: 10px 12px;
      vertical-align: top;
    }}
    td.passage {{ white-space: pre-wrap; min-width: 280px; width: 30%; max-width: 640px; word-break: break-word; }}
    .summary-table th, .summary-table td {{ border-color: rgba(148, 163, 184, 0.4); }}
    .chart-grid {{
      display: grid;
      gap: 22px;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      margin-bottom: 24px;
    }}
    .chart {{
      margin: 0;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .chart img {{
      width: 100%;
      border-radius: 12px;
      border: 1px solid rgba(148, 163, 184, 0.35);
      background: #ffffff;
    }}
    .chart figcaption {{ font-size: 0.85rem; color: var(--ink-500); }}
    .media-body {{
      display: flex;
      flex-direction: column;
      gap: 24px;
    }}
    .chart-note {{
      margin: 12px 0 0 0;
      font-size: 0.9rem;
      color: var(--ink-500);
    }}
    .plotly-chart {{
      width: 100%;
      min-height: 420px;
      flex: 1;
    }}
    .story-grid {{
      display: grid;
      gap: 20px;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    }}
    .story-card {{
      border-radius: 18px;
      border: 1px solid rgba(15, 23, 42, 0.12);
      padding: 22px 22px 18px;
      background: #ffffff;
      box-shadow: 0 16px 36px rgba(15, 23, 42, 0.08);
      position: relative;
    }}
    .story-card::after {{
      content: "";
      position: absolute;
      inset: 0 0 auto 0;
      height: 4px;
      background: var(--accent-color, var(--accent-2));
      border-radius: 10px 10px 0 0;
    }}
    .story-card header {{ margin-bottom: 14px; }}
    .story-card h3 {{
      margin: 0;
      font-size: 1.05rem;
      color: var(--accent-color, var(--accent-2));
    }}
    .story-card header p {{
      margin: 4px 0 0 0;
      color: var(--ink-500);
      font-size: 0.9rem;
    }}
    .story-list {{
      list-style: none;
      margin: 0;
      padding: 0;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }}
    .story-list li {{
      display: grid;
      grid-template-columns: 32px 1fr;
      gap: 12px;
      align-items: start;
    }}
    .story-rank {{
      font-weight: 600;
      font-size: 1.1rem;
      color: var(--accent-color, var(--accent-2));
      text-align: center;
      padding-top: 2px;
    }}
    .story-title {{ font-weight: 600; color: var(--ink-800); font-size: 0.95rem; }}
    .story-meta {{ font-size: 0.85rem; color: var(--ink-500); margin: 6px 0; }}
    .story-score {{ font-size: 0.85rem; color: var(--accent-color, var(--accent-2)); font-weight: 600; }}
    .developer-block {{ margin-top: 24px; }}
    .developer-block:first-of-type {{ margin-top: 0; }}
    .developer-block h3 {{
      margin: 0 0 18px 0;
      font-size: 1.25rem;
      color: var(--ink-900);
    }}
    .metrics-table table {{ margin-top: 12px; }}
    .table-card {{ padding: 0; overflow: hidden; }}
    .table-wrapper {{ overflow: auto; }}
    .bar {{
      position: relative;
      background: #f0f4f9;
      margin-bottom: 6px;
      height: 24px;
      border-radius: 4px;
      overflow: hidden;
    }}
    .fill {{
      position: absolute;
      left: 0;
      top: 0;
      bottom: 0;
      opacity: 0.9;
    }}
    .bar-label {{
      position: relative;
      z-index: 1;
      padding-left: 6px;
      line-height: 24px;
      font-size: 0.88rem;
      color: var(--ink-800);
      font-weight: 600;
    }}
    .resizer {{
      position: absolute;
      right: 0;
      top: 0;
      width: 6px;
      cursor: col-resize;
      user-select: none;
      height: 100%;
    }}
    .link-icon {{
      margin-right: 6px;
      text-decoration: none;
      font-size: 0.95rem;
    }}
    .link-icon:hover {{ text-decoration: underline; }}
    .passage-text {{ white-space: pre-wrap; }}
    .empty-note {{ font-size: 0.95rem; color: var(--ink-500); }}
    .error-note {{ color: #d32f2f; margin: 0; }}
    @media (max-width: 860px) {{
      body {{ padding: 24px; }}
      .report-page {{ padding: 32px 24px 48px; }}
      .report-header {{
        flex-direction: column;
        align-items: flex-start;
      }}
      .header-metrics {{
        width: 100%;
        justify-content: space-between;
      }}
    }}
  </style>
</head>
<body>
  <div class=\"report-page\">
    {header_html}
    {frames_section_html}
    {llm_coverage_section_html}
    {llm_binned_section_html}
    {time_series_section_html}
    {top_media_section_html}
    {top_stories_section_html}
    {developer_section_html}
  </div>
  <script>
    (function() {{
      const table = document.querySelector('.table-wrapper table');
      if (!table) return;
      const setColumnWidth = (index, width) => {{
        const cells = table.querySelectorAll(`tr > *:nth-child(${{index + 1}})`);
        cells.forEach((cell) => {{ cell.style.width = width + 'px'; }});
      }};
      table.querySelectorAll('thead th').forEach((th, index) => {{
        const resizer = th.querySelector('.resizer');
        if (!resizer) return;
        let startX = 0;
        let startWidth = 0;
        const onMouseMove = (event) => {{
          const delta = event.pageX - startX;
          const newWidth = Math.max(160, startWidth + delta);
          setColumnWidth(index, newWidth);
        }};
        const onMouseUp = () => {{
          document.removeEventListener('mousemove', onMouseMove);
          document.removeEventListener('mouseup', onMouseUp);
        }};
        resizer.addEventListener('mousedown', (event) => {{
          startX = event.pageX;
          startWidth = th.offsetWidth;
          document.addEventListener('mousemove', onMouseMove);
          document.addEventListener('mouseup', onMouseUp);
        }});
      }});
    }})();
  </script>
</body>
</html>
"""

    output_path.write_text(html_content, encoding="utf-8")


__all__ = ["write_html_report"]
