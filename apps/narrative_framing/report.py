"""HTML report generation for the narrative framing workflow."""

from __future__ import annotations

import base64
import html
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from efi_analyser.frames import FrameAssignment, FrameSchema

_PALETTE = [
    "#4F8EF7",
    "#F78E4F",
    "#6CCB5F",
    "#C678DD",
    "#F7C84F",
    "#4FC7F7",
    "#F76F8E",
    "#8E9BF7",
    "#4FF7B6",
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
    
    # Precision bars
    y_pos = np.arange(len(frames))
    bars1 = ax1.barh(y_pos, precisions, color=colors, alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=10)
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
    ax2.set_yticklabels(labels, fontsize=10)
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


def _generate_metrics_table_rows(metrics: Dict[str, Dict[str, float]], frame_names: Dict[str, str]) -> str:
    """Generate HTML table rows for metrics display."""
    rows = []
    for frame_id, frame_metrics in metrics.items():
        frame_name = frame_names.get(frame_id, frame_id)
        rows.append(
            f"<tr>"
            f"<td>{html.escape(frame_name)}</td>"
            f"<td>{frame_metrics['precision']:.3f}</td>"
            f"<td>{frame_metrics['recall']:.3f}</td>"
            f"<td>{frame_metrics['f1']:.3f}</td>"
            f"<td>{frame_metrics['auc']:.3f}</td>"
            f"<td>{int(frame_metrics['support'])}</td>"
            f"</tr>"
        )
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
    for frame_id, score in sorted_items[:5]:
        frame_entry = frame_lookup.get(frame_id, {})
        label = frame_entry.get("short", frame_id)
        width = max(2, int(score * 100))
        color = color_map.get(frame_id, "#4F8EF7")
        prob_bars.append(
            "<div class=\"bar\">"
            f"<div class=\"fill\" style=\"width:{width}%; background:{color};\"></div>"
            f"<span>{html.escape(label)} ({score:.2f})</span>"
            "</div>"
        )
    return "".join(prob_bars) if prob_bars else "—"


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
            
            # Compute metrics
            metrics = _compute_classifier_metrics(assignments, frame_ids, classifier_lookup=classifier_lookup)
            
            # Generate plots
            precision_recall_b64 = _plot_precision_recall_bars(metrics, frame_names, color_map)
            confusion_matrix_b64 = _plot_confusion_matrix(assignments, frame_ids, frame_names, classifier_lookup=classifier_lookup)
            roc_curves_b64 = _plot_roc_curves(assignments, frame_ids, frame_names, color_map, classifier_lookup=classifier_lookup)
            performance_dashboard_b64 = _plot_performance_dashboard(metrics, frame_names, color_map)
            
            # Create HTML for classifier performance section
            classifier_plots_html = f"""
            <div class="classifier-performance">
                <h3>Classifier Performance Analysis</h3>
                
                <div class="performance-grid">
                    <div class="plot-container">
                        <h4>Precision & Recall by Frame</h4>
                        <img src="data:image/png;base64,{precision_recall_b64}" alt="Precision and Recall by Frame" style="width: 100%; max-width: 600px; border: 1px solid #e0e0e0; border-radius: 6px;" />
                    </div>
                    
                    <div class="plot-container">
                        <h4>Performance Dashboard</h4>
                        <img src="data:image/png;base64,{performance_dashboard_b64}" alt="Performance Dashboard" style="width: 100%; max-width: 600px; border: 1px solid #e0e0e0; border-radius: 6px;" />
                    </div>
                    
                    <div class="plot-container">
                        <h4>Confusion Matrix</h4>
                        <img src="data:image/png;base64,{confusion_matrix_b64}" alt="Confusion Matrix" style="width: 100%; max-width: 400px; border: 1px solid #e0e0e0; border-radius: 6px;" />
                    </div>
                    
                    <div class="plot-container">
                        <h4>ROC Curves by Frame</h4>
                        <img src="data:image/png;base64,{roc_curves_b64}" alt="ROC Curves by Frame" style="width: 100%; max-width: 500px; border: 1px solid #e0e0e0; border-radius: 6px;" />
                    </div>
                </div>
                
                <div class="metrics-table">
                    <h4>Detailed Metrics by Frame</h4>
                    <table class="summary-table">
                        <thead>
                            <tr>
                                <th>Frame</th>
                                <th>Precision</th>
                                <th>Recall</th>
                                <th>F1 Score</th>
                                <th>AUC</th>
                                <th>Support</th>
                            </tr>
                        </thead>
                        <tbody>
                            {_generate_metrics_table_rows(metrics, frame_names)}
                        </tbody>
                    </table>
                </div>
            </div>
            """
        except Exception as e:
            classifier_plots_html = f"""
            <div class="classifier-performance">
                <h3>Classifier Performance Analysis</h3>
                <p style="color: #d32f2f;">Error generating classifier performance plots: {html.escape(str(e))}</p>
            </div>
            """

    legend_items = []
    for frame in schema.frames:
        color = color_map.get(frame.frame_id, "#4F8EF7")
        short_label = frame.short_name or (frame.name.split()[0] if frame.name else frame.frame_id)
        legend_items.append(
            "<li>"
            f"<span class=\"legend-swatch\" style=\"background:{color};\"></span>"
            f"<strong>{html.escape(short_label)}</strong> — {html.escape(frame.name)}"
            "</li>"
        )

    frame_share_rows = []
    if global_frame_share:
        ordered_share = sorted(global_frame_share.items(), key=lambda item: item[1], reverse=True)
        for frame_id, value in ordered_share:
            label = frame_lookup.get(frame_id, {}).get("name", frame_id)
            frame_share_rows.append(
                "<tr>"
                f"<td>{html.escape(label)}</td>"
                f"<td>{frame_id}</td>"
                f"<td>{value:.2%}</td>"
                "</tr>"
            )

    coverage_text = """No documents were classified."""
    if classified_documents > 0:
        coverage_text = f"Classifier applied to {classified_documents} documents."
        if classifier_sample_limit:
            coverage_text += f" Target sample: {classifier_sample_limit}."

    domain_counts_chart_html = ""
    domain_distribution_chart_html = ""
    domain_counts_table_html = ""

    if domain_counts:
        domain_counts_chart_b64 = _plot_domain_counts_bar(domain_counts[:20])
        if domain_counts_chart_b64:
            domain_counts_chart_html = (
                "<figure class=\"chart\">"
                f"<img src=\"data:image/png;base64,{domain_counts_chart_b64}\" alt=\"Top domains by document count\" />"
                "<figcaption>Top domains ranked by number of classified documents.</figcaption>"
                "</figure>"
            )
        rows = []
        for domain, count in domain_counts[:20]:
            rows.append(
                "<tr>"
                f"<td>{html.escape(domain)}</td>"
                f"<td>{count}</td>"
                "</tr>"
            )
        if rows:
            domain_counts_table_html = (
                "<div class=\"domain-table\">"
                "<table class=\"summary-table\">"
                "<thead><tr><th>Domain</th><th>Documents</th></tr></thead>"
                f"<tbody>{''.join(rows)}</tbody>"
                "</table>"
                "</div>"
            )

    if domain_frame_summaries:
        ordered_domain_frames = [entry for entry in domain_frame_summaries if entry.get("shares")]
        if ordered_domain_frames:
            domain_frame_chart_b64 = _plot_domain_frame_facets(ordered_domain_frames, frame_lookup, color_map)
            if domain_frame_chart_b64:
                domain_distribution_chart_html = (
                    "<figure class=\"chart\">"
                    f"<img src=\"data:image/png;base64,{domain_frame_chart_b64}\" alt=\"Frame distribution by domain\" />"
                    "<figcaption>Frame score distribution for the top domains. Each subplot shows average frame scores across documents from that domain.</figcaption>"
                    "</figure>"
                )

    chart_html = ""
    if area_chart_b64:
        chart_html = (
            "<figure class=\"chart\">"
            f"<img src=\"data:image/png;base64,{area_chart_b64}\" alt=\"Frame importance over time\" />"
            "<figcaption>Frame share over time (stacked area chart).</figcaption>"
            "</figure>"
        )

    timeseries_note = ""
    if timeseries_records:
        date_values = [str(item.get("date", "")) for item in timeseries_records if item.get("date")]
        if date_values:
            start = min(date_values)
            end = max(date_values)
            timeseries_note = f"Data covers {html.escape(start)} to {html.escape(end)}."

    document_summary_sections: List[str] = []
    document_summary_sections.append(f"<p>{coverage_text}</p>")
    if timeseries_note:
        document_summary_sections.append(f"<p>{timeseries_note}</p>")
    if frame_share_rows:
        document_summary_sections.append(
            "<div class=\"frame-share\">"
            "<h3>Length-weighted frame distribution</h3>"
            "<table class=\"summary-table\">"
            "<thead><tr><th>Frame</th><th>ID</th><th>Share</th></tr></thead>"
            f"<tbody>{''.join(frame_share_rows)}</tbody>"
            "</table>"
            "</div>"
        )
    if domain_counts_chart_html or domain_counts_table_html or domain_distribution_chart_html:
        domain_section_parts = [
            "<div class=\"domain-stats\">",
            "<h3>Source domain analysis</h3>",
        ]
        if domain_counts_chart_html:
            domain_section_parts.append(domain_counts_chart_html)
        if domain_counts_table_html:
            domain_section_parts.append(domain_counts_table_html)
        if domain_distribution_chart_html:
            domain_section_parts.append(domain_distribution_chart_html)
        domain_section_parts.append("</div>")
        document_summary_sections.append("".join(domain_section_parts))
    if chart_html:
        document_summary_sections.append(chart_html)
    
    # Add classifier performance plots
    if classifier_plots_html:
        document_summary_sections.append(classifier_plots_html)

    summary_html = "".join(document_summary_sections) if document_summary_sections else "<p>No classifier summary available.</p>"

    details_sections = []
    for frame in schema.frames:
        section_parts = [
            "<section class=\"frame-detail\">",
            f"<h3>{html.escape(frame.short_name or frame.frame_id)} — {html.escape(frame.name)}</h3>",
            f"<p>{html.escape(frame.description)}</p>",
        ]
        if frame.keywords:
            section_parts.append(
                f"<p><strong>Keywords:</strong> {html.escape(', '.join(frame.keywords))}</p>"
            )
        if frame.examples:
            section_parts.append(
                f"<p><strong>Examples:</strong> {html.escape('; '.join(frame.examples[:2]))}</p>"
            )
        section_parts.append("</section>")
        details_sections.append("".join(section_parts))

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

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Frame Assignments Report</title>
  <style>
    html, body {{ height: 100%; }}
    body {{ font-family: Arial, sans-serif; margin: 0; padding: 24px; box-sizing: border-box; overflow: auto; }}
    h1 {{ margin-bottom: 12px; }}
    h2 {{ margin: 0 0 8px 0; font-size: 1.1rem; }}
    .summary {{ display: flex; flex-direction: column; gap: 24px; margin-bottom: 24px; }}
    .summary p {{ margin: 0; font-size: 0.95rem; }}
    .legend {{ list-style: none; padding: 0; margin: 0; display: flex; flex-wrap: wrap; gap: 12px; }}
    .legend li {{ display: flex; align-items: center; gap: 8px; font-size: 0.95rem; }}
    .legend-swatch {{ width: 14px; height: 14px; border-radius: 2px; display: inline-block; }}
    .details-grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }}
    .frame-detail {{ border: 1px solid #e0e0e0; border-radius: 6px; padding: 12px; background: #fafafa; }}
    .frame-detail h3 {{ margin: 0 0 6px 0; font-size: 1rem; }}
    .summary-table {{ border-collapse: collapse; width: 100%; margin-top: 8px; font-size: 0.95rem; }}
    .summary-table th, .summary-table td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
    .summary-table thead th {{ background: #f5f5f5; position: static; }}
    .chart {{ display: flex; flex-direction: column; gap: 6px; }}
    .chart img {{ width: 100%; max-width: 720px; border: 1px solid #e0e0e0; border-radius: 6px; background: #fff; }}
    .chart figcaption {{ font-size: 0.85rem; color: #555; }}
    table {{ border-collapse: collapse; width: max(100%, 1200px); table-layout: auto; }}
    thead th {{ position: sticky; top: 0; background: #f5f5f5; z-index: 2; min-width: 160px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    td.passage {{ white-space: pre-wrap; min-width: 280px; }}
    .table-wrapper {{ overflow: visible; border: 1px solid #ccc; }}
    .bar {{ position: relative; background: #f0f0f0; margin-bottom: 6px; height: 24px; border-radius: 4px; overflow: hidden; }}
    .fill {{ position: absolute; left: 0; top: 0; bottom: 0; opacity: 0.85; }}
    .bar span {{ position: relative; z-index: 1; padding-left: 6px; line-height: 24px; font-size: 0.9rem; color: #1a1a1a; }}
    .resizer {{ position: absolute; right: 0; top: 0; width: 6px; cursor: col-resize; user-select: none; height: 100%; }}
    .link-icon {{ margin-right: 6px; text-decoration: none; font-size: 0.95rem; }}
    .link-icon:hover {{ text-decoration: underline; }}
    .passage-text {{ white-space: pre-wrap; }}
    .classifier-performance {{ margin-top: 24px; }}
    .performance-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 16px 0; }}
    .plot-container {{ text-align: center; }}
    .plot-container h4 {{ margin: 0 0 12px 0; font-size: 1rem; color: #333; }}
    .metrics-table {{ margin-top: 20px; }}
    .metrics-table h4 {{ margin: 0 0 12px 0; font-size: 1rem; color: #333; }}
    .domain-stats {{ display: flex; flex-direction: column; gap: 16px; margin-top: 20px; }}
    .domain-table table {{ width: 100%; max-width: 520px; }}
    .domain-table th, .domain-table td {{ text-align: left; }}
  </style>
</head>
<body>
  <h1>Frame Report: {html.escape(schema.domain)}</h1>
  <div class="summary">
    <section>
      <h2>Classifier Summary</h2>
      {summary_html}
    </section>
    <section>
      <h2>Frame Legend</h2>
      <ul class="legend">{''.join(legend_items)}</ul>
    </section>
    <section>
      <h2>Frame Details</h2>
      <div class="details-grid">{''.join(details_sections)}</div>
    </section>
  </div>
  <div class="table-wrapper">
    <table>
      <thead>
        <tr>
          <th>Passage Text<div class="resizer"></div></th>
          <th>LLM Probabilities<div class="resizer"></div></th>
          <th>Classifier Probabilities<div class="resizer"></div></th>
          <th>Rationale<div class="resizer"></div></th>
          <th>Evidence<div class="resizer"></div></th>
        </tr>
      </thead>
      <tbody>
        {table_html}
      </tbody>
    </table>
  </div>
  <script>
    (function() {{
      const table = document.querySelector('table');
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
          const newWidth = Math.max(140, startWidth + delta);
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
