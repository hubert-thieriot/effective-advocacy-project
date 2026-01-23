"""Plotting functions for discourse analysis frame aggregates.

Works with DataFrame-based FrameAggregates for simple, clean visualizations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

# Plotly imports
import plotly.graph_objects as go
import plotly.io as pio

try:
    import kaleido  # noqa: F401 - needed for static image export
except ImportError:
    pass

# Suppress verbose Kaleido/Chromium logging
logging.getLogger("kaleido").setLevel(logging.WARNING)
logging.getLogger("choreographer").setLevel(logging.WARNING)


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


def build_color_map(frame_ids: Sequence[str]) -> Dict[str, str]:
    """Build a color map for frame IDs."""
    return {fid: _PALETTE[i % len(_PALETTE)] for i, fid in enumerate(frame_ids)}


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert hex color to rgba string."""
    value = hex_color.lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    try:
        r, g, b = int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)
    except ValueError:
        r = g = b = 0
    return f"rgba({r}, {g}, {b}, {max(0.0, min(alpha, 1.0))})"


def wrap_label_html(text: str, max_len: int = 16) -> str:
    """Insert <br> breaks into text for multi-line labels."""
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
            current.append(w)
            current_len += (1 + wlen) if current_len > 0 else wlen
    if current:
        lines.append(" ".join(current))
    return "<br>".join(lines)


def _export_figure(
    fig: go.Figure,
    export_path: Optional[Path],
    div_id: str,
) -> None:
    """Export figure to PNG and HTML if path provided."""
    if export_path is None:
        return
    try:
        export_path.parent.mkdir(parents=True, exist_ok=True)
        # PNG export
        img_bytes = pio.to_image(fig, format="png", scale=3)
        export_path.write_bytes(img_bytes)
        # HTML export
        html_path = export_path.with_suffix(".html")
        html_content = fig.to_html(
            include_plotlyjs="cdn",
            div_id=div_id,
            config={"displayModeBar": False, "responsive": True},
        )
        html_path.write_text(html_content, encoding="utf-8")
    except Exception as exc:
        print(f"Export failed for {div_id}: {exc}")


def _fig_to_html_fragment(fig: go.Figure, div_id: str) -> str:
    """Convert Plotly figure to embeddable HTML fragment."""
    html = fig.to_html(
        include_plotlyjs=False,
        div_id=div_id,
        config={"displayModeBar": False, "responsive": True, "staticPlot": True},
    )
    # Extract just the div and script parts
    if '<div id="' in html and "</script>" in html:
        div_start = html.find('<div id="')
        div_end = html.find("</div>", div_start) + 6
        script_start = html.find('<script type="text/javascript">', div_end)
        script_end = html.find("</script>", script_start) + 9
        return html[div_start:div_end] + html[script_start:script_end]
    return f'<div id="{div_id}" class="plotly-chart"></div>'


def plot_global_distribution(
    global_totals: pd.DataFrame,
    frame_labels: Optional[Dict[str, str]] = None,
    color_map: Optional[Dict[str, str]] = None,
    *,
    export_path: Optional[Path] = None,
) -> str:
    """Plot global frame distribution as a bar chart.

    Args:
        global_totals: DataFrame with columns [frame_id, weight, doc_count].
        frame_labels: Optional mapping of frame_id to display label.
        color_map: Optional mapping of frame_id to color.
        export_path: Optional path to export PNG/HTML.

    Returns:
        HTML fragment for embedding.
    """
    if global_totals is None or global_totals.empty:
        return ""

    df = global_totals.sort_values("weight", ascending=False).copy()
    frame_ids = df["frame_id"].tolist()
    weights = df["weight"].tolist()

    if color_map is None:
        color_map = build_color_map(frame_ids)
    if frame_labels is None:
        frame_labels = {}

    labels = [wrap_label_html(frame_labels.get(fid, fid), 16) for fid in frame_ids]
    colors = [color_map.get(fid, "#057d9f") for fid in frame_ids]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=weights,
                marker_color=colors,
                hovertemplate="%{x}<br>%{y:.1%}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        margin={"l": 40, "r": 20, "t": 20, "b": 0},
        xaxis={"title": "", "tickmode": "linear", "tickangle": 0, "automargin": True},
        yaxis={"title": "", "tickformat": ".0%"},
        height=450,
    )

    _export_figure(fig, export_path, "global-distribution")
    return _fig_to_html_fragment(fig, "global-distribution")


def plot_by_year(
    by_year: pd.DataFrame,
    frame_labels: Optional[Dict[str, str]] = None,
    color_map: Optional[Dict[str, str]] = None,
    *,
    export_path: Optional[Path] = None,
) -> str:
    """Plot frame distribution by year as grouped bars.

    Args:
        by_year: DataFrame with columns [year, frame_id, weight, doc_count].
        frame_labels: Optional mapping of frame_id to display label.
        color_map: Optional mapping of frame_id to color.
        export_path: Optional path to export PNG/HTML.

    Returns:
        HTML fragment for embedding.
    """
    if by_year is None or by_year.empty:
        return ""

    df = by_year.copy()
    years = sorted(df["year"].dropna().unique())
    frame_ids = df.groupby("frame_id")["weight"].mean().sort_values(ascending=False).index.tolist()

    if color_map is None:
        color_map = build_color_map(frame_ids)
    if frame_labels is None:
        frame_labels = {}

    labels = [wrap_label_html(frame_labels.get(fid, fid), 16) for fid in frame_ids]
    num_years = len(years)

    traces = []
    for idx, year in enumerate(years):
        year_df = df[df["year"] == year].set_index("frame_id")
        weights = [year_df.loc[fid, "weight"] if fid in year_df.index else 0.0 for fid in frame_ids]
        # Alpha: older years more transparent
        alpha = 0.5 + (idx / max(num_years - 1, 1)) * 0.5
        colors = [hex_to_rgba(color_map.get(fid, "#057d9f"), alpha) for fid in frame_ids]
        traces.append(
            go.Bar(
                name=str(int(year)),
                x=labels,
                y=weights,
                marker_color=colors,
                hovertemplate=f"%{{x}}<br>Year: {int(year)}<br>%{{y:.1%}}<extra></extra>",
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        barmode="group",
        margin={"l": 40, "r": 20, "t": 20, "b": 0},
        xaxis={"title": "", "tickmode": "linear", "tickangle": 0, "automargin": True},
        yaxis={"title": "", "tickformat": ".0%"},
        height=450,
        legend={"orientation": "h", "yanchor": "top", "y": 1.08, "x": 0.5, "xanchor": "center"},
    )

    _export_figure(fig, export_path, "by-year")
    return _fig_to_html_fragment(fig, "by-year")


def plot_by_domain(
    by_domain: pd.DataFrame,
    frame_labels: Optional[Dict[str, str]] = None,
    color_map: Optional[Dict[str, str]] = None,
    *,
    top_n: int = 15,
    export_path: Optional[Path] = None,
) -> str:
    """Plot frame distribution by domain as a faceted chart.

    Args:
        by_domain: DataFrame with columns [domain, frame_id, weight, doc_count].
        frame_labels: Optional mapping of frame_id to display label.
        color_map: Optional mapping of frame_id to color.
        top_n: Number of top domains to show.
        export_path: Optional path to export PNG/HTML.

    Returns:
        HTML fragment for embedding.
    """
    if by_domain is None or by_domain.empty:
        return ""

    df = by_domain.copy()

    # Get top domains by doc_count
    domain_counts = df.groupby("domain")["doc_count"].first().sort_values(ascending=False)
    top_domains = domain_counts.head(top_n).index.tolist()
    df = df[df["domain"].isin(top_domains)]

    frame_ids = df.groupby("frame_id")["weight"].mean().sort_values(ascending=False).index.tolist()

    if color_map is None:
        color_map = build_color_map(frame_ids)
    if frame_labels is None:
        frame_labels = {}

    labels = [wrap_label_html(frame_labels.get(fid, fid), 12) for fid in frame_ids]

    # Create subplots
    from plotly.subplots import make_subplots

    n_domains = len(top_domains)
    cols = min(4, max(2, int(n_domains**0.5 * 1.2)))
    rows = (n_domains + cols - 1) // cols

    subplot_titles = [
        f"<b>{domain}</b> (n={int(domain_counts[domain])})" for domain in top_domains
    ]

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
        shared_xaxes=True,
        shared_yaxes=True,
    )

    colors = [color_map.get(fid, "#057d9f") for fid in frame_ids]

    for idx, domain in enumerate(top_domains):
        row = (idx // cols) + 1
        col = (idx % cols) + 1
        domain_df = df[df["domain"] == domain].set_index("frame_id")
        weights = [domain_df.loc[fid, "weight"] if fid in domain_df.index else 0.0 for fid in frame_ids]
        fig.add_trace(
            go.Bar(x=labels, y=weights, marker_color=colors, showlegend=False),
            row=row,
            col=col,
        )

    fig.update_layout(
        height=max(400, rows * 150),
        margin={"t": 40, "b": 20, "l": 40, "r": 20},
        showlegend=False,
    )
    fig.update_yaxes(tickformat=".0%")
    fig.update_xaxes(showticklabels=False)
    fig.update_annotations(font_size=9)

    _export_figure(fig, export_path, "by-domain")
    return _fig_to_html_fragment(fig, "by-domain")


def plot_by_corpus(
    by_corpus: pd.DataFrame,
    frame_labels: Optional[Dict[str, str]] = None,
    color_map: Optional[Dict[str, str]] = None,
    corpus_labels: Optional[Dict[str, str]] = None,
    *,
    export_path: Optional[Path] = None,
) -> str:
    """Plot frame distribution by corpus as grouped bars.

    Args:
        by_corpus: DataFrame with columns [corpus, frame_id, weight, doc_count].
        frame_labels: Optional mapping of frame_id to display label.
        color_map: Optional mapping of frame_id to color.
        corpus_labels: Optional mapping of corpus name to display label.
        export_path: Optional path to export PNG/HTML.

    Returns:
        HTML fragment for embedding.
    """
    if by_corpus is None or by_corpus.empty:
        return ""

    df = by_corpus.copy()
    corpora = df["corpus"].unique().tolist()
    frame_ids = df.groupby("frame_id")["weight"].mean().sort_values(ascending=False).index.tolist()

    if color_map is None:
        color_map = build_color_map(frame_ids)
    if frame_labels is None:
        frame_labels = {}
    if corpus_labels is None:
        corpus_labels = {}

    labels = [wrap_label_html(frame_labels.get(fid, fid), 16) for fid in frame_ids]
    corpus_colors = {c: _PALETTE[i % len(_PALETTE)] for i, c in enumerate(corpora)}

    traces = []
    for corpus in corpora:
        corpus_df = df[df["corpus"] == corpus].set_index("frame_id")
        weights = [corpus_df.loc[fid, "weight"] if fid in corpus_df.index else 0.0 for fid in frame_ids]
        display_name = corpus_labels.get(corpus, corpus)
        traces.append(
            go.Bar(
                name=display_name,
                x=labels,
                y=weights,
                marker_color=corpus_colors[corpus],
                hovertemplate=f"%{{x}}<br>{display_name}<br>%{{y:.1%}}<extra></extra>",
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        barmode="group",
        margin={"l": 40, "r": 20, "t": 20, "b": 0},
        xaxis={"title": "", "tickmode": "linear", "tickangle": 0, "automargin": True},
        yaxis={"title": "", "tickformat": ".0%"},
        height=450,
        legend={"orientation": "h", "yanchor": "top", "y": 1.08, "x": 0.5, "xanchor": "center"},
    )

    _export_figure(fig, export_path, "by-corpus")
    return _fig_to_html_fragment(fig, "by-corpus")


def plot_document_count_by_domain(
    by_domain: pd.DataFrame,
    *,
    top_n: int = 20,
    export_path: Optional[Path] = None,
) -> str:
    """Plot document counts by domain as horizontal bar chart.

    Args:
        by_domain: DataFrame with columns [domain, frame_id, weight, doc_count].
        top_n: Number of top domains to show.
        export_path: Optional path to export PNG/HTML.

    Returns:
        HTML fragment for embedding.
    """
    if by_domain is None or by_domain.empty:
        return ""

    # Get unique domain counts
    domain_counts = by_domain.groupby("domain")["doc_count"].first().sort_values(ascending=False)
    top_domains = domain_counts.head(top_n)

    fig = go.Figure(
        data=[
            go.Bar(
                y=top_domains.index.tolist(),
                x=top_domains.values.tolist(),
                orientation="h",
                marker_color="#057d9f",
                hovertemplate="%{y}<br>%{x} documents<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        margin={"l": 120, "r": 30, "t": 20, "b": 40},
        xaxis={"title": "Documents"},
        yaxis={"title": "", "autorange": "reversed"},
        height=max(300, 25 * len(top_domains)),
    )

    _export_figure(fig, export_path, "domain-counts")
    return _fig_to_html_fragment(fig, "domain-counts")


__all__ = [
    "build_color_map",
    "plot_global_distribution",
    "plot_by_year",
    "plot_by_domain",
    "plot_by_corpus",
    "plot_document_count_by_domain",
]
