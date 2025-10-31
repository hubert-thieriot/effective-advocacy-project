"""
HTML report generation for the word count (theme keyword) application.

Implements interactive Plotly charts with modern styling matching the narrative framing report.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from apps.narrative_framing.aggregation import DocumentFrameAggregate

# Color palette matching narrative framing report
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


@dataclass
class ThemeDocStat:
    document_id: str
    url: str
    title: str
    date: str
    domain: str
    hits: int


def _render_plotly_fragment(
    div_id: str,
    data: Sequence[Dict[str, object]],
    layout: Dict[str, object],
    *,
    config: Optional[Dict[str, object]] = None,
) -> str:
    """Render a Plotly chart fragment."""
    if not data:
        return ""
    # Disable interactivity by default to prevent scroll trapping
    config = config or {"displayModeBar": False, "responsive": True, "staticPlot": True, "scrollZoom": False, "doubleClick": False}
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


def _render_total_docs_timeseries(daily_total_docs: Dict[str, int]) -> str:
    """Render total documents per day with 7-day rolling average."""
    if not daily_total_docs:
        return ""
    
    # Build DataFrame
    rows = []
    for date, count in daily_total_docs.items():
        rows.append({"date": date, "count": int(count)})
    if not rows:
        return ""
    
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if df.empty:
        return ""
    
    # Sort by date and fill missing days with 0, then apply 7-day rolling average
    df = df.sort_values("date").set_index("date")
    df = df.asfreq("D", fill_value=0)
    df["smooth"] = df["count"].rolling(window=7, min_periods=1).mean()
    
    data = [{
        "type": "scatter",
        "mode": "lines",
        "name": "Articles per day",
        "x": df.index.strftime("%Y-%m-%d").tolist(),
        "y": df["smooth"].tolist(),
        "line": {"color": "#1E3D58", "width": 2.5},
        "hovertemplate": "Date: %{x}<br>Articles (7-day avg): %{y}<extra></extra>"
    }]
    
    layout = {
        "title": {"text": "Total Articles Per Day", "font": {"size": 18, "color": "#1e293b"}},
        "xaxis": {"title": "Date"},
        "yaxis": {"title": "Articles per day (7-day avg)"},
        "margin": {"l": 60, "r": 20, "t": 60, "b": 60},
        "plot_bgcolor": "rgba(0,0,0,0)",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter, sans-serif", "size": 12},
    }
    
    return _render_plotly_fragment("total-docs-timeseries-chart", data, layout)


def _render_theme_bar(theme_counts: Dict[str, int], theme_names: Dict[str, str]) -> str:
    """Render theme coverage bar chart."""
    if not theme_counts:
        return ""
    
    # Sort themes by count descending
    sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
    theme_ids = [t[0] for t in sorted_themes]
    counts = [t[1] for t in sorted_themes]
    labels = [theme_names.get(tid, tid) for tid in theme_ids]
    
    # Assign colors from palette
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(theme_ids))]
    
    data = [{
        "type": "bar",
        "x": labels,
        "y": counts,
        "marker": {"color": colors, "opacity": 0.85},
        "hovertemplate": "<b>%{x}</b><br>Documents: %{y}<extra></extra>"
    }]
    
    layout = {
        "title": {"text": "Theme Coverage", "font": {"size": 18, "color": "#1e293b"}},
        "xaxis": {"title": "Theme", "tickangle": -20},
        "yaxis": {"title": "Documents mentioning theme"},
        "margin": {"l": 60, "r": 20, "t": 60, "b": 80},
        "plot_bgcolor": "rgba(0,0,0,0)",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter, sans-serif", "size": 12}
    }
    
    return _render_plotly_fragment("theme-bar-chart", data, layout)


def _render_timeseries_lines(daily_counts: Dict[str, Dict[str, int]], theme_names: Dict[str, str]) -> str:
    """Render daily trend line chart."""
    if not daily_counts:
        return ""
    
    # Build DataFrame
    rows = []
    for tid, series in daily_counts.items():
        for d, v in series.items():
            rows.append({"theme_id": tid, "date": d, "value": int(v)})
    if not rows:
        return ""
    
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    if df.empty:
        return ""
    
    # Apply 7-day rolling average
    frames = []
    for tid, group in df.groupby("theme_id"):
        g = group.sort_values("date").set_index("date")["value"].asfreq("D", fill_value=0)
        g = g.rolling(window=7, min_periods=1).mean()
        frames.append(pd.DataFrame({"date": g.index, "value": g.values, "theme_id": tid}))
    plot_df = pd.concat(frames, ignore_index=True)

    # Create traces for each theme
    data = []
    theme_ids = list(daily_counts.keys())
    for i, tid in enumerate(theme_ids):
        theme_data = plot_df[plot_df["theme_id"] == tid]
        if not theme_data.empty:
            data.append({
                "type": "scatter",
                "mode": "lines",
                "name": theme_names.get(tid, tid),
                "x": theme_data["date"].dt.strftime("%Y-%m-%d").tolist(),
                "y": theme_data["value"].tolist(),
                "line": {"color": _PALETTE[i % len(_PALETTE)], "width": 2.5},
                "hovertemplate": f"<b>{theme_names.get(tid, tid)}</b><br>Date: %{{x}}<br>Docs (7-day avg): %{{y}}<extra></extra>"
            })
    
    layout = {
        "title": {"text": "Daily Trend by Theme", "font": {"size": 18, "color": "#1e293b"}},
        "xaxis": {"title": "Date"},
        "yaxis": {"title": "Docs per day (7-day avg)"},
        "margin": {"l": 60, "r": 20, "t": 60, "b": 60},
        "plot_bgcolor": "rgba(0,0,0,0)",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter, sans-serif", "size": 12},
        "legend": {"x": 1.02, "y": 1, "bgcolor": "rgba(255,255,255,0.8)"}
    }
    
    return _render_plotly_fragment("timeseries-chart", data, layout)


def _render_occurrence_percentage_bars(
    documents: Sequence[DocumentFrameAggregate],
    theme_ids: Sequence[str],
    theme_names: Dict[str, str],
    *,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
) -> str:
    if not documents or not theme_ids:
        return ""
    total_docs = len(documents)
    counts: Dict[str, int] = {tid: 0 for tid in theme_ids}
    for doc in documents:
        for tid in theme_ids:
            try:
                val = float(doc.frame_scores.get(tid, 0.0))
            except Exception:
                val = 0.0
            if val > 0.0:
                counts[tid] += 1
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    tids = [fid for fid, _ in ordered]
    shares = [(counts[fid] / total_docs) if total_docs > 0 else 0.0 for fid in tids]
    labels = [theme_names.get(fid, fid) for fid in tids]
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(tids))]
    data = [{
        "type": "bar",
        "x": labels,
        "y": shares,
        "marker": {"color": colors},
        "hovertemplate": "%{x}<br>%{y:.1%}<extra></extra>",
    }]
    top_margin = 100 if title else 20
    layout = {
        "margin": {"l": 40, "r": 20, "t": top_margin, "b": 70},
        "xaxis": {"title": "", "tickmode": "linear", "tickangle": 0, "automargin": True},
        "yaxis": {"title": "Share of documents", "tickformat": ".0%"},
        "height": 420,
        "legend": {"orientation": "h", "yanchor": "top", "y": 1.02, "x": 0, "xanchor": "left"},
    }
    if title:
        title_text = f"<b>{title}</b>"
        if subtitle:
            title_text += f"<br><span style='font-size:14px;color:#333'>{subtitle}</span>"
        layout["title"] = {
            "text": title_text,
            "x": 0,
            "xanchor": "left",
            "yanchor": "top",
            "pad": {"b": 4},
            "font": {"size": 18, "color": "#333"},
        }
    return _render_plotly_fragment("occurrence-percentage-chart", data, layout)


def _render_occurrence_by_year(
    documents: Sequence[DocumentFrameAggregate],
    theme_ids: Sequence[str],
    theme_names: Dict[str, str],
) -> str:
    if not documents or not theme_ids:
        return ""
    from collections import defaultdict
    from datetime import date
    from dateutil import parser

    theme_year_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
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
        for tid in theme_ids:
            try:
                val = float(doc.frame_scores.get(tid, 0.0))
            except Exception:
                val = 0.0
            if val > 0.0:
                theme_year_counts[tid][year] += 1
    if not years_set:
        return ""
    years = sorted(years_set)
    totals_by_theme: Dict[str, int] = {tid: sum(theme_year_counts[tid].values()) for tid in theme_ids}
    theme_ids_sorted = sorted(theme_ids, key=lambda tid: totals_by_theme.get(tid, 0), reverse=True)
    labels = [theme_names.get(tid, tid) for tid in theme_ids_sorted]

    traces: List[Dict[str, object]] = []
    num_years = len(years)
    for idx, year in enumerate(years):
        alpha = 0.3 + (idx / max(num_years - 1, 1)) * 0.7
        year_total = year_totals[year]
        if year_total > 0:
            shares = [(theme_year_counts[tid][year] / year_total) for tid in theme_ids_sorted]
        else:
            shares = [0.0] * len(theme_ids_sorted)
        colors_with_alpha = [
            f"rgba({int(int(c[1:3],16))},{int(int(c[3:5],16))},{int(int(c[5:7],16))},{alpha})"
            for c in [_PALETTE[i % len(_PALETTE)] for i in range(len(theme_ids_sorted))]
        ]
        traces.append({
            "type": "bar",
            "name": str(year),
            "x": labels,
            "y": shares,
            "marker": {"color": colors_with_alpha},
            "hovertemplate": "%{x}<br>Year: " + str(year) + "<br>%{y:.1%}<extra></extra>",
            "showlegend": True,
        })
    layout = {
        "barmode": "group",
        "margin": {"l": 40, "r": 20, "t": 20, "b": 60},
        "xaxis": {"title": "", "tickmode": "linear", "tickangle": 0, "automargin": True},
        "yaxis": {"title": "Share of documents", "tickformat": ".0%"},
        "height": 500,
        "legend": {"orientation": "h", "yanchor": "top", "y": 1.06, "x": 0.5, "xanchor": "center"},
    }
    return _render_plotly_fragment("occurrence-by-year-chart", traces, layout)


def _render_timeseries_share_lines(
    documents: Sequence[DocumentFrameAggregate],
    theme_ids: Sequence[str],
    theme_names: Dict[str, str],
) -> str:
    if not documents or not theme_ids:
        return ""
    from collections import defaultdict
    # Build daily counts per theme and total docs per day
    daily_theme_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    daily_totals: Dict[str, int] = defaultdict(int)
    for doc in documents:
        day = str(doc.published_at or "").strip()[:10]
        if not day:
            continue
        daily_totals[day] += 1
        for tid in theme_ids:
            try:
                v = float(doc.frame_scores.get(tid, 0.0))
            except Exception:
                v = 0.0
            if v > 0.0:
                daily_theme_counts[tid][day] += 1

    # Construct long-form records with share per day per theme
    rows: List[Dict[str, object]] = []
    for tid in theme_ids:
        for d, cnt in daily_theme_counts[tid].items():
            denom = daily_totals.get(d, 0)
            share = (cnt / denom) if denom > 0 else 0.0
            rows.append({"theme_id": tid, "date": d, "share": share})
    if not rows:
        return ""
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    if df.empty:
        return ""
    traces: List[Dict[str, object]] = []
    for i, tid in enumerate(theme_ids):
        series = df[df["theme_id"] == tid][["date", "share"]].copy()
        if series.empty:
            continue
        series = series.set_index("date").asfreq("D").fillna(0.0)
        series["smooth"] = series["share"].rolling(window=30, min_periods=1).mean()
        x_vals = series.index.strftime("%Y-%m-%d").tolist()
        y_vals = series["smooth"].clip(0, 1).round(5).tolist()
        traces.append({
            "type": "scatter",
            "mode": "lines",
            "name": theme_names.get(tid, tid),
            "x": x_vals,
            "y": y_vals,
            "line": {"color": _PALETTE[i % len(_PALETTE)], "width": 2},
            "hovertemplate": "%{x}<br>%{y:.2%}<extra>" + theme_names.get(tid, tid) + "</extra>",
        })
    layout = {
        "margin": {"l": 60, "r": 30, "t": 30, "b": 60},
        "yaxis": {"tickformat": ".0%", "title": "Share", "range": [0, 1]},
        "xaxis": {"title": "Date"},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        "hovermode": "x unified",
        "height": 520,
    }
    return _render_plotly_fragment("timeseries-share-lines-chart", traces, layout)


def _render_domain_bar(domain_counts: Dict[str, int]) -> str:
    """Render top media sources bar chart."""
    if not domain_counts:
        return ""
    
    # Sort and take top 20
    items = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    domains = [d for d, _ in items]
    counts = [c for _, c in items]
    
    data = [{
        "type": "bar",
        "x": domains,
        "y": counts,
        "marker": {"color": "#1E3D58", "opacity": 0.85},
        "hovertemplate": "<b>%{x}</b><br>Documents: %{y}<extra></extra>"
    }]
    
    layout = {
        "title": {"text": "Top Media Sources", "font": {"size": 18, "color": "#1e293b"}},
        "xaxis": {"title": "Domain", "tickangle": -40},
        "yaxis": {"title": "Documents"},
        "margin": {"l": 60, "r": 20, "t": 60, "b": 100},
        "plot_bgcolor": "rgba(0,0,0,0)",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter, sans-serif", "size": 12}
    }
    
    return _render_plotly_fragment("domain-chart", data, layout)


def _render_cooccurrence_heatmap(co_matrix: Dict[Tuple[str, str], int], theme_order: List[str], theme_names: Dict[str, str], total_docs: int) -> str:
    """Render theme co-occurrence heatmap."""
    if not co_matrix or not total_docs:
        return ""
    
    # Build percentage matrix
    z_rows = []
    labels = [theme_names.get(t, t) for t in theme_order]
    for a in theme_order:
        row = []
        for b in theme_order:
            v = co_matrix.get((a, b), 0)
            pct = (v / total_docs) * 100.0
            row.append(pct)
        z_rows.append(row)
    
    data = [{
        "type": "heatmap",
        "z": z_rows,
        "x": labels,
        "y": labels,
        "colorscale": "Blues",
        "hovertemplate": "<b>%{y} × %{x}</b><br>Co-occurrence: %{z:.2f}%<extra></extra>",
        "showscale": True
    }]
    
    layout = {
        "title": {"text": "Theme Co-occurrence (% of documents)", "font": {"size": 18, "color": "#1e293b"}},
        "xaxis": {"tickangle": -45},
        "yaxis": {"autorange": "reversed"},
        "margin": {"l": 80, "r": 20, "t": 60, "b": 80},
        "plot_bgcolor": "rgba(0,0,0,0)",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": "Inter, sans-serif", "size": 12}
    }
    
    return _render_plotly_fragment("cooccurrence-chart", data, layout)


def generate_html_report(
    output_path: Path,
    *,
    case_title: str,
    theme_counts: Dict[str, int],
    daily_counts: Dict[str, Dict[str, int]],
    domain_counts: Dict[str, int],
    cooccurrence: Dict[Tuple[str, str], int],
    top_docs: Dict[str, List[ThemeDocStat]],
    theme_names: Dict[str, str],
    total_docs: int,
    date_from: Optional[str],
    date_to: Optional[str],
    document_aggregates: Sequence[DocumentFrameAggregate],
    daily_total_docs: Dict[str, int],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate Plotly charts
    theme_chart = _render_theme_bar(theme_counts, theme_names)
    total_docs_chart = _render_total_docs_timeseries(daily_total_docs)
    timeseries_chart = _render_timeseries_lines(daily_counts, theme_names)
    domain_chart = _render_domain_bar(domain_counts)
    cooccurrence_chart = _render_cooccurrence_heatmap(cooccurrence, list(theme_counts.keys()), theme_names, total_docs)
    # Narrative-like charts based on document aggregates
    theme_share_chart = _render_occurrence_percentage_bars(document_aggregates, list(theme_counts.keys()), theme_names,
                                                          title="Theme Coverage (% of documents)")
    by_year_chart = _render_occurrence_by_year(document_aggregates, list(theme_counts.keys()), theme_names)
    share_trend_chart = _render_timeseries_share_lines(document_aggregates, list(theme_counts.keys()), theme_names)

    # Build top stories HTML
    top_html_parts: List[str] = []
    for tid, items in top_docs.items():
        if not items:
            continue
        tname = theme_names.get(tid, tid)
        rows = []
        for it in items[:10]:
            title = it.title or it.url
            rows.append(
                f"<tr><td>{it.date}</td><td><a href=\"{it.url}\" target=\"_blank\">{title}</a></td><td>{it.domain}</td><td>{it.hits}</td></tr>"
            )
        table = (
            f"<h3>{tname}</h3>"
            "<table class=\"results-table\">"
            "<thead><tr><th>Date</th><th>Title</th><th>Domain</th><th>Keyword hits</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )
        top_html_parts.append(table)

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{case_title} – Word Count Report</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
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
      padding: 12px 16px;
      background: var(--slate-100);
      border-left: 3px solid var(--accent-2);
      border-radius: 0 6px 6px 0;
      font-size: 0.9rem;
      color: var(--ink-600);
    }}
    .chart-container {{
      background: #ffffff;
      border-radius: 12px;
      padding: 24px;
      box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
      margin: 20px 0;
    }}
    .plotly-chart {{
      width: 100%;
      height: 500px;
    }}
    .grid-2 {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 32px;
    }}
    @media (min-width: 1200px) {{
      .grid-2 {{ grid-template-columns: 1fr 1fr; }}
    }}
    .results-table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 16px;
      background: #ffffff;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
    }}
    .results-table th {{
      background: var(--slate-100);
      color: var(--ink-800);
      font-weight: 600;
      padding: 12px 16px;
      text-align: left;
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .results-table td {{
      padding: 12px 16px;
      border-top: 1px solid var(--border);
      font-size: 0.9rem;
    }}
    .results-table tr:hover {{
      background: var(--slate-50);
    }}
    .theme-section {{
      margin-bottom: 32px;
    }}
    .theme-section h3 {{
      color: var(--ink-900);
      margin: 0 0 12px 0;
      font-size: 1.3rem;
    }}
  </style>
</head>
<body>
  <div class="report-page">
    <div class="report-header">
      <div class="heading-text">
        <div class="eyebrow">Word Count Analysis</div>
        <h1>{case_title}</h1>
        <p>Keyword frequency analysis across media sources</p>
        <div class="timeline-note">
          <strong>Date range:</strong> {(date_from or '—')} to {(date_to or '—')}
        </div>
      </div>
      <div class="header-metrics">
        <div class="metric">
          <span class="metric-value">{total_docs:,}</span>
          <span class="metric-label">Documents</span>
        </div>
      </div>
  </div>

    <div class="report-section">
      <div class="section-heading">
    <h2>Theme Coverage</h2>
    <p>Number of documents mentioning each sector-specific theme.</p>
      </div>
      <div class="section-body">
        <div class="chart-container">
          {theme_chart if theme_chart else '<p>No data available</p>'}
        </div>
      </div>
  </div>

    <div class="report-section">
      <div class="section-heading">
        <h2>Document Aggregates</h2>
        <p>Percentage share and distribution of themes across documents.</p>
      </div>
      <div class="section-body">
        <div class="chart-container">
          {theme_share_chart if theme_share_chart else '<p>No data available</p>'}
        </div>
        <div class="chart-container">
          <h3 style="margin-top: 0; color: var(--ink-900);">Yearly Distribution by Theme</h3>
          {by_year_chart if by_year_chart else '<p>No data available</p>'}
        </div>
        <div class="chart-container">
          <h3 style="margin-top: 0; color: var(--ink-900);">Share Trend by Theme (30-day average)</h3>
          {share_trend_chart if share_trend_chart else '<p>No data available</p>'}
        </div>
      </div>
    </div>

    <div class="report-section">
      <div class="section-heading">
        <h2>Article Volume Over Time</h2>
        <p>Total number of articles per day (7-day rolling average).</p>
      </div>
      <div class="section-body">
        <div class="chart-container">
          {total_docs_chart if total_docs_chart else '<p>No data available</p>'}
        </div>
      </div>
    </div>

    <div class="report-section">
      <div class="section-heading">
    <h2>Daily Trend by Theme</h2>
        <p>7-day rolling average of documents per day showing temporal patterns.</p>
      </div>
      <div class="section-body">
        <div class="chart-container">
          {timeseries_chart if timeseries_chart else '<p>No data available</p>'}
        </div>
      </div>
  </div>

    <div class="report-section">
      <div class="section-heading">
        <h2>Media Sources & Co-occurrence</h2>
        <p>Top media sources and theme co-occurrence patterns.</p>
      </div>
      <div class="section-body">
        <div class="grid-2">
          <div class="chart-container">
            <h3 style="margin-top: 0; color: var(--ink-900);">Top Media Sources</h3>
            {domain_chart if domain_chart else '<p>No data available</p>'}
          </div>
          <div class="chart-container">
            <h3 style="margin-top: 0; color: var(--ink-900);">Theme Co-occurrence</h3>
            <p style="margin: 8px 0 16px 0; color: var(--ink-500); font-size: 0.9rem;">
              Share of documents where themes co-occur (percentage of all docs).
            </p>
            {cooccurrence_chart if cooccurrence_chart else '<p>No data available</p>'}
          </div>
        </div>
      </div>
    </div>

    <div class="report-section">
      <div class="section-heading">
        <h2>Top Stories per Theme</h2>
        <p>Most relevant articles for each theme based on keyword frequency.</p>
      </div>
      <div class="section-body">
        {''.join(f'<div class="theme-section">{table}</div>' for table in top_html_parts)}
      </div>
    </div>
  </div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path
