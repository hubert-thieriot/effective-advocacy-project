"""Visualization helpers for narrative framing reports."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def render_frame_area_chart(
    time_series: pd.DataFrame,
    frame_order: Sequence[str],
    frame_labels: Dict[str, str],
    output_path: Path,
    rolling_window: int = 30,
) -> Optional[Path]:
    """Render a stacked area chart describing frame share over time with optional running average."""

    if time_series.empty:
        return None

    pivot = (
        time_series.pivot(index="date", columns="frame_id", values="share")
        .reindex(columns=frame_order)
        .fillna(0.0)
        .sort_index()
    )
    if pivot.empty or pivot.shape[1] == 0:
        return None

    # Calculate rolling average if window is specified and we have enough data
    if rolling_window > 1 and len(pivot) > rolling_window:
        pivot_rolling = pivot.rolling(window=rolling_window, min_periods=1).mean()
        title_suffix = f" (30-day running average)"
    else:
        pivot_rolling = pivot
        title_suffix = ""

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = sns.color_palette("Set3", n_colors=len(frame_order))
    stack_values = [pivot_rolling[column].to_numpy() for column in pivot_rolling.columns]

    ax.stackplot(
        pivot_rolling.index,
        *stack_values,
        labels=[frame_labels.get(fid, fid) for fid in pivot_rolling.columns],
        colors=colors,
        alpha=0.8,
        edgecolors='white',
        linewidth=0.5
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Frame Share", fontsize=12, fontweight='bold')
    ax.set_xlabel("Date", fontsize=12, fontweight='bold')
    ax.set_title(f"Frame Evolution Over Time{title_suffix}", fontsize=14, fontweight='bold', pad=20)
    
    # Improve legend
    ax.legend(
        loc="upper left", 
        bbox_to_anchor=(1.0, 1.0), 
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=10
    )
    
    # Add grid
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300, facecolor='white')
    plt.close(fig)
    return output_path


def image_to_base64(path: Path) -> str:
    """Return the base64 encoding for an image on disk."""

    return base64.b64encode(path.read_bytes()).decode("ascii")


__all__ = ["render_frame_area_chart", "image_to_base64"]
