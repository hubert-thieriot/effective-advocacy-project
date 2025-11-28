"""Europe map plotters for narrative framing results."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from .base import BasePlotter, PlotConfig
from .registry import register_plotter
from ._utils import load_corpus_index, load_document_aggregates


# Map country names (as they appear in data) to ISO 3166-1 alpha-3 codes
COUNTRY_TO_ISO3 = {
    "Albania": "ALB", "Austria": "AUT", "Belarus": "BLR", "Belgium": "BEL",
    "Bosnia-Herzegovina": "BIH", "Bosnia and Herzegovina": "BIH", "Bulgaria": "BGR",
    "Croatia": "HRV", "Cyprus": "CYP", "Czech Republic": "CZE", "Czechia": "CZE",
    "Denmark": "DNK", "Estonia": "EST", "Finland": "FIN", "France": "FRA",
    "Germany": "DEU", "Greece": "GRC", "Hungary": "HUN", "Iceland": "ISL",
    "Ireland": "IRL", "Italy": "ITA", "Latvia": "LVA", "Lithuania": "LTU",
    "Luxembourg": "LUX", "Malta": "MLT", "Moldova": "MDA", "Montenegro": "MNE",
    "Netherlands": "NLD", "North Macedonia": "MKD", "Norway": "NOR", "Poland": "POL",
    "Portugal": "PRT", "Romania": "ROU", "Serbia": "SRB", "Slovakia": "SVK",
    "Slovenia": "SVN", "Spain": "ESP", "Sweden": "SWE", "Switzerland": "CHE",
    "Ukraine": "UKR", "United Kingdom": "GBR", "Northern Ireland": "GBR",
    "Kosovo": "KOS", "Bosnia and Herz.": "BIH", "N. Macedonia": "MKD",
}

ISO3_TO_NAME = {v: k for k, v in COUNTRY_TO_ISO3.items() if k not in ["Bosnia and Herz.", "N. Macedonia"]}
ISO3_TO_NAME.update({
    "ALB": "Albania", "AUT": "Austria", "BLR": "Belarus", "BEL": "Belgium",
    "BIH": "Bosnia-Herzegovina", "BGR": "Bulgaria", "HRV": "Croatia", "CYP": "Cyprus",
    "CZE": "Czech Republic", "DNK": "Denmark", "EST": "Estonia", "FIN": "Finland",
    "FRA": "France", "DEU": "Germany", "GRC": "Greece", "HUN": "Hungary",
    "ISL": "Iceland", "IRL": "Ireland", "ITA": "Italy", "LVA": "Latvia",
    "LTU": "Lithuania", "LUX": "Luxembourg", "MLT": "Malta", "MDA": "Moldova",
    "MNE": "Montenegro", "NLD": "Netherlands", "MKD": "North Macedonia", "NOR": "Norway",
    "POL": "Poland", "PRT": "Portugal", "ROU": "Romania", "SRB": "Serbia",
    "SVK": "Slovakia", "SVN": "Slovenia", "ESP": "Spain", "SWE": "Sweden",
    "CHE": "Switzerland", "UKR": "Ukraine", "GBR": "United Kingdom", "KOS": "Kosovo",
})

FRAME_COLORS = {
    "1": "#a65628", "2": "#984ea3", "3": "#ff7f00", "4": "#ffff33",
    "5": "#377eb8", "6": "#e41a1c", "7": "#4daf4a", "8": "#f781bf", "9": "#66c2a5",
    "factory_farming": "#e41a1c", "cruel_treatments": "#984ea3", "anti_hunting": "#4daf4a",
    "pet_companion": "#ff7f00", "plant_based": "#377eb8", "cage_free": "#ffff33",
    "slaughterhouses": "#a65628", "anti_lab_experiments": "#f781bf",
    "general_animal_welfare": "#66c2a5",
}

FRAME_DISPLAY_NAMES = {
    "1": "Slaughterhouses", "2": "Cruel Treatments", "3": "Pet/Companion Animals",
    "4": "Cage-Free", "5": "Plant-Based", "6": "Factory Farming",
    "7": "Anti-Hunting", "8": "Anti-Lab Experiments", "9": "General Animal Welfare",
    "factory_farming": "Factory Farming", "cruel_treatments": "Cruel Treatments",
    "anti_hunting": "Anti-Hunting", "pet_companion": "Pet/Companion Animals",
    "plant_based": "Plant-Based", "cage_free": "Cage-Free",
    "slaughterhouses": "Slaughterhouses", "anti_lab_experiments": "Anti-Lab Experiments",
    "general_animal_welfare": "General Animal Welfare",
}


def _aggregates_to_dicts(aggregates) -> List[dict]:
    """Convert Aggregates.documents_weighted to list of dicts."""
    docs = getattr(aggregates, "documents_weighted", None) or []
    return [
        {
            "doc_id": getattr(d, "doc_id", ""),
            "frame_scores": getattr(d, "frame_scores", {}),
            "total_weight": getattr(d, "total_weight", 1),
        }
        for d in docs
    ]


def aggregate_by_country(
    documents: List[dict],
    corpus_index: Dict[str, dict],
    metric: str = "mean",
    frame: Optional[str] = None,
) -> Dict[str, float]:
    """Aggregate frame scores by country (using ISO3 codes)."""
    country_scores: Dict[str, List[float]] = {}
    
    for doc in documents:
        doc_id = doc.get("doc_id", "")
        if not doc_id:
            continue
        
        country_name = corpus_index.get(doc_id, {}).get("extra", {}).get("country_name")
        if not country_name:
            continue
        
        iso3 = COUNTRY_TO_ISO3.get(country_name)
        if not iso3:
            continue
        
        frame_scores = doc.get("frame_scores", {})
        if frame:
            score = frame_scores.get(frame, 0.0)
        else:
            score = sum(frame_scores.values())
        
        if iso3 not in country_scores:
            country_scores[iso3] = []
        country_scores[iso3].append(score)
    
    result = {}
    for iso3, scores in country_scores.items():
        if metric == "mean":
            result[iso3] = np.mean(scores) if scores else 0.0
        elif metric == "sum":
            result[iso3] = sum(scores)
        else:
            result[iso3] = np.mean(scores) if scores else 0.0
    
    return result


def aggregate_dominant_frame_by_country(
    documents: List[dict],
    corpus_index: Dict[str, dict],
    exclude_frames: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Find the dominant frame for each country."""
    exclude_set = set(exclude_frames or [])
    country_frame_scores: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    country_weights: Dict[str, float] = defaultdict(float)
    
    for doc in documents:
        doc_id = doc.get("doc_id", "")
        if not doc_id:
            continue
        
        country_name = corpus_index.get(doc_id, {}).get("extra", {}).get("country_name")
        if not country_name:
            continue
        
        iso3 = COUNTRY_TO_ISO3.get(country_name)
        if not iso3:
            continue
        
        frame_scores = doc.get("frame_scores", {})
        weight = doc.get("total_weight", 1)
        
        for frame, score in frame_scores.items():
            if frame not in exclude_set:
                country_frame_scores[iso3][frame] += score * weight
        country_weights[iso3] += weight
    
    result = {}
    for iso3 in country_frame_scores:
        frame_scores = country_frame_scores[iso3]
        if frame_scores and country_weights[iso3] > 0:
            normalized = {f: s / country_weights[iso3] for f, s in frame_scores.items()}
            dominant = max(normalized.items(), key=lambda x: x[1])
            if dominant[1] > 0:
                result[iso3] = dominant[0]
    
    return result


def _get_europe_geodata() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load Natural Earth country boundaries for Europe (50m resolution)."""
    import geodatasets
    url = "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip"
    world = gpd.read_file(url)
    
    iso3_col = None
    for col in ["ADM0_A3", "adm0_a3", "ISO_A3", "iso_a3"]:
        if col in world.columns:
            iso3_col = col
            break
    
    if iso3_col is None:
        raise ValueError(f"Could not find ISO3 column. Available: {list(world.columns)}")
    
    world["iso3"] = world[iso3_col]
    europe_iso3 = set(ISO3_TO_NAME.keys())
    europe = world[world["iso3"].isin(europe_iso3)].copy()
    context_iso3 = {"RUS", "TUR", "MAR", "DZA", "TUN", "LBY", "EGY"}
    context = world[world["iso3"].isin(context_iso3)].copy()
    
    return europe, context


def _save_figure(output_path: Optional[Path]) -> None:
    """Save or show the current figure."""
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fmt = output_path.suffix.lower().lstrip(".")
        if fmt not in ("png", "svg", "pdf", "jpg"):
            fmt = "png"
        plt.savefig(
            output_path,
            format=fmt,
            dpi=150 if fmt in ("png", "jpg") else None,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        print(f"✅ Saved map to {output_path}")
    else:
        plt.show()
    plt.close()


def create_europe_map(
    country_scores: Dict[str, float],
    output_path: Optional[Path] = None,
    cmap: str = "Greens",
    figsize: tuple = (12, 10),
    colorbar_label: str = "Weight of animal-welfare related frames",
) -> None:
    """Create a Europe map colored by country scores."""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Open Sans', 'Helvetica Neue', 'Arial', 'sans-serif']
    
    europe, context = _get_europe_geodata()
    europe = europe.to_crs(epsg=3857)
    context = context.to_crs(epsg=3857)
    europe["score"] = europe["iso3"].map(country_scores)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_facecolor("#f0f0f0")
    
    valid_scores = [s for s in europe["score"] if s is not None and s > 0]
    if valid_scores:
        vmin, vmax = 0, max(valid_scores)
    else:
        vmin, vmax = 0, 1
    
    context.plot(ax=ax, color="#e8e8e8", edgecolor="#ffffff", linewidth=0.3)
    europe[europe["score"].isna() | (europe["score"] == 0)].plot(
        ax=ax, color="#d0d0d0", edgecolor="#ffffff", linewidth=0.5
    )
    
    countries_with_data = europe[(europe["score"].notna()) & (europe["score"] > 0)]
    if len(countries_with_data) > 0:
        countries_with_data.plot(
            column="score", ax=ax, legend=False, cmap=cmap,
            edgecolor="#ffffff", linewidth=0.5, vmin=vmin, vmax=vmax,
        )
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar_ax = fig.add_axes([0.25, 0.08, 0.3, 0.012])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_ticks([])
        cbar.outline.set_visible(False)
        cbar_ax.text(1.05, 0.5, colorbar_label, transform=cbar_ax.transAxes,
                     fontsize=10, va='center', ha='left', color='#666666')
    
    ax.set_xlim(-2_800_000, 5_000_000)
    ax.set_ylim(4_000_000, 11_500_000)
    ax.axis("off")
    _save_figure(output_path)


def create_dominant_frame_map(
    dominant_frames: Dict[str, str],
    output_path: Optional[Path] = None,
    figsize: tuple = (12, 10),
) -> None:
    """Create a Europe map colored by dominant frame per country."""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Open Sans', 'Helvetica Neue', 'Arial', 'sans-serif']
    
    europe, context = _get_europe_geodata()
    europe = europe.to_crs(epsg=3857)
    context = context.to_crs(epsg=3857)
    europe["dominant_frame"] = europe["iso3"].map(dominant_frames)
    europe["color"] = europe["dominant_frame"].map(FRAME_COLORS)
    europe["color"] = europe["color"].fillna("#d0d0d0")
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_facecolor("#f0f0f0")
    
    context.plot(ax=ax, color="#e8e8e8", edgecolor="#ffffff", linewidth=0.3)
    no_data = europe[europe["dominant_frame"].isna()]
    if len(no_data) > 0:
        no_data.plot(ax=ax, color="#d0d0d0", edgecolor="#ffffff", linewidth=0.5)
    
    with_data = europe[europe["dominant_frame"].notna()]
    if len(with_data) > 0:
        with_data.plot(ax=ax, color=with_data["color"], edgecolor="#ffffff", linewidth=0.5)
    
    frames_present = set(dominant_frames.values())
    legend_patches = [
        mpatches.Patch(color=FRAME_COLORS.get(f, "#888"), label=FRAME_DISPLAY_NAMES.get(f, f))
        for f in sorted(frames_present)
        if f in FRAME_COLORS
    ]
    
    # Create horizontal legend bar underneath the map (similar to colorbar layout)
    if legend_patches:
        # Create a horizontal legend bar at the bottom, similar to colorbar position
        legend_ax = fig.add_axes([0.25, 0.08, 0.5, 0.03])
        legend_ax.axis('off')
        
        # Create horizontal legend with items in a single row
        legend_elements = [
            mpatches.Patch(color=FRAME_COLORS.get(f, "#888"), label=FRAME_DISPLAY_NAMES.get(f, f))
            for f in sorted(frames_present)
            if f in FRAME_COLORS
        ]
        
        # Create horizontal legend in a single row
        n_items = len(legend_elements)
        legend = legend_ax.legend(
            handles=legend_elements,
            loc='center',
            ncol=n_items,  # All items in one row
            fontsize=9,
            frameon=False,
            columnspacing=1.2,
            handlelength=1.2,
            handletextpad=0.4,
        )
        
    
    ax.set_xlim(-2_800_000, 5_000_000)
    ax.set_ylim(4_000_000, 11_500_000)
    ax.axis("off")
    _save_figure(output_path)


@register_plotter
class EuropeMapPlotter(BasePlotter):
    """Generate Europe choropleth map colored by frame scores."""
    
    name = "europe_map"
    
    def plot(self, config: PlotConfig) -> Optional[Path]:
        if self.state.aggregates:
            documents = _aggregates_to_dicts(self.state.aggregates)
        else:
            documents = load_document_aggregates(self.results_dir)
        
        corpus_index = self.corpus_index or {}
        if not corpus_index:
            corpus_index = load_corpus_index(self.results_dir)
        
        country_scores = aggregate_by_country(
            documents, corpus_index,
            metric=config.get_option("metric", "mean"),
            frame=config.get_option("frame", None),
        )
        
        output_path = self.get_output_path(config, "europe_map.png")
        create_europe_map(
            country_scores,
            output_path=output_path,
            cmap=config.get_option("cmap", "Greens"),
        )
        return output_path


@register_plotter
class DominantFrameMapPlotter(BasePlotter):
    """Generate Europe map colored by dominant frame per country."""
    
    name = "dominant_frame_map"
    
    def plot(self, config: PlotConfig) -> Optional[Path]:
        if self.state.aggregates:
            documents = _aggregates_to_dicts(self.state.aggregates)
        else:
            documents = load_document_aggregates(self.results_dir)
        
        corpus_index = self.corpus_index or {}
        if not corpus_index:
            corpus_index = load_corpus_index(self.results_dir)
        
        dominant_frames = aggregate_dominant_frame_by_country(
            documents, corpus_index,
            exclude_frames=config.get_option("exclude_frames", None),
        )
        
        output_path = self.get_output_path(config, "europe_dominant_frame.png")
        create_dominant_frame_map(dominant_frames, output_path=output_path)
        return output_path
