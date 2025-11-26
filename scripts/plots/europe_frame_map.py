#!/usr/bin/env python3
"""Generate a Europe map colored by frame scores from narrative framing results.

Usage:
    poetry run python scripts/plots/europe_frame_map.py results/narrative_framing/manifesto_europe_animalwelfare
    poetry run python scripts/plots/europe_frame_map.py results/narrative_framing/manifesto_europe_animalwelfare --output map.png
    poetry run python scripts/plots/europe_frame_map.py results/narrative_framing/manifesto_europe_animalwelfare --metric mean --frame anti_hunting
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import geodatasets
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


# Map country names (as they appear in data) to ISO 3166-1 alpha-3 codes
COUNTRY_TO_ISO3 = {
    # Standard names
    "Albania": "ALB",
    "Austria": "AUT",
    "Belarus": "BLR",
    "Belgium": "BEL",
    "Bosnia-Herzegovina": "BIH",
    "Bosnia and Herzegovina": "BIH",
    "Bulgaria": "BGR",
    "Croatia": "HRV",
    "Cyprus": "CYP",
    "Czech Republic": "CZE",
    "Czechia": "CZE",
    "Denmark": "DNK",
    "Estonia": "EST",
    "Finland": "FIN",
    "France": "FRA",
    "Germany": "DEU",
    "Greece": "GRC",
    "Hungary": "HUN",
    "Iceland": "ISL",
    "Ireland": "IRL",
    "Italy": "ITA",
    "Latvia": "LVA",
    "Lithuania": "LTU",
    "Luxembourg": "LUX",
    "Malta": "MLT",
    "Moldova": "MDA",
    "Montenegro": "MNE",
    "Netherlands": "NLD",
    "North Macedonia": "MKD",
    "Norway": "NOR",
    "Poland": "POL",
    "Portugal": "PRT",
    "Romania": "ROU",
    "Serbia": "SRB",
    "Slovakia": "SVK",
    "Slovenia": "SVN",
    "Spain": "ESP",
    "Sweden": "SWE",
    "Switzerland": "CHE",
    "Ukraine": "UKR",
    "United Kingdom": "GBR",
    "Northern Ireland": "GBR",
    "Kosovo": "KOS",
    # Natural Earth name variants
    "Bosnia and Herz.": "BIH",
    "Czechia": "CZE",
    "N. Macedonia": "MKD",
}

# Reverse mapping: ISO3 to display name
ISO3_TO_NAME = {
    "ALB": "Albania",
    "AUT": "Austria",
    "BLR": "Belarus",
    "BEL": "Belgium",
    "BIH": "Bosnia-Herzegovina",
    "BGR": "Bulgaria",
    "HRV": "Croatia",
    "CYP": "Cyprus",
    "CZE": "Czech Republic",
    "DNK": "Denmark",
    "EST": "Estonia",
    "FIN": "Finland",
    "FRA": "France",
    "DEU": "Germany",
    "GRC": "Greece",
    "HUN": "Hungary",
    "ISL": "Iceland",
    "IRL": "Ireland",
    "ITA": "Italy",
    "LVA": "Latvia",
    "LTU": "Lithuania",
    "LUX": "Luxembourg",
    "MLT": "Malta",
    "MDA": "Moldova",
    "MNE": "Montenegro",
    "NLD": "Netherlands",
    "MKD": "North Macedonia",
    "NOR": "Norway",
    "POL": "Poland",
    "PRT": "Portugal",
    "ROU": "Romania",
    "SRB": "Serbia",
    "SVK": "Slovakia",
    "SVN": "Slovenia",
    "ESP": "Spain",
    "SWE": "Sweden",
    "CHE": "Switzerland",
    "UKR": "Ukraine",
    "GBR": "United Kingdom",
    "KOS": "Kosovo",
}


def load_corpus_index(results_dir: Path) -> Dict[str, dict]:
    """Load corpus index to map doc_id to metadata (including country)."""
    # Try to find the corpus from the config
    config_dir = results_dir / "configs"
    corpus_name = None
    
    if config_dir.exists():
        for config_file in sorted(config_dir.glob("*.yaml"), reverse=True):
            try:
                import yaml
                config = yaml.safe_load(config_file.read_text())
                corpus_name = config.get("corpus")
                if corpus_name:
                    break
            except Exception:
                continue
    
    if not corpus_name:
        # Try to infer from results folder name
        corpus_name = results_dir.name.replace("_animalwelfare", "").replace("_", "_")
    
    # Look for corpus index
    corpus_paths = [
        Path("corpora") / corpus_name / "index.jsonl",
        Path("corpora") / f"{corpus_name}" / "index.jsonl",
    ]
    
    for corpus_path in corpus_paths:
        if corpus_path.exists():
            index = {}
            for line in corpus_path.read_text().splitlines():
                if line.strip():
                    doc = json.loads(line)
                    index[doc["id"]] = doc
            return index
    
    raise FileNotFoundError(f"Could not find corpus index for {corpus_name}")


def load_document_aggregates(results_dir: Path) -> List[dict]:
    """Load document-level frame aggregates, computed directly from LLM annotations."""
    # Load raw LLM assignments
    assignments_path = results_dir / "frame_assignments.json"
    if not assignments_path.exists():
        raise FileNotFoundError(f"No frame_assignments.json found in {results_dir}")
    
    assignments = json.loads(assignments_path.read_text())
    
    # Aggregate by document (weighted by text length)
    from collections import defaultdict
    docs = defaultdict(lambda: {"total_weight": 0, "frame_sums": defaultdict(float)})
    
    for a in assignments:
        passage_id = a.get("passage_id", "")
        doc_id = passage_id.split(":")[0] if ":" in passage_id else passage_id
        text = a.get("passage_text", "")
        weight = len(text)
        probs = a.get("probabilities", {})
        
        docs[doc_id]["total_weight"] += weight
        for frame, prob in probs.items():
            docs[doc_id]["frame_sums"][frame] += prob * weight
    
    # Convert to list format expected by aggregate_by_country
    result = []
    for doc_id, data in docs.items():
        if data["total_weight"] > 0:
            frame_scores = {
                frame: s / data["total_weight"] 
                for frame, s in data["frame_sums"].items()
            }
        else:
            frame_scores = dict(data["frame_sums"])
        
        result.append({
            "doc_id": doc_id,
            "frame_scores": frame_scores,
            "total_weight": data["total_weight"],
        })
    
    return result


def aggregate_by_country(
    documents: List[dict],
    corpus_index: Dict[str, dict],
    metric: str = "mean",
    frame: Optional[str] = None,
) -> Dict[str, float]:
    """Aggregate frame scores by country (using ISO3 codes).
    
    Args:
        documents: Document aggregates with frame_scores
        corpus_index: Mapping of doc_id to metadata
        metric: "mean" or "sum" - how to aggregate scores per country
        frame: Specific frame to use, or None for sum of all frames
    
    Returns:
        Dict mapping ISO3 country code to aggregated score
    """
    country_scores: Dict[str, List[float]] = {}
    
    for doc in documents:
        doc_id = doc.get("doc_id", "")
        if not doc_id:
            continue
        
        # Look up country from corpus index
        meta = corpus_index.get(doc_id, {})
        country_name = meta.get("country_name")
        if not country_name:
            continue
        
        # Convert to ISO3
        iso3 = COUNTRY_TO_ISO3.get(country_name)
        if not iso3:
            print(f"⚠️  Unknown country: {country_name}")
            continue
        
        # Calculate score for this document
        frame_scores = doc.get("frame_scores", {})
        if frame:
            score = frame_scores.get(frame, 0.0)
        else:
            # Sum of all frame scores
            score = sum(frame_scores.values())
        
        if iso3 not in country_scores:
            country_scores[iso3] = []
        country_scores[iso3].append(score)
    
    # Aggregate per country
    result = {}
    for iso3, scores in country_scores.items():
        if metric == "mean":
            result[iso3] = np.mean(scores) if scores else 0.0
        elif metric == "sum":
            result[iso3] = sum(scores)
        else:
            result[iso3] = np.mean(scores) if scores else 0.0
    
    return result


def get_europe_geodata() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load Natural Earth country boundaries for Europe (50m resolution)."""
    # Download Natural Earth countries data (50m = medium resolution, good balance)
    url = "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip"
    world = gpd.read_file(url)
    
    # Use ADM0_A3 as primary (more reliable than ISO_A3 which has -99 for France, Kosovo, etc.)
    iso3_col = None
    for col in ["ADM0_A3", "adm0_a3", "ISO_A3", "iso_a3"]:
        if col in world.columns:
            iso3_col = col
            break
    
    if iso3_col is None:
        raise ValueError(f"Could not find ISO3 column. Available: {list(world.columns)}")
    
    # Normalize to 'iso3' column
    world["iso3"] = world[iso3_col]
    
    # Filter to Europe by ISO3 codes
    europe_iso3 = set(ISO3_TO_NAME.keys())
    europe = world[world["iso3"].isin(europe_iso3)].copy()
    
    # Also get context countries (neighbors) for background
    context_iso3 = {"RUS", "TUR", "MAR", "DZA", "TUN", "LBY", "EGY"}  # Russia, Turkey, North Africa
    context = world[world["iso3"].isin(context_iso3)].copy()
    
    return europe, context
    
    # Also get name column
    name_col = None
    for col in ["NAME", "name", "ADMIN", "admin"]:
        if col in europe.columns:
            name_col = col
            break
    if name_col:
        europe["name"] = europe[name_col]
    
    return europe


def create_europe_map(
    country_scores: Dict[str, float],
    output_path: Optional[Path] = None,
    cmap: str = "Greens",
    figsize: tuple = (12, 10),
    colorbar_label: str = "Weight of animal-welfare related frames",
) -> None:
    """Create a Europe map colored by country scores.
    
    Args:
        country_scores: Dict mapping ISO3 code to score
        output_path: Path to save the figure (PNG or SVG)
        cmap: Matplotlib colormap name
        figsize: Figure size in inches
        colorbar_label: Label for the colorbar
    """
    # Set font to Open Sans (same as docs website)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Open Sans', 'Helvetica Neue', 'Arial', 'sans-serif']
    
    # Load geographic data (europe + context countries)
    europe, context = get_europe_geodata()
    
    # Reproject to EPSG:3857 (Web Mercator)
    europe = europe.to_crs(epsg=3857)
    context = context.to_crs(epsg=3857)
    
    # Map scores to countries by ISO3 code
    europe["score"] = europe["iso3"].map(country_scores)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Set background color
    ax.set_facecolor("#f0f0f0")
    
    # Get score range for colormap
    valid_scores = [s for s in europe["score"] if s is not None and s > 0]
    if valid_scores:
        vmin, vmax = 0, max(valid_scores)
    else:
        vmin, vmax = 0, 1
    
    # Plot context countries in very light grey (for geographic context)
    context.plot(
        ax=ax,
        color="#e8e8e8",
        edgecolor="#ffffff",
        linewidth=0.3,
    )
    
    # Plot European countries without data in grey
    europe[europe["score"].isna() | (europe["score"] == 0)].plot(
        ax=ax,
        color="#d0d0d0",
        edgecolor="#ffffff",
        linewidth=0.5,
    )
    
    # Plot countries with data
    countries_with_data = europe[(europe["score"].notna()) & (europe["score"] > 0)]
    if len(countries_with_data) > 0:
        countries_with_data.plot(
            column="score",
            ax=ax,
            legend=False,  # We'll add a custom colorbar
            cmap=cmap,
            edgecolor="#ffffff",
            linewidth=0.5,
            vmin=vmin,
            vmax=vmax,
        )
        
        # Add horizontal colorbar under the map (smaller, label on right, no ticks)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar_ax = fig.add_axes([0.25, 0.08, 0.3, 0.012])  # [left, bottom, width, height]
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_ticks([])  # No ticks/numbers
        cbar.outline.set_visible(False)
        # Add label to the right of the colorbar (grey text)
        cbar_ax.text(1.05, 0.5, colorbar_label, transform=cbar_ax.transAxes,
                     fontsize=10, va='center', ha='left', color='#666666')
    
    # Crop to mainland Europe (exclude Atlantic islands like Canaries, Azores, Madeira)
    # These are approximate bounds in EPSG:3857 (Web Mercator)
    # Roughly: Iceland to Cyprus (lon), Portugal to Norway (lat)
    ax.set_xlim(-2_800_000, 5_000_000)   # ~-25° to ~45° longitude
    ax.set_ylim(4_000_000, 11_500_000)   # ~34° to ~72° latitude
    
    # No title, no caption, just the map
    ax.axis("off")
    
    # Save or show
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine format from extension
        fmt = output_path.suffix.lower().lstrip(".")
        if fmt not in ("png", "svg", "pdf"):
            fmt = "png"
        
        plt.savefig(
            output_path,
            format=fmt,
            dpi=150 if fmt == "png" else None,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        print(f"✅ Saved map to {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate Europe map colored by frame scores"
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Path to narrative framing results folder",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output file path (PNG or SVG). If not specified, displays interactively.",
    )
    parser.add_argument(
        "--metric",
        choices=["sum", "mean"],
        default="sum",
        help="Aggregation metric: sum (total presence) or mean (average per document)",
    )
    parser.add_argument(
        "--frame",
        type=str,
        default=None,
        help="Specific frame to visualize. If not specified, uses sum of all frames.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="Greens",
        help="Matplotlib colormap (default: Greens)",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use random scores for testing visualization",
    )
    
    args = parser.parse_args()
    
    if not args.results_dir.exists():
        print(f"❌ Results directory not found: {args.results_dir}")
        return 1
    
    print(f"📂 Loading data from {args.results_dir}")
    
    # Load data
    corpus_index = load_corpus_index(args.results_dir)
    print(f"   Loaded {len(corpus_index)} documents from corpus index")
    
    documents = load_document_aggregates(args.results_dir)
    print(f"   Loaded {len(documents)} document aggregates")
    
    # Aggregate by country (or use random for testing)
    if args.random:
        # Generate random scores for all European countries
        country_scores = {iso3: np.random.uniform(0.1, 1.0) for iso3 in ISO3_TO_NAME.keys()}
        print(f"   Generated random scores for {len(country_scores)} countries")
    else:
        country_scores = aggregate_by_country(
            documents,
            corpus_index,
            metric=args.metric,
            frame=args.frame,
        )
        print(f"   Aggregated scores for {len(country_scores)} countries")
    
    # Print summary
    if country_scores:
        sorted_scores = sorted(country_scores.items(), key=lambda x: x[1], reverse=True)
        print("\n📊 Top countries by score:")
        for country, score in sorted_scores[:10]:
            name = ISO3_TO_NAME.get(country, country)
            print(f"   {name}: {score:.4f}")
    
    # Create map
    print(f"\n🗺️  Generating map...")
    create_europe_map(
        country_scores,
        output_path=args.output,
        cmap=args.cmap,
    )
    
    return 0


if __name__ == "__main__":
    exit(main())

