"""Party bar chart plotters for narrative framing results."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence
import csv
import io
import json
import html

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import plotly.graph_objects as go

from .base import BasePlotter, PlotConfig
from .registry import register_plotter
from ._utils import load_corpus_index
from ..types import FrameAssignment


COUNTRY_FLAGS = {
    "Albania": "🇦🇱", "Austria": "🇦🇹", "Belgium": "🇧🇪", "Bosnia-Herzegovina": "🇧🇦",
    "Bulgaria": "🇧🇬", "Croatia": "🇭🇷", "Cyprus": "🇨🇾", "Czech Republic": "🇨🇿",
    "Denmark": "🇩🇰", "Estonia": "🇪🇪", "Finland": "🇫🇮", "France": "🇫🇷",
    "Germany": "🇩🇪", "Greece": "🇬🇷", "Hungary": "🇭🇺", "Iceland": "🇮🇸",
    "Ireland": "🇮🇪", "Italy": "🇮🇹", "Latvia": "🇱🇻", "Lithuania": "🇱🇹",
    "Luxembourg": "🇱🇺", "Malta": "🇲🇹", "Montenegro": "🇲🇪", "Netherlands": "🇳🇱",
    "North Macedonia": "🇲🇰", "Norway": "🇳🇴", "Poland": "🇵🇱", "Portugal": "🇵🇹",
    "Romania": "🇷🇴", "Serbia": "🇷🇸", "Slovakia": "🇸🇰", "Slovenia": "🇸🇮",
    "Spain": "🇪🇸", "Sweden": "🇸🇪", "Switzerland": "🇨🇭", "Ukraine": "🇺🇦",
    "United Kingdom": "🇬🇧",
}

COUNTRY_COLORS = {
    "Netherlands": "#FF6B00", "Portugal": "#006600", "Germany": "#000000",
    "Sweden": "#006AA7", "United Kingdom": "#C8102E", "Ireland": "#169B62",
    "Hungary": "#477050", "Spain": "#F1BF00", "France": "#0055A4",
    "Montenegro": "#C40C0C", "Serbia": "#C6363C", "Lithuania": "#006A44",
    "Malta": "#CF142B", "Greece": "#0D5EAF", "Bosnia-Herzegovina": "#002395",
    "Italy": "#008C45", "Denmark": "#C60C30", "Finland": "#003580",
    "Poland": "#DC143C", "Slovenia": "#005DA4", "Ukraine": "#FFD500",
    "Estonia": "#0072CE", "Romania": "#002B7F", "Croatia": "#FF0000",
    "Iceland": "#02529C", "Latvia": "#9E3039", "Albania": "#E41E20",
}



def _assignments_to_dicts(assignments) -> List[dict]:
    """Convert FrameAssignments to list of dicts."""
    return [
        {
            "passage_id": x.passage_id,
            "passage_text": x.passage_text,
            "probabilities": x.probabilities,
        }
        for x in assignments
    ]


# Cache for party data from CSV
_PARTY_NAMES_CACHE: Optional[Dict[str, Dict[str, str]]] = None  # party_id -> {name_english, abbrev}
_PARTY_FAMILIES_CACHE: Optional[Dict[str, str]] = None  # party_id -> family

# Political family color palette
POLITICAL_FAMILY_COLORS: Dict[str, str] = {
    "green": "#2ca25f",  # green
    "animalist": "#1d9147",  # slightly darker green
    "centre-left": "#e75480",  # rose
    "centre": "#d4a5a5",  # light rose
    "centre-right": "#1f78b4",  # blue
    "far-left": "#d73027",  # bright red
    "far-right": "#67000d",  # darker brown
    "liberal": "#f6c344",  # yellow
    "other": "#9e9e9e",  # grey
    "unknown": "#cccccc",  # light grey for unknown families
}


def _load_party_names() -> Dict[str, Dict[str, str]]:
    """Load party names from parties_MPDataset_MPDS2025a.csv.
    
    Returns:
        Dict mapping party_id (as string) to dict with 'name', 'name_english', and 'abbrev' keys
    """
    global _PARTY_NAMES_CACHE
    if _PARTY_NAMES_CACHE is not None:
        return _PARTY_NAMES_CACHE
    
    script_dir = Path(__file__).parent
    csv_path = script_dir / "parties_MPDataset_MPDS2025a.csv"
    
    party_names = {}
    
    try:
        if csv_path.exists():
            with csv_path.open('r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    party_id = str(row.get("party", "")).strip()
                    name = row.get("name", "").strip()
                    name_english = row.get("name_english", "").strip()
                    abbrev = row.get("abbrev", "").strip()
                    
                    if party_id:
                        party_names[party_id] = {
                            "name": name,
                            "name_english": name_english,
                            "abbrev": abbrev,
                        }
            
            print(f"✅ Loaded {len(party_names)} party names from Manifesto CSV")
        else:
            print(f"⚠️ Manifesto CSV not found at {csv_path}")
        _PARTY_NAMES_CACHE = party_names
        return party_names
    except Exception as e:
        print(f"⚠️ Could not load party names from CSV: {e}")
        _PARTY_NAMES_CACHE = {}
        return {}


def _load_party_families() -> Dict[str, str]:
    """Load party families from european_parties_with_families_updated.csv.
    
    Returns:
        Dict mapping party_id (as string) to family name
    """
    global _PARTY_FAMILIES_CACHE
    if _PARTY_FAMILIES_CACHE is not None:
        return _PARTY_FAMILIES_CACHE
    
    script_dir = Path(__file__).parent
    csv_path = script_dir / "parties_families.csv"
    
    party_families = {}
    
    try:
        if csv_path.exists():
            with csv_path.open('r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Try both "party" and "party_id" column names
                    party_id = str(row.get("party", "") or row.get("party_id", "")).strip()
                    family = row.get("family", "").strip()
                    
                    if party_id and family:
                        party_families[party_id] = family
            
            print(f"✅ Loaded {len(party_families)} party families from CSV")
        else:
            print(f"⚠️ Families CSV not found at {csv_path}")
        _PARTY_FAMILIES_CACHE = party_families
        return party_families
    except Exception as e:
        print(f"⚠️ Could not load party families from CSV: {e}")
        _PARTY_FAMILIES_CACHE = {}
        return {}


def get_party_name(
    party_id: str,
    use_english_name: bool = False,
    max_length: int = 25
) -> str:
    """Get party name from Manifesto Project CSV.
    
    Args:
        party_id: Party ID from metadata
        use_english_name: If True, use name_english from CSV; if False, prefer abbrev
        max_length: Maximum length for party name. If exceeded, use abbrev
        
    Returns:
        Party name (or abbrev based on use_english_name and max_length)
    """
    if not party_id or party_id == "Unknown":
        return "Unknown"
    
    party_names = _load_party_names()
    if party_id in party_names:
        name = party_names[party_id].get("name", "").strip()  # Original language name
        name_english_raw = party_names[party_id].get("name_english", "").strip()
        abbrev = party_names[party_id].get("abbrev", "").strip()
        
        # If name_english is "NA" or empty, use original name
        if not name_english_raw or name_english_raw.upper() == "NA":
            name_english = name
        else:
            name_english = name_english_raw
        
        # Ensure we have a valid name (not "NA")
        if not name_english or name_english.upper() == "NA":
            name_english = name
        
        if use_english_name:
            # Use full English name (or original if English is NA), but fall back to abbrev if name is too long
            # name_english is already set to name if it was "NA", so we can use it directly
            if name_english and name_english.upper() != "NA":
                if len(name_english) > max_length and abbrev and abbrev.upper() != "NA":
                    return abbrev
                # Enforce max_length even if we use the original name
                if len(name_english) > max_length:
                    return name_english[:max_length-1] + "…"
                return name_english
            elif abbrev and abbrev.upper() != "NA":
                return abbrev
            elif name:
                # Enforce max_length on original name
                if len(name) > max_length:
                    return name[:max_length-1] + "…"
                return name
        else:
            # Prefer abbrev, but fall back to name_english (or original name) if abbrev is not available
            if abbrev and abbrev.upper() != "NA":
                return abbrev
            elif name_english and name_english.upper() != "NA":
                # Even if use_english_name is False, use name if abbrev is missing
                if len(name_english) > max_length:
                    return name_english[:max_length-1] + "…"
                return name_english
            elif name:
                # Enforce max_length on original name
                if len(name) > max_length:
                    return name[:max_length-1] + "…"
                return name
    
    # Fallback to party_id if not found
    return party_id


def _build_party_family_map() -> Dict[str, str]:
    """Create a party_id -> political family map from european_parties_with_families_updated.csv.
    
    Returns:
        Dict mapping party_id (as string) to family name
    """
    return _load_party_families()


def aggregate_by_party(
    assignments: Sequence[FrameAssignment],
    corpus_index: Dict[str, dict],
    use_english_names: bool = False,
    max_length: int = 20,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, str], Dict[str, Optional[str]]]:
    """Aggregate frame assignments by party_id within each country.
    
    Args:
        assignments: Frame assignments
        corpus_index: Corpus metadata index
        use_english_names: If True, try to use name_english from Manifesto API
        max_length: Maximum length for party names
        
    Returns:
        Tuple of:
        - Dict mapping country -> party_id -> score
        - Dict mapping party_id -> party_name
        - Dict mapping party_id -> party_color (or None)
    """
    doc_scores: Dict[str, Tuple[float, float]] = {}
    
    for a in assignments:
        doc_id = a.metadata.get("doc_id", "")
        weight = len(a.passage_text)
        probs = a.probabilities
        frame_sum = sum(probs.values()) * weight
        
        if doc_id not in doc_scores:
            doc_scores[doc_id] = (0.0, 0.0)
        current = doc_scores[doc_id]
        doc_scores[doc_id] = (current[0] + frame_sum, current[1] + weight)
    
    # Aggregate by country and party_id
    country_party_scores: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    party_ids_seen: set = set()
    
    for doc_id, (frame_sum, total_weight) in doc_scores.items():
        meta = corpus_index.get(doc_id, {}).get("extra", {})
        country = meta.get("country_name", "Unknown")
        party_id = meta.get("party_id", "Unknown")
        
        if party_id and party_id != "Unknown":
            score = frame_sum / total_weight if total_weight > 0 else 0.0
            country_party_scores[country][party_id] += score
            party_ids_seen.add(party_id)
    
    family_map = _build_party_family_map()

    # Create party_id -> name and party_id -> color maps
    party_names: Dict[str, str] = {}
    party_colors: Dict[str, Optional[str]] = {}
    
    # Build a mapping of party_id -> doc_id for get_party_name fallback
    party_id_to_doc_id: Dict[str, str] = {}
    for doc_id, (_, _) in doc_scores.items():
        meta = corpus_index.get(doc_id, {}).get("extra", {})
        party_id = meta.get("party_id", "Unknown")
        if party_id and party_id != "Unknown" and party_id not in party_id_to_doc_id:
            party_id_to_doc_id[party_id] = doc_id
    
    for party_id in party_ids_seen:
        # Get party name
        party_name = get_party_name(
            party_id=party_id,
            use_english_name=use_english_names,
            max_length=max_length
        )
        party_names[party_id] = party_name
        
        # Map to political family color
        family = family_map.get(party_id, "unknown")
        party_colors[party_id] = POLITICAL_FAMILY_COLORS.get(family, POLITICAL_FAMILY_COLORS["unknown"])
    
    return dict(country_party_scores), party_names, party_colors


def create_party_bar_charts(
    country_party_scores: Dict[str, Dict[str, float]],
    output_path: Optional[Path] = None,
    figsize_per_country: Tuple[float, float] = (5, 3),
    max_cols: int = 5,
    bar_height: float = 0.5,
    party_names: Optional[Dict[str, str]] = None,
    party_colors: Optional[Dict[str, Optional[str]]] = None,
) -> None:
    """Create bar charts showing frame weight per party, one subplot per country.
    
    Args:
        country_party_scores: Dict mapping country -> party_id -> score
        party_names: Dict mapping party_id -> party_name
        party_colors: Dict mapping party_id -> color (or None)
    """
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Open Sans', 'Helvetica Neue', 'Arial', 'sans-serif']
    
    countries_with_data = {
        country: parties
        for country, parties in country_party_scores.items()
        if any(s > 0 for s in parties.values())
    }
    
    if not countries_with_data:
        print("⚠️ No countries with non-zero scores found.")
        return
    
    sorted_countries = sorted(
        countries_with_data.keys(),
        key=lambda c: sum(countries_with_data[c].values()),
        reverse=True
    )
    
    max_parties = max(len(p) for p in countries_with_data.values())
    n_countries = len(sorted_countries)
    n_cols = min(max_cols, n_countries)
    n_rows = (n_countries + n_cols - 1) // n_cols
    
    # Make charts bigger
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per_country[0] * n_cols * 1.2, figsize_per_country[1] * n_rows * 1.2))
    
    if n_countries == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    cmap = plt.cm.Greens
    global_max = max(max(p.values()) for p in countries_with_data.values())
    
    # Very light background color for panels
    panel_bg_color = (0.98, 0.98, 0.99)  # Very slightly blue-tinted grey
    
    for idx, country in enumerate(sorted_countries):
        ax = axes[idx]
        parties = countries_with_data[country]  # party_id -> score
        sorted_parties = sorted(parties.items(), key=lambda x: x[1], reverse=True)
        
        # Get party names and colors
        party_name_list = []
        scores = []
        colors = []
        for party_id, score in sorted_parties:
            # Get party name
            if party_names and party_id in party_names:
                party_name = party_names[party_id]
            else:
                party_name = party_id  # Fallback to party_id
            party_name_list.append(party_name)
            scores.append(score)
            
            # Get party color
            if party_colors and party_id in party_colors:
                party_color = party_colors[party_id]
            else:
                party_color = None
            
            if party_color:
                colors.append(party_color)
            else:
                # Fallback to gradient based on score
                colors.append(cmap(0.3 + 0.6 * (score / global_max)))
        
        # Set panel background color
        ax.set_facecolor(panel_bg_color)
        
        y_pos = np.arange(len(party_name_list))
        ax.barh(y_pos, scores, height=bar_height, color=colors, edgecolor=colors, linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(party_name_list, fontsize=7)  # Reduced from 9
        ax.invert_yaxis()
        ax.set_title(country, fontsize=10, fontweight='bold', pad=5)  # Reduced from 12
        ax.set_ylim(max_parties - 0.5, -0.5)
        
        # Style the axes
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#888888')  # Grey out bottom spine
        ax.tick_params(left=False)
        ax.tick_params(bottom=True, colors='#888888')  # Grey out x-axis ticks and labels
        ax.xaxis.label.set_color('#888888')  # Grey out x-axis label if present
        
        # Grey out grid lines
        ax.grid(axis='x', color='#e0e0e0', linestyle='-', linewidth=0.5, alpha=0.5)
    
    for idx in range(n_countries, len(axes)):
        axes[idx].set_visible(False)
    
    # Add family color legend at the bottom
    if party_colors:
        # Collect unique families that appear in the data
        family_map = _build_party_family_map()
        families_in_data = set()
        family_to_color = {}
        
        for country, parties in country_party_scores.items():
            for party_id in parties.keys():
                if party_id in party_colors and party_colors[party_id]:
                    family = family_map.get(party_id, "unknown")
                    families_in_data.add(family)
                    family_to_color[family] = party_colors[party_id]
        
        # Create legend entries
        if families_in_data:
            # Sort families for consistent ordering
            sorted_families = sorted(families_in_data)
            legend_elements = []
            for family in sorted_families:
                color = family_to_color.get(family, POLITICAL_FAMILY_COLORS.get(family, "#9e9e9e"))
                # Capitalize first letter of family name
                label = family.replace("-", " ").title()
                legend_elements.append(mpatches.Patch(facecolor=color, edgecolor='none', label=label))
            
            # Add legend at the bottom of the figure on one row
            fig.legend(
                handles=legend_elements,
                loc='lower center',
                bbox_to_anchor=(0.5, -0.01),  # Reduced margin at top of legend
                ncol=len(legend_elements),  # All items in one row
                frameon=False,
                fontsize=10,
                columnspacing=1.5,
                handlelength=1.5,
                handletextpad=0.5
            )
            
            # Adjust layout with more spacing between charts and reduced bottom margin for legend
            plt.subplots_adjust(hspace=0.8, wspace=0.7, bottom=0.06)
            plt.tight_layout(rect=[0, 0.06, 1, 1])
        else:
            plt.tight_layout()
    else:
        plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fmt = output_path.suffix.lower().lstrip(".")
        if fmt not in ("png", "svg", "pdf", "jpg"):
            fmt = "png"
        plt.savefig(output_path, format=fmt, dpi=300 if fmt in ("png", "jpg") else None,
                    bbox_inches="tight", facecolor="white", edgecolor="none")
        print(f"✅ Saved chart to {output_path}")
    else:
        plt.show()
    plt.close()


def create_all_parties_chart(
    country_party_scores: Dict[str, Dict[str, float]],
    output_path: Optional[Path] = None,
    top_n: int = 30,
    party_names: Optional[Dict[str, str]] = None,
    party_colors: Optional[Dict[str, Optional[str]]] = None,
) -> None:
    """Create a single horizontal bar chart with all parties colored by party color.
    
    Args:
        country_party_scores: Dict mapping country -> party_id -> score
        party_names: Dict mapping party_id -> party_name
        party_colors: Dict mapping party_id -> color (or None)
    """
    all_parties = []
    for country, parties in country_party_scores.items():
        for party_id, score in parties.items():
            if score > 0:
                all_parties.append((party_id, country, score))
    
    all_parties.sort(key=lambda x: x[2], reverse=True)
    all_parties = all_parties[:top_n]
    
    if not all_parties:
        print("⚠️ No parties with non-zero scores found.")
        return
    
    all_parties = all_parties[::-1]
    labels = []
    scores = []
    colors = []
    
    # Collect families for legend
    family_map = _build_party_family_map()
    families_in_data = {}
    
    for party_id, country, score in all_parties:
        flag = COUNTRY_FLAGS.get(country, "")
        
        # Get party name
        if party_names and party_id in party_names:
            party_name = party_names[party_id]
        else:
            party_name = party_id  # Fallback to party_id
        short_name = party_name[:20] + "…" if len(party_name) > 20 else party_name
        labels.append(f"{flag} {short_name}")
        scores.append(score)
        
        # Get party color if available, otherwise use country color
        if party_colors and party_id in party_colors:
            party_color = party_colors[party_id]
        else:
            party_color = None
        
        if party_color:
            colors.append(party_color)
            # Track family for legend
            family = family_map.get(party_id, "other")
            if family not in families_in_data:
                families_in_data[family] = party_color
        else:
            colors.append(COUNTRY_COLORS.get(country, "#888888"))
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=scores, orientation='h',
        marker=dict(color=colors, line=dict(color=colors, width=1)),
        textposition='none',
        showlegend=False,
    ))
    
    # Add invisible traces for legend (one per family)
    if families_in_data:
        sorted_families = sorted(families_in_data.keys())
        for family in sorted_families:
            family_color = families_in_data[family]
            family_label = family.replace("-", " ").title()
            fig.add_trace(go.Bar(
                x=[None],
                y=[None],
                marker_color=family_color,
                name=family_label,
                showlegend=True,
                legendgroup=family_label,
            ))
    
    # Configure layout with horizontal legend on one row
    legend_config = {}
    if families_in_data:
        legend_config = {
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.15,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 9},
            "itemwidth": 30,
            "tracegroupgap": 5,
        }
    
    fig.update_layout(
        font=dict(family="Open Sans, Helvetica Neue, Arial, sans-serif", size=11),
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=10, r=10, t=10, b=50 if families_in_data else 10),
        height=max(400, len(all_parties) * 25), width=800,
        xaxis=dict(showgrid=True, gridcolor='#eee', zeroline=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=11)),
        showlegend=bool(families_in_data),
        legend=legend_config if families_in_data else None,
    )
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fmt = output_path.suffix.lower().lstrip(".")
        if fmt == "html":
            fig.write_html(str(output_path))
        else:
            fig.write_image(str(output_path), scale=2)
        print(f"✅ Saved chart to {output_path}")
    else:
        fig.show()


@register_plotter
class PartyBarsPlotter(BasePlotter):
    """Generate bar charts showing frame weight per party, by country."""
    
    name = "party_bars"
    
    def plot(self, config: PlotConfig) -> Optional[Path]:
        corpus_index = self.corpus_index or {}
        if not corpus_index:
            corpus_index = load_corpus_index(self.results_dir)
        
        use_english_names = config.get_option("use_english_names", False)
        max_length = config.get_option("max_length", 25)
        
        country_party_scores, party_names, party_colors = aggregate_by_party(
            self.state.assignments, 
            corpus_index,
            use_english_names=use_english_names,
            max_length=max_length,
        )
        output_path = self.get_output_path(config, "party_bars.png")
        create_party_bar_charts(
            country_party_scores, 
            output_path=output_path, 
            party_names=party_names,
            party_colors=party_colors
        )
        return output_path


def create_interactive_party_bars_html(
    country_party_scores: Dict[str, Dict[str, float]],
    output_path: Optional[Path] = None,
    use_english_names: bool = False,
    embeddable: bool = False,
    party_names: Optional[Dict[str, str]] = None,
    party_colors: Optional[Dict[str, Optional[str]]] = None,
) -> str:
    """Create an interactive HTML with dropdown to select countries.
    
    Args:
        country_party_scores: Dict mapping country -> party -> score
        output_path: Optional path to save HTML file
        use_english_names: Whether to use English party names
        
    Returns:
        HTML content as string
    """
    # Filter countries with data
    countries_with_data = {
        country: parties
        for country, parties in country_party_scores.items()
        if any(s > 0 for s in parties.values())
    }
    
    if not countries_with_data:
        return "<p>No party data available.</p>"
    
    sorted_countries = sorted(countries_with_data.keys())
    
    # Prepare data for each country
    country_data = {}
    global_max = 0.0
    
    for country, parties in countries_with_data.items():
        sorted_parties = sorted(parties.items(), key=lambda x: x[1], reverse=True)
        # parties is party_id -> score
        party_id_list = [p[0] for p in sorted_parties]
        scores = [p[1] for p in sorted_parties]
        
        # Get party names and colors
        party_name_list = []
        party_colors_list = []
        for party_id, score in sorted_parties:
            # Get party name
            if party_names and party_id in party_names:
                party_name = party_names[party_id]
            else:
                party_name = party_id  # Fallback to party_id
            party_name_list.append(party_name)
            
            # Get party color
            if party_colors and party_id in party_colors:
                party_color = party_colors[party_id]
            else:
                party_color = None
            party_colors_list.append(party_color)
        
        country_data[country] = {
            "parties": party_name_list,
            "scores": scores,
            "colors": party_colors_list,
        }
        if scores:
            global_max = max(global_max, max(scores))
    
    # Create HTML with Plotly
    country_options = "".join([
        f'<option value="{html.escape(country)}">{html.escape(country)}</option>'
        for country in sorted_countries
    ])
    
    # Prepare JavaScript data
    js_data = {}
    for country, data in country_data.items():
        js_data[country] = {
            "parties": data["parties"],
            "scores": data["scores"],
            "colors": data.get("colors", []),
        }
    
    # Create HTML content - either standalone or embeddable
    if embeddable:
        # Embeddable version: just the content without full HTML structure
        html_content = f"""<div class="interactive-party-bars-container">
    <div class="controls">
        <label for="country-select">Select Country:</label>
        <select id="country-select" onchange="updateChart()">
            {country_options}
        </select>
    </div>
    <div id="chart"></div>
</div>

<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
    .interactive-party-bars-container {{
        font-family: 'Open Sans', 'Helvetica Neue', Arial, sans-serif;
        margin: 20px 0;
    }}
    .interactive-party-bars-container .controls {{
        margin-bottom: 20px;
    }}
    .interactive-party-bars-container label {{
        display: block;
        margin-bottom: 8px;
        font-weight: 600;
        color: #555;
    }}
    .interactive-party-bars-container select {{
        padding: 8px 12px;
        font-size: 14px;
        border: 1px solid #ddd;
        border-radius: 4px;
        background: white;
        min-width: 200px;
    }}
    #chart {{
        margin-top: 20px;
    }}
</style>
<script>
    (function() {{
        const countryData = {json.dumps(js_data)};
        const globalMax = {global_max};
        
        function updateChart() {{
            const select = document.getElementById('country-select');
            if (!select) return;
            const country = select.value;
            const data = countryData[country];
            
            if (!data) {{
                return;
            }}
            
            const parties = data.parties;
            const scores = data.scores;
            const partyColors = data.colors || [];
            
            // Use party colors if available, otherwise create gradient based on scores
            const colors = scores.map((s, i) => {{
                if (partyColors[i]) {{
                    return partyColors[i];
                }}
                // Fallback to gradient
                const intensity = 0.3 + 0.6 * (s / globalMax);
                const r = Math.floor(34 + intensity * 221);
                const g = Math.floor(139 + intensity * 116);
                const b = Math.floor(34 + intensity * 221);
                return `rgb(${{r}}, ${{g}}, ${{b}})`;
            }});
            
            const trace = {{
                x: scores,
                y: parties,
                type: 'bar',
                orientation: 'h',
                marker: {{
                    color: colors,
                    line: {{
                        color: colors,
                        width: 1
                    }}
                }},
                text: scores.map(s => s.toFixed(3)),
                textposition: 'outside',
                hovertemplate: '<b>%{{y}}</b><br>Score: %{{x:.3f}}<extra></extra>'
            }};
            
            const layout = {{
                title: {{
                    text: `Frame Scores - ${{country}}`,
                    font: {{ size: 18 }}
                }},
                xaxis: {{
                    title: 'Frame Score',
                    showgrid: true,
                    gridcolor: '#eee'
                }},
                yaxis: {{
                    title: 'Party',
                    autorange: 'reversed',
                    showgrid: false
                }},
                margin: {{ l: 200, r: 50, t: 60, b: 50 }},
                height: Math.max(400, parties.length * 40),
                plot_bgcolor: 'white',
                paper_bgcolor: 'white'
            }};
            
            Plotly.newPlot('chart', [trace], layout, {{ responsive: true }});
        }}
        
        // Initialize with first country when DOM is ready
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', updateChart);
        }} else {{
            updateChart();
        }}
    }})();
</script>"""
    else:
        # Standalone version: full HTML document
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Party Frame Scores by Country</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Open Sans', 'Helvetica Neue', Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            margin-top: 0;
            color: #333;
        }}
        .controls {{
            margin-bottom: 20px;
        }}
        label {{
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #555;
        }}
        select {{
            padding: 8px 12px;
            font-size: 14px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background: white;
            min-width: 200px;
        }}
        #chart {{
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Party Frame Scores by Country</h1>
        <div class="controls">
            <label for="country-select">Select Country:</label>
            <select id="country-select" onchange="updateChart()">
                {country_options}
            </select>
        </div>
        <div id="chart"></div>
    </div>
    
    <script>
        const countryData = {json.dumps(js_data)};
        const globalMax = {global_max};
        
        function updateChart() {{
            const select = document.getElementById('country-select');
            const country = select.value;
            const data = countryData[country];
            
            if (!data) {{
                return;
            }}
            
            const parties = data.parties;
            const scores = data.scores;
            const partyColors = data.colors || [];
            
            // Use party colors if available, otherwise create gradient based on scores
            const colors = scores.map((s, i) => {{
                if (partyColors[i]) {{
                    return partyColors[i];
                }}
                // Fallback to gradient
                const intensity = 0.3 + 0.6 * (s / globalMax);
                const r = Math.floor(34 + intensity * 221);
                const g = Math.floor(139 + intensity * 116);
                const b = Math.floor(34 + intensity * 221);
                return `rgb(${{r}}, ${{g}}, ${{b}})`;
            }});
            
            const trace = {{
                x: scores,
                y: parties,
                type: 'bar',
                orientation: 'h',
                marker: {{
                    color: colors,
                    line: {{
                        color: colors,
                        width: 1
                    }}
                }},
                text: scores.map(s => s.toFixed(3)),
                textposition: 'outside',
                hovertemplate: '<b>%{{y}}</b><br>Score: %{{x:.3f}}<extra></extra>'
            }};
            
            const layout = {{
                title: {{
                    text: `Frame Scores - ${{country}}`,
                    font: {{ size: 18 }}
                }},
                xaxis: {{
                    title: 'Frame Score',
                    showgrid: true,
                    gridcolor: '#eee'
                }},
                yaxis: {{
                    title: 'Party',
                    autorange: 'reversed',
                    showgrid: false
                }},
                margin: {{ l: 200, r: 50, t: 60, b: 50 }},
                height: Math.max(400, parties.length * 40),
                plot_bgcolor: 'white',
                paper_bgcolor: 'white'
            }};
            
            Plotly.newPlot('chart', [trace], layout, {{ responsive: true }});
        }}
        
        // Initialize with first country
        updateChart();
    </script>
</body>
</html>"""
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_content, encoding="utf-8")
        print(f"✅ Saved interactive party bars HTML to {output_path}")
    
    return html_content


@register_plotter
class AllPartiesPlotter(BasePlotter):
    """Generate single bar chart with all parties colored by country."""
    
    name = "all_parties"
    
    def plot(self, config: PlotConfig) -> Optional[Path]:
        corpus_index = self.corpus_index or {}
        if not corpus_index:
            corpus_index = load_corpus_index(self.results_dir)
        
        use_english_names = config.get_option("use_english_names", False)
        max_length = config.get_option("max_length", 25)
        
        country_party_scores, party_names, party_colors = aggregate_by_party(
            self.state.assignments, 
            corpus_index,
            use_english_names=use_english_names,
            max_length=max_length,
        )
        output_path = self.get_output_path(config, "all_parties.png")
        create_all_parties_chart(
            country_party_scores, 
            output_path=output_path, 
            party_names=party_names,
            party_colors=party_colors
        )
        return output_path


@register_plotter
class InteractivePartyBarsPlotter(BasePlotter):
    """Generate interactive HTML with dropdown to select countries."""
    
    name = "interactive_party_bars"
    
    def plot(self, config: PlotConfig) -> Optional[Path]:
        corpus_index = self.corpus_index or {}
        if not corpus_index:
            corpus_index = load_corpus_index(self.results_dir)
        
        use_english_names = config.get_option("use_english_names", False)
        max_length = config.get_option("max_length", 25)
        
        country_party_scores, party_names, party_colors = aggregate_by_party(
            self.state.assignments, 
            corpus_index,
            use_english_names=use_english_names,
            max_length=max_length,
        )
        
        output_path = self.get_output_path(config, "interactive_party_bars.html")
        create_interactive_party_bars_html(
            country_party_scores,
            output_path=output_path,
            use_english_names=use_english_names,
            party_names=party_names,
            party_colors=party_colors
        )
        return output_path
