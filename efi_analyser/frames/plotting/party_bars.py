"""Party bar chart plotters for narrative framing results."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from .base import BasePlotter, PlotConfig
from .registry import register_plotter
from ._utils import load_corpus_index


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

PARTY_NAME_OVERRIDES = {
    "Academician Muamer Zukorlic - Straight ahead - Party of Justice and Reconciliation (SPP) - Democratic Party of Macedonians (DPM)": "Zukorlic Coalition",
    "National Alliance 'All For Latvia!' – 'For Fatherland and Freedom - Latvian National Independence Movement'": "NA/TB-LNNK",
    "Coalition of South Tyrolean People's Party and Trentino Tyrolean Autonomist Party": "SVP-PATT",
    "Ivica Dačić –  Socialist Party of Serbia, United Serbia – Dragan Markovic Palma": "SPS/JS Coalition",
    "Alliance of Federation of Young Democrats - Christian Democratic People's Party": "FiDeSz-KDNP",
    "Serbian Movement Dveri - Movement for the Restoration of the Kingdom of Seria": "Dveri-POKS",
    "Together 2014 -Dialogue for Hungary Electoral Alliance": "E14-PM",
    "Alliance for a Better Future of Bosnia and Herzegovina": "SBB BiH",
    "The People's European Union of Bosnia and Herzegovina": "NES",
    "Party of Democratic Progress of the Republika Srpska": "PDP RS",
    "Croatian Democratic Union of Bosnia and Herzegovina": "HDZ BiH",
    "Christian Democratic People's Party of Switzerland": "CVP/PDC",
    "The Alliance - Social Democratic Party of Iceland": "S (Alliance)",
    "Social Democratic Party of Bosnia and Herzegovina": "SDP BiH",
    "Christian Democratic Union/Christian Social Union": "CDU/CSU",
    "Party with a First and Last Name - Smart - Focus": "SSIP-P-F",
    "Sahra Wagenknecht Alliance – Reason and Justice": "BSW",
    "Homeland Union - Lithuanian Christian Democrats": "TS-LKD",
    "Party of Socialists of the Republic of Moldova": "PSRM",
    "Conservative Democratic Party of Switzerland": "BDP/PBD",
    "New Ecologic and Social People's Union": "NUPES",
    "Our Homeland Movement": "Mi Hazánk",
    "Family of the Irish": "Aontú",
}


def shorten_party_name(name: str, max_len: int = 25) -> str:
    """Shorten party name using overrides or truncation."""
    if name in PARTY_NAME_OVERRIDES:
        return PARTY_NAME_OVERRIDES[name]
    if len(name) > max_len:
        return name[:max_len-2] + "…"
    return name


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


def aggregate_by_party(
    assignments: List[dict],
    corpus_index: Dict[str, dict],
) -> Dict[str, Dict[str, float]]:
    """Aggregate frame assignments by party within each country."""
    doc_scores: Dict[str, Tuple[float, float]] = {}
    
    for a in assignments:
        passage_id = a.get("passage_id", "")
        doc_id = passage_id.split(":")[0] if ":" in passage_id else passage_id
        text = a.get("passage_text", "")
        weight = len(text)
        probs = a.get("probabilities", {})
        frame_sum = sum(probs.values()) * weight
        
        if doc_id not in doc_scores:
            doc_scores[doc_id] = (0.0, 0.0)
        current = doc_scores[doc_id]
        doc_scores[doc_id] = (current[0] + frame_sum, current[1] + weight)
    
    country_party_scores: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    
    for doc_id, (frame_sum, total_weight) in doc_scores.items():
        meta = corpus_index.get(doc_id, {})
        country = meta.get("country_name", "Unknown")
        party = meta.get("party_abbrev") or meta.get("party_name", "Unknown")
        score = frame_sum / total_weight if total_weight > 0 else 0.0
        country_party_scores[country][party] += score
    
    return dict(country_party_scores)


def create_party_bar_charts(
    country_party_scores: Dict[str, Dict[str, float]],
    output_path: Optional[Path] = None,
    figsize_per_country: Tuple[float, float] = (5, 3),
    max_cols: int = 3,
    bar_height: float = 0.5,
) -> None:
    """Create bar charts showing frame weight per party, one subplot per country."""
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
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per_country[0] * n_cols, figsize_per_country[1] * n_rows))
    
    if n_countries == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    cmap = plt.cm.Greens
    global_max = max(max(p.values()) for p in countries_with_data.values())
    
    for idx, country in enumerate(sorted_countries):
        ax = axes[idx]
        parties = countries_with_data[country]
        sorted_parties = sorted(parties.items(), key=lambda x: x[1], reverse=True)
        party_names = [shorten_party_name(p[0]) for p in sorted_parties]
        scores = [p[1] for p in sorted_parties]
        colors = [cmap(0.3 + 0.6 * (s / global_max)) for s in scores]
        
        y_pos = np.arange(len(party_names))
        ax.barh(y_pos, scores, height=bar_height, color=colors, edgecolor=colors, linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(party_names, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(country, fontsize=11, fontweight='bold', pad=5)
        ax.set_ylim(max_parties - 0.5, -0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(left=False)
    
    for idx in range(n_countries, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fmt = output_path.suffix.lower().lstrip(".")
        if fmt not in ("png", "svg", "pdf", "jpg"):
            fmt = "png"
        plt.savefig(output_path, format=fmt, dpi=150 if fmt in ("png", "jpg") else None,
                    bbox_inches="tight", facecolor="white", edgecolor="none")
        print(f"✅ Saved chart to {output_path}")
    else:
        plt.show()
    plt.close()


def create_all_parties_chart(
    country_party_scores: Dict[str, Dict[str, float]],
    output_path: Optional[Path] = None,
    top_n: int = 30,
) -> None:
    """Create a single horizontal bar chart with all parties colored by country."""
    all_parties = []
    for country, parties in country_party_scores.items():
        for party, score in parties.items():
            if score > 0:
                all_parties.append((party, country, score))
    
    all_parties.sort(key=lambda x: x[2], reverse=True)
    all_parties = all_parties[:top_n]
    
    if not all_parties:
        print("⚠️ No parties with non-zero scores found.")
        return
    
    all_parties = all_parties[::-1]
    labels = []
    scores = []
    colors = []
    
    for party, country, score in all_parties:
        flag = COUNTRY_FLAGS.get(country, "")
        short_name = shorten_party_name(party, max_len=20)
        labels.append(f"{flag} {short_name}")
        scores.append(score)
        colors.append(COUNTRY_COLORS.get(country, "#888888"))
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=scores, orientation='h',
        marker=dict(color=colors, line=dict(color=colors, width=1)),
        textposition='none',
    ))
    
    fig.update_layout(
        font=dict(family="Open Sans, Helvetica Neue, Arial, sans-serif", size=11),
        plot_bgcolor='white', paper_bgcolor='white',
        margin=dict(l=10, r=10, t=10, b=10),
        height=max(400, len(all_parties) * 25), width=800,
        xaxis=dict(showgrid=True, gridcolor='#eee', zeroline=False),
        yaxis=dict(showgrid=False, tickfont=dict(size=11)),
        showlegend=False,
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
        if self.state.assignments:
            assignments = _assignments_to_dicts(self.state.assignments)
        else:
            import json
            assignments_path = self.results_dir / "frame_assignments.json"
            if assignments_path.exists():
                assignments = json.loads(assignments_path.read_text())
            else:
                print(f"⚠️ No assignments found in state or at {assignments_path}")
                return None
        
        corpus_index = self.corpus_index or {}
        if not corpus_index:
            corpus_index = load_corpus_index(self.results_dir)
        
        country_party_scores = aggregate_by_party(assignments, corpus_index)
        output_path = self.get_output_path(config, "party_bars.png")
        create_party_bar_charts(country_party_scores, output_path=output_path)
        return output_path


@register_plotter
class AllPartiesPlotter(BasePlotter):
    """Generate single bar chart with all parties colored by country."""
    
    name = "all_parties"
    
    def plot(self, config: PlotConfig) -> Optional[Path]:
        if self.state.assignments:
            assignments = _assignments_to_dicts(self.state.assignments)
        else:
            import json
            assignments_path = self.results_dir / "frame_assignments.json"
            if assignments_path.exists():
                assignments = json.loads(assignments_path.read_text())
            else:
                print(f"⚠️ No assignments found in state or at {assignments_path}")
                return None
        
        corpus_index = self.corpus_index or {}
        if not corpus_index:
            corpus_index = load_corpus_index(self.results_dir)
        
        country_party_scores = aggregate_by_party(assignments, corpus_index)
        output_path = self.get_output_path(config, "all_parties.png")
        create_all_parties_chart(country_party_scores, output_path=output_path)
        return output_path
