"""Plotting utilities for narrative framing results.

This module provides a registry of plotters that can be configured
via the `additional_plots` section in config files.

Usage in config:
    additional_plots:
      - type: europe_map
        output: plots/europe_map.png
        options:
          cmap: Greens
      - type: dominant_frame_map
      - type: party_bars
        options:
          mode: by_country
      - type: all_parties
      - type: frame_examples
        options:
          format: html

Available plotters:
    - europe_map: Choropleth map of Europe by frame scores
    - dominant_frame_map: Categorical map showing dominant frame per country
    - party_bars: Bar charts of frame weight per party, by country
    - all_parties: Single bar chart with all parties colored by country
    - frame_examples: Table of top example chunks per frame
"""

from .base import BasePlotter, PlotConfig
from .registry import (
    register_plotter,
    get_plotter,
    get_available_plotters,
    run_plots,
)

# Import plotters to trigger registration
from . import europe_map
from . import party_bars
from . import frame_examples

__all__ = [
    "BasePlotter",
    "PlotConfig",
    "register_plotter",
    "get_plotter",
    "get_available_plotters",
    "run_plots",
]

