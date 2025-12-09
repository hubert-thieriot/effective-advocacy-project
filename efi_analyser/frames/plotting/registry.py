"""Plotter registry for discovering and instantiating plotters."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Type, TYPE_CHECKING

from .base import BasePlotter, PlotConfig

if TYPE_CHECKING:
    from apps.narrative_framing.run import WorkflowState
    from apps.narrative_framing.config import NarrativeFramingConfig


# Global registry of plotter classes
_PLOTTER_REGISTRY: Dict[str, Type[BasePlotter]] = {}


def register_plotter(cls: Type[BasePlotter]) -> Type[BasePlotter]:
    """Decorator to register a plotter class.
    
    Usage:
        @register_plotter
        class MyPlotter(BasePlotter):
            name = "my_plot"
            ...
    """
    if not hasattr(cls, 'name') or not cls.name:
        raise ValueError(f"Plotter class {cls.__name__} must define a 'name' attribute")
    _PLOTTER_REGISTRY[cls.name] = cls
    return cls


def get_plotter(
    name: str,
    state: "WorkflowState",
    config: "NarrativeFramingConfig",
    results_dir: Path,
    corpus_index: Optional[Dict[str, dict]] = None,
    export_dir: Optional[Path] = None,
    export_plots_dir: Optional[Path] = None,
) -> Optional[BasePlotter]:
    """Get an instantiated plotter by name.
    
    Args:
        name: Plotter name (e.g., "europe_map")
        state: Workflow state containing schema, assignments, aggregates, etc.
        config: Configuration for the narrative framing workflow
        results_dir: Directory where results are stored
        corpus_index: Optional corpus index mapping doc_id to metadata
        export_dir: Optional export directory for copying plots
        export_plots_dir: Optional export directory for copying plots (from report.export_plots_dir)
        
    Returns:
        Instantiated plotter, or None if not found
    """
    cls = _PLOTTER_REGISTRY.get(name)
    if cls is None:
        print(f"⚠️ Unknown plotter type: {name}")
        return None
    return cls(state, config, results_dir, corpus_index, export_dir, export_plots_dir)


def get_available_plotters() -> List[str]:
    """Get list of all registered plotter names."""
    return list(_PLOTTER_REGISTRY.keys())


def run_plots(
    state: "WorkflowState",
    config: "NarrativeFramingConfig",
    results_dir: Path,
    plot_configs: List[PlotConfig],
    corpus_index: Optional[Dict[str, dict]] = None,
    export_dir: Optional[Path] = None,
    export_plots_dir: Optional[Path] = None,
) -> List[Path]:
    """Run multiple plots based on configuration.
    
    Args:
        state: Workflow state containing schema, assignments, aggregates, etc.
        config: Configuration for the narrative framing workflow
        results_dir: Directory where results are stored
        plot_configs: List of plot configurations
        corpus_index: Optional corpus index mapping doc_id to metadata
        export_dir: Optional export directory for copying plots
        export_plots_dir: Optional export directory for copying plots (from report.export_plots_dir)
        
    Returns:
        List of paths to generated plots
    """
    generated = []
    
    for plot_config in plot_configs:
        plotter = get_plotter(
            plot_config.type,
            state,
            config,
            results_dir,
            corpus_index,
            export_dir,
            export_plots_dir,
        )
        if plotter is None:
            continue
        
        try:
            output_path = plotter.plot(plot_config)
            if output_path:
                generated.append(output_path)
                print(f"✅ Generated {plot_config.type}: {output_path}")
                # Copy to export dirs if configured
                plotter.copy_to_export(output_path, plot_config)
        except Exception as e:
            print(f"❌ Failed to generate {plot_config.type}: {e}")
    
    return generated

