"""Base class for narrative framing plotters."""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from apps.narrative_framing.run import WorkflowState
    from apps.narrative_framing.config import NarrativeFramingConfig


@dataclass
class PlotConfig:
    """Configuration for a single plot."""
    type: str
    output: Optional[str] = None  # Relative to results_dir/plots/
    options: Optional[Dict[str, Any]] = None
    export_as: Optional[str] = None  # Filename for export_dir (e.g., "manifesto_map.png")
    
    def get_option(self, key: str, default: Any = None) -> Any:
        """Get an option value with default."""
        if self.options is None:
            return default
        return self.options.get(key, default)


class BasePlotter(ABC):
    """Base class for all plotters."""
    
    # Subclasses must define this
    name: str = "base"
    
    def __init__(
        self,
        state: "WorkflowState",
        config: "NarrativeFramingConfig",
        results_dir: Path,
        corpus_index: Optional[Dict[str, dict]] = None,
        export_dir: Optional[Path] = None,
        export_plots_dir: Optional[Path] = None,
    ):
        """Initialize plotter with state and config.
        
        Args:
            state: Workflow state containing schema, assignments, aggregates, etc.
            config: Configuration for the narrative framing workflow
            results_dir: Directory where results are stored
            corpus_index: Optional corpus index mapping doc_id to metadata
            export_dir: Optional export directory for copying plots to docs/assets etc.
            export_plots_dir: Optional export directory for copying plots (from report.export_plots_dir)
        """
        self.state = state
        self.config = config
        self.results_dir = Path(results_dir)
        self.corpus_index = corpus_index or {}
        self.export_dir = export_dir
        self.export_plots_dir = export_plots_dir
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def plot(self, config: PlotConfig) -> Optional[Path]:
        """Generate the plot.
        
        Args:
            config: Plot configuration including output path and options
            
        Returns:
            Path to the generated plot, or None if failed
        """
        pass
    
    def get_output_path(self, config: PlotConfig, default_name: str) -> Path:
        """Get the output path for a plot.
        
        Args:
            config: Plot configuration
            default_name: Default filename if not specified in config
            
        Returns:
            Absolute path for the output file
        """
        if config.output:
            return self.results_dir / config.output
        return self.plots_dir / default_name
    
    def copy_to_export(self, source: Path, config: PlotConfig) -> List[Path]:
        """Copy generated plot to export directories if configured.
        
        Args:
            source: Path to the generated plot
            config: Plot configuration with optional export_as filename
            
        Returns:
            List of paths to exported files (may be empty)
        """
        exported = []
        
        # Use export_as if specified, otherwise use source filename
        filename = config.export_as or source.name
        
        # Export to export_dir if configured
        if self.export_dir:
            dest = Path(self.export_dir) / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            print(f"   📤 Exported to {dest}")
            exported.append(dest)
        
        # Export to export_plots_dir if configured
        if self.export_plots_dir:
            dest = Path(self.export_plots_dir) / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            print(f"   📤 Exported to {dest}")
            exported.append(dest)
        
        return exported

