"""
Base report generator for all analysis results.
"""

import json
import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from ..types import ReportConfig


class BaseReportGenerator(ABC):
    """Base class for all report generators."""
    
    def __init__(self, config: ReportConfig):
        self.config = config
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def generate_csv(self, results: Any, output_path: Optional[Path] = None) -> Path:
        """Generate CSV report from results."""
        pass
    
    @abstractmethod
    def generate_json(self, results: Any, output_path: Optional[Path] = None) -> Path:
        """Generate JSON report from results."""
        pass
    
    @abstractmethod
    def generate_html(self, results: Any, output_path: Optional[Path] = None) -> Path:
        """Generate HTML report from results."""
        pass
    
    def generate_pdf(self, results: Any, output_path: Optional[Path] = None) -> Path:
        """Generate PDF report from HTML (optional)."""
        try:
            import weasyprint
            html_path = self.generate_html(results)
            if output_path is None:
                output_path = self.output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            
            weasyprint.HTML(filename=str(html_path)).write_pdf(str(output_path))
            return output_path
        except ImportError:
            raise ImportError("weasyprint not available for PDF generation")
    
    def generate_all(self, results: Any) -> Dict[str, Path]:
        """Generate all configured report formats."""
        report_paths = {}
        
        if "csv" in self.config.formats:
            report_paths['csv'] = self.generate_csv(results)
        
        if "json" in self.config.formats:
            report_paths['json'] = self.generate_json(results)
        
        if "html" in self.config.formats:
            report_paths['html'] = self.generate_html(results)
        
        if "pdf" in self.config.formats:
            try:
                report_paths['pdf'] = self.generate_pdf(results)
            except ImportError:
                # PDF generation failed, but continue with other formats
                pass
        
        return report_paths
    
    def _get_default_output_path(self, format_name: str, base_name: str = "report") -> Path:
        """Generate default output path for a report format."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return self.output_dir / f"{base_name}_{timestamp}.{format_name}"
    
    def _ensure_output_dir(self, output_path: Path) -> None:
        """Ensure output directory exists."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
