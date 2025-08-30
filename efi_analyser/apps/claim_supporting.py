"""
Claim Supporting App

High-level interface for analyzing claims against a corpus.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..types import ClaimSupportingConfig, ClaimSupportingResults, ReportConfig
from ..pipeline.claim_supporting import ClaimSupportingPipeline
from ..report_generator.claim_supporting import ClaimSupportingReportGenerator

logger = logging.getLogger(__name__)


class ClaimSupportingApp:
    """High-level application interface for claim supporting analysis."""
    
    def __init__(self, config: ClaimSupportingConfig):
        """
        Initialize the Claim Supporting App.
        
        Args:
            config: Configuration for the claim supporting pipeline
        """
        self.config = config
        self.pipeline = ClaimSupportingPipeline(config)
        
        # Create report config
        report_config = ReportConfig(
            output_dir=config.workspace_path / "reports" / "claim_supporting",
            formats=config.output_formats
        )
        self.report_generator = ClaimSupportingReportGenerator(report_config)
    
    def analyze_claims(self, claims: List[str]) -> List[ClaimSupportingResults]:
        """
        Analyze claims against the corpus.
        
        Args:
            claims: List of claims to analyze
            
        Returns:
            List of ClaimSupportingResults for each claim
        """
        logger.info(f"Starting claim supporting analysis for {len(claims)} claims")
        
        try:
            results = self.pipeline.run(claims)
            logger.info(f"Successfully analyzed {len(results)} claims")
            return results
            
        except Exception as e:
            logger.error(f"Failed to analyze claims: {e}")
            raise
    
    def generate_reports(self, results: List[ClaimSupportingResults], output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """
        Generate reports from analysis results.
        
        Args:
            results: List of ClaimSupportingResults
            output_dir: Optional custom output directory
            
        Returns:
            Dictionary mapping report type to file path
        """
        if output_dir:
            # Update report generator config
            report_config = ReportConfig(
                output_dir=output_dir,
                formats=self.config.output_formats
            )
            self.report_generator = ClaimSupportingReportGenerator(report_config)
        
        logger.info(f"Generating reports for {len(results)} claim results")
        
        try:
            report_paths = self.report_generator.generate_all(results)
            logger.info(f"Generated {len(report_paths)} reports")
            return report_paths
            
        except Exception as e:
            logger.error(f"Failed to generate reports: {e}")
            raise
    
    def analyze_claims_with_reports(self, claims: List[str], output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Analyze claims and generate reports in one operation.
        
        Args:
            claims: List of claims to analyze
            output_dir: Optional custom output directory
            
        Returns:
            Dictionary containing results and report paths
        """
        # Analyze claims
        results = self.analyze_claims(claims)
        
        # Generate reports
        report_paths = self.generate_reports(results, output_dir)
        
        return {
            'results': results,
            'reports': report_paths,
            'summary': {
                'total_claims': len(results),
                'total_results': sum(len(r.results) for r in results),
                'entailment_count': sum(r.entailment_count for r in results),
                'neutral_count': sum(r.neutral_count for r in results),
                'contradiction_count': sum(r.contradiction_count for r in results)
            }
        }
