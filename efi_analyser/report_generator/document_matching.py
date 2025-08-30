"""
Report generator for document matching results.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from .base import BaseReportGenerator
from ..types import DocumentMatchingResults, FindingResults, DocumentMatch, ReportConfig


class DocumentMatchingReportGenerator(BaseReportGenerator):
    """Generates reports from DocumentMatchingResults."""
    
    def __init__(self, config: ReportConfig):
        super().__init__(config)
    
    def generate_csv(self, results: DocumentMatchingResults, output_path: Optional[Path] = None) -> Path:
        """Generate a CSV report with all matches and scores."""
        if output_path is None:
            output_path = self._get_default_output_path("csv", "document_matching")
        
        self._ensure_output_dir(output_path)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            # Determine all rescorer names from the data
            rescorer_names = set()
            for finding_result in results.results_by_finding.values():
                rescorer_names.update(finding_result.rescorer_scores.keys())
            
            # Create CSV headers
            fieldnames = [
                'finding_id', 'finding_text', 'chunk_id', 'chunk_text',
                'cosine_score'
            ] + [f'{name}_score' for name in sorted(rescorer_names)]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write data rows
            for finding_id, finding_result in results.results_by_finding.items():
                for match in finding_result.matches:
                    row = {
                        'finding_id': finding_id,
                        'finding_text': finding_result.finding_text[:200] + '...' if len(finding_result.finding_text) > 200 else finding_result.finding_text,
                        'chunk_id': match.chunk_id,
                        'chunk_text': match.chunk_text[:200] + '...' if len(match.chunk_text) > 200 else match.chunk_text,
                        'cosine_score': match.cosine_score,
                    }
                    
                    # Add rescorer scores
                    for rescorer_name in sorted(rescorer_names):
                        row[f'{rescorer_name}_score'] = match.rescorer_scores.get(rescorer_name, 0.0)
                    
                    writer.writerow(row)
        
        return output_path
    
    def generate_json(self, results: DocumentMatchingResults, output_path: Optional[Path] = None) -> Path:
        """Generate a detailed JSON report."""
        if output_path is None:
            output_path = self._get_default_output_path("json", "document_matching")
        
        self._ensure_output_dir(output_path)
        
        # Convert results to a serializable format
        report_data = {
            'summary': {
                'findings_processed': results.findings_processed,
                'total_matches': results.total_matches,
                'timestamp': results.metadata.get('timestamp', ''),
                'top_k': results.metadata.get('top_k', ''),
                'filters_applied': results.metadata.get('filters_applied', False)
            },
            'findings': {}
        }
        
        for finding_id, finding_result in results.results_by_finding.items():
            report_data['findings'][finding_id] = {
                'finding_text': finding_result.finding_text,
                'matches': [
                    {
                        'chunk_id': match.chunk_id,
                        'chunk_text': match.chunk_text,
                        'cosine_score': match.cosine_score,
                        'rescorer_scores': match.rescorer_scores,
                        'metadata': match.metadata
                    }
                    for match in finding_result.matches
                ],
                'rescorer_scores': finding_result.rescorer_scores,
                'timing': finding_result.timing
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def generate_html(self, results: DocumentMatchingResults, output_path: Optional[Path] = None) -> Path:
        """Generate a comprehensive HTML report."""
        if output_path is None:
            output_path = self._get_default_output_path("html", "document_matching")
        
        self._ensure_output_dir(output_path)
        
        html_content = self._generate_html_content(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
    
    def _generate_html_content(self, results: DocumentMatchingResults) -> str:
        """Generate HTML content for the report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Document Matching Report</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ margin: 20px 0; }}
                .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
                .stat-card {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; text-align: center; }}
                .stat-number {{ font-size: 2em; font-weight: bold; color: #2c5aa0; }}
                .finding-section {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                .finding-text {{ font-weight: bold; font-size: 1.1em; margin-bottom: 15px; }}
                .match-table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                .match-table th, .match-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .match-table th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Document Matching Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Findings processed: {results.findings_processed}</p>
                <p>Total matches: {results.total_matches}</p>
            </div>
            
            <div class="summary">
                <h2>Summary Statistics</h2>
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-number">{results.findings_processed}</div>
                        <div>Findings Processed</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{results.total_matches}</div>
                        <div>Total Matches</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{len(results.results_by_finding)}</div>
                        <div>Findings with Results</div>
                    </div>
                </div>
            </div>
        """
        
        # Add per-finding analysis
        for finding_id, finding_result in results.results_by_finding.items():
            html_content += f"""
            <div class="finding-section">
                <div class="finding-text">Finding: {finding_result.finding_text[:100]}...</div>
                <p>Matches found: {len(finding_result.matches)}</p>
                
                <table class="match-table">
                    <tr>
                        <th>Chunk ID</th>
                        <th>Cosine Score</th>
                        <th>Rescorer Scores</th>
                    </tr>
            """
            
            for match in finding_result.matches:
                rescorer_scores_str = ", ".join([f"{name}: {score:.3f}" for name, score in match.rescorer_scores.items()])
                html_content += f"""
                    <tr>
                        <td>{match.chunk_id}</td>
                        <td>{match.cosine_score:.3f}</td>
                        <td>{rescorer_scores_str}</td>
                    </tr>
                """
            
            html_content += """
                </table>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        return html_content
