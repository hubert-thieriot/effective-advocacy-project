"""
Report Generator for Document Matching Results

Generates various report formats from DocumentMatchingResults objects.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from efi_core.types import DocumentMatchingResults, FindingResults, DocumentMatch


class ReportGenerator:
    """Generates reports from DocumentMatchingResults."""
    
    def __init__(self, results: DocumentMatchingResults):
        self.results = results
    
    def generate_csv_report(self, output_path: Path) -> None:
        """Generate a CSV report with all matches and scores."""
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            # Determine all rescorer names from the data
            rescorer_names = set()
            for finding_result in self.results.results_by_finding.values():
                rescorer_names.update(finding_result.rescorer_scores.keys())
            
            # Create CSV headers
            fieldnames = [
                'finding_id', 'finding_text', 'chunk_id', 'chunk_text',
                'cosine_score'
            ] + [f'{name}_score' for name in sorted(rescorer_names)]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write data rows
            for finding_id, finding_result in self.results.results_by_finding.items():
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
    
    def generate_json_report(self, output_path: Path) -> None:
        """Generate a detailed JSON report."""
        # Convert results to a serializable format
        report_data = {
            'summary': {
                'findings_processed': self.results.findings_processed,
                'total_matches': self.results.total_matches,
                'timestamp': self.results.metadata.get('timestamp', ''),
                'top_k': self.results.metadata.get('top_k', ''),
                'filters_applied': self.results.metadata.get('filters_applied', False)
            },
            'findings': {}
        }
        
        for finding_id, finding_result in self.results.results_by_finding.items():
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
    
    def generate_html_report(self, output_path: Path, title: str = "Document Matching Report") -> None:
        """Generate a comprehensive HTML report."""
        html_content = self._generate_html_content(title)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_html_content(self, title: str) -> str:
        """Generate the HTML content for the report."""
        # CSS styles
        css = """
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
            h2 { color: #34495e; margin-top: 30px; }
            .summary { background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }
            .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
            .summary-item { background: white; padding: 15px; border-radius: 5px; text-align: center; }
            .summary-number { font-size: 24px; font-weight: bold; color: #3498db; }
            .finding-section { margin: 30px 0; border: 1px solid #bdc3c7; border-radius: 5px; }
            .finding-header { background: #34495e; color: white; padding: 15px; border-radius: 5px 5px 0 0; }
            .finding-text { padding: 15px; background: #f8f9fa; border-bottom: 1px solid #dee2e6; }
            .matches-table { width: 100%; border-collapse: collapse; margin: 15px 0; }
            .matches-table th, .matches-table td { border: 1px solid #dee2e6; padding: 8px; text-align: left; }
            .matches-table th { background: #ecf0f1; font-weight: bold; }
            .score-cell { text-align: center; }
            .cosine-score { background: #e8f5e8; }
            .rescorer-score { background: #fff3cd; }
            .chunk-text { max-width: 300px; word-wrap: break-word; }
            .finding-text-content { max-width: 800px; word-wrap: break-word; }
        </style>
        """
        
        # Summary section
        summary_html = f"""
        <div class="summary">
            <h2>Summary</h2>
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="summary-number">{self.results.findings_processed}</div>
                    <div>Findings Processed</div>
                </div>
                <div class="summary-item">
                    <div class="summary-number">{self.results.total_matches}</div>
                    <div>Total Matches</div>
                </div>
                <div class="summary-item">
                    <div class="summary-number">{self.results.metadata.get('top_k', 'N/A')}</div>
                    <div>Top K</div>
                </div>
                <div class="summary-item">
                    <div class="summary-number">{'Yes' if self.results.metadata.get('filters_applied') else 'No'}</div>
                    <div>Filters Applied</div>
                </div>
            </div>
            <p><strong>Timestamp:</strong> {self.results.metadata.get('timestamp', 'N/A')}</p>
        </div>
        """
        
        # Findings sections
        findings_html = ""
        for finding_id, finding_result in self.results.results_by_finding.items():
            # Get rescorer names for table headers
            rescorer_names = list(finding_result.rescorer_scores.keys())
            
            # Create table headers
            headers = ['Rank', 'Cosine Score'] + rescorer_names + ['Chunk Text']
            header_html = ''.join([f'<th>{header}</th>' for header in headers])
            
            # Create table rows
            rows_html = ""
            for i, match in enumerate(finding_result.matches):
                row_html = f"""
                <tr>
                    <td>{i+1}</td>
                    <td class="score-cell cosine-score">{match.cosine_score:.4f}</td>
                """
                
                # Add rescorer scores
                for rescorer_name in rescorer_names:
                    score = match.rescorer_scores.get(rescorer_name, 0.0)
                    row_html += f'<td class="score-cell rescorer-score">{score:.4f}</td>'
                
                row_html += f"""
                    <td class="chunk-text">{match.chunk_text[:150]}{'...' if len(match.chunk_text) > 150 else ''}</td>
                </tr>
                """
                rows_html += row_html
            
            findings_html += f"""
            <div class="finding-section">
                <div class="finding-header">
                    <h3>Finding {list(self.results.results_by_finding.keys()).index(finding_id) + 1}</h3>
                </div>
                <div class="finding-text">
                    <div class="finding-text-content">{finding_result.finding_text}</div>
                </div>
                <table class="matches-table">
                    <thead>
                        <tr>{header_html}</tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>
            """
        
        # Complete HTML
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            {css}
        </head>
        <body>
            <div class="container">
                <h1>{title}</h1>
                {summary_html}
                <h2>Detailed Results</h2>
                {findings_html}
            </div>
        </body>
        </html>
        """
        
        return html
    
    def generate_all_reports(self, output_dir: Path, base_filename: str) -> Dict[str, Path]:
        """Generate all report formats and return paths to generated files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generated_files = {}
        
        # Generate CSV report
        csv_path = output_dir / f"{base_filename}.csv"
        self.generate_csv_report(csv_path)
        generated_files['csv'] = csv_path
        
        # Generate JSON report
        json_path = output_dir / f"{base_filename}.json"
        self.generate_json_report(json_path)
        generated_files['json'] = json_path
        
        # Generate HTML report
        html_path = output_dir / f"{base_filename}.html"
        self.generate_html_report(html_path)
        generated_files['html'] = html_path
        
        return generated_files
