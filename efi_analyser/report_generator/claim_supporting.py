"""
Report generator for claim supporting results.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from .base import BaseReportGenerator
from ..types import ClaimSupportingResults, ClaimSupportingResult, ReportConfig


class ClaimSupportingReportGenerator(BaseReportGenerator):
    """Generates reports from ClaimSupportingResults."""
    
    def __init__(self, config: ReportConfig):
        super().__init__(config)
    
    def generate_csv(self, results: List[ClaimSupportingResults], output_path: Optional[Path] = None) -> Path:
        """Generate a CSV report with all claim supporting results."""
        if output_path is None:
            output_path = self._get_default_output_path("csv", "claim_supporting")
        
        self._ensure_output_dir(output_path)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'claim', 'date', 'media', 'classification', 'entailment_score',
                'neutral_score', 'contradiction_score', 'cosine_score', 'url', 'title'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write data rows
            for claim_result in results:
                for result in claim_result.results:
                    row = {
                        'claim': result.claim,
                        'date': result.date,
                        'media': result.media_source,
                        'classification': result.classification,
                        'entailment_score': result.entailment_score,
                        'neutral_score': result.neutral_score,
                        'contradiction_score': result.contradiction_score,
                        'cosine_score': result.cosine_score,
                        'url': result.url,
                        'title': result.title
                    }
                    writer.writerow(row)
        
        return output_path
    
    def generate_json(self, results: List[ClaimSupportingResults], output_path: Optional[Path] = None) -> Path:
        """Generate a detailed JSON report."""
        if output_path is None:
            output_path = self._get_default_output_path("json", "claim_supporting")
        
        self._ensure_output_dir(output_path)
        
        # Convert results to a serializable format
        report_data = {
            'summary': {
                'total_claims': len(results),
                'total_results': sum(len(r.results) for r in results),
                'timestamp': datetime.now().isoformat(),
                'classification_summary': self._get_classification_summary(results)
            },
            'claims': {}
        }
        
        for claim_result in results:
            report_data['claims'][claim_result.claim] = {
                'total_chunks': claim_result.total_chunks,
                'entailment_count': claim_result.entailment_count,
                'contradiction_count': claim_result.contradiction_count,
                'neutral_count': claim_result.neutral_count,
                'media_breakdown': claim_result.media_breakdown,
                'results': [
                    {
                        'chunk_id': result.chunk_id,
                        'document_id': result.document_id,
                        'classification': result.classification,
                        'entailment_score': result.entailment_score,
                        'neutral_score': result.neutral_score,
                        'contradiction_score': result.contradiction_score,
                        'cosine_score': result.cosine_score,
                        'media_source': result.media_source,
                        'date': result.date,
                        'url': result.url,
                        'title': result.title
                    }
                    for result in claim_result.results
                ]
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def generate_html(self, results: List[ClaimSupportingResults], output_path: Optional[Path] = None) -> Path:
        """Generate a comprehensive HTML report."""
        if output_path is None:
            output_path = self._get_default_output_path("html", "claim_supporting")
        
        self._ensure_output_dir(output_path)
        
        html_content = self._generate_html_content(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
    
    def _generate_html_content(self, results: List[ClaimSupportingResults]) -> str:
        """Generate HTML content for the report."""
        classification_summary = self._get_classification_summary(results)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Claim Supporting Report</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ margin: 20px 0; }}
                .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
                .stat-card {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; text-align: center; }}
                .stat-number {{ font-size: 2em; font-weight: bold; color: #2c5aa0; }}
                .claim-section {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                .claim-text {{ font-weight: bold; font-size: 1.1em; margin-bottom: 15px; }}
                .table-container {{ margin: 20px 0; overflow-x: auto; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .classification-entailment {{ color: #28a745; }}
                .classification-contradiction {{ color: #dc3545; }}
                .classification-neutral {{ color: #6c757d; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Claim Supporting Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Total Claims: {len(results)}</p>
                <p>Total Results: {sum(len(r.results) for r in results)}</p>
            </div>
            
            <div class="summary">
                <h2>Overall Classification Summary</h2>
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-number">{classification_summary['entailment']}</div>
                        <div>Entailment</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{classification_summary['neutral']}</div>
                        <div>Neutral</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{classification_summary['contradiction']}</div>
                        <div>Contradiction</div>
                    </div>
                </div>
            </div>
        """
        
        # Add per-claim analysis
        for claim_result in results:
            html_content += f"""
            <div class="claim-section">
                <div class="claim-text">Claim: {claim_result.claim[:100]}...</div>
                
                <h3>Overall Classification</h3>
                <div class="stats">
                    <div class="stat-card">
                        <div class="stat-number">{claim_result.entailment_count}</div>
                        <div>Entailment ({(claim_result.entailment_count/claim_result.total_chunks*100) if claim_result.total_chunks > 0 else 0:.1f}%)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{claim_result.neutral_count}</div>
                        <div>Neutral ({(claim_result.neutral_count/claim_result.total_chunks*100) if claim_result.total_chunks > 0 else 0:.1f}%)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{claim_result.contradiction_count}</div>
                        <div>Contradiction ({(claim_result.contradiction_count/claim_result.total_chunks*100) if claim_result.total_chunks > 0 else 0:.1f}%)</div>
                    </div>
                </div>
                
                <h3>Media Source Breakdown</h3>
                <div class="table-container">
                    <table>
                        <tr>
                            <th>Media Source</th>
                            <th>Articles</th>
                            <th>Entailment</th>
                            <th>Neutral</th>
                            <th>Contradiction</th>
                        </tr>
            """
            
            for media, counts in claim_result.media_breakdown.items():
                html_content += f"""
                        <tr>
                            <td>{media}</td>
                            <td>{counts.get('total', 0)}</td>
                            <td>{counts.get('entailment', 0)}</td>
                            <td>{counts.get('neutral', 0)}</td>
                            <td>{counts.get('contradiction', 0)}</td>
                        </tr>
                """
            
            html_content += """
                    </table>
                </div>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        return html_content
    
    def _get_classification_summary(self, results: List[ClaimSupportingResults]) -> Dict[str, int]:
        """Get overall classification summary across all claims."""
        total_entailment = sum(r.entailment_count for r in results)
        total_neutral = sum(r.neutral_count for r in results)
        total_contradiction = sum(r.contradiction_count for r in results)
        
        return {
            'entailment': total_entailment,
            'neutral': total_neutral,
            'contradiction': total_contradiction
        }
