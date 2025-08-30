"""
EFI Analyser - Word Occurrence Report Generator

Generates reports for word occurrence analysis results.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from .base import BaseReportGenerator
from ..types import WordOccurrenceResults, ReportConfig


class WordOccurrenceReportGenerator(BaseReportGenerator):
    """Report generator for word occurrence analysis results."""

    def __init__(self, config: ReportConfig):
        super().__init__(config)

    def generate_csv(self, results: WordOccurrenceResults, output_path: Optional[Path] = None) -> Path:
        """Generate CSV report from word occurrence results."""
        if output_path is None:
            output_path = self._get_default_output_path("csv", "word_occurrence")
        
        self._ensure_output_dir(output_path)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            # Write summary
            writer = csv.writer(csvfile)
            writer.writerow(['Word Occurrence Analysis Summary'])
            writer.writerow([])
            writer.writerow(['Keywords'] + results.keywords)
            writer.writerow(['Total Documents', results.total_documents])
            writer.writerow([])
            
            # Write keyword counts and percentages
            writer.writerow(['Keyword', 'Documents Mentioning', 'Percentage'])
            for keyword in results.keywords:
                count = results.keyword_counts.get(keyword, 0)
                percentage = results.keyword_percentages.get(keyword, 0.0)
                writer.writerow([keyword, count, f"{percentage:.2f}%"])
            
            writer.writerow([])
            
            # Write detailed results
            writer.writerow(['Document ID', 'URL', 'Title', 'Date'] + [f'{k}_count' for k in results.keywords] + ['Total Keywords'])
            for result in results.results:
                row = [
                    result.document_id,
                    result.url,
                    result.title,
                    result.date
                ]
                # Add keyword counts
                for keyword in results.keywords:
                    row.append(result.keyword_counts.get(keyword, 0))
                row.append(result.total_keywords)
                writer.writerow(row)
        
        return output_path

    def generate_json(self, results: WordOccurrenceResults, output_path: Optional[Path] = None) -> Path:
        """Generate JSON report from word occurrence results."""
        if output_path is None:
            output_path = self._get_default_output_path("json", "word_occurrence")
        
        self._ensure_output_dir(output_path)
        
        # Convert results to serializable format
        report_data = {
            "analysis_type": "word_occurrence",
            "timestamp": datetime.now().isoformat(),
            "keywords": results.keywords,
            "total_documents": results.total_documents,
            "keyword_counts": results.keyword_counts,
            "keyword_percentages": results.keyword_percentages,
            "timing_stats": results.timing_stats,
            "metadata": results.metadata,
            "detailed_results": [
                {
                    "document_id": r.document_id,
                    "url": r.url,
                    "title": r.title,
                    "date": r.date,
                    "keyword_counts": r.keyword_counts,
                    "keyword_positions": r.keyword_positions,
                    "total_keywords": r.total_keywords,
                    "metadata": r.metadata
                }
                for r in results.results
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as jsonfile:
            json.dump(report_data, jsonfile, indent=2, ensure_ascii=False)
        
        return output_path

    def generate_html(self, results: WordOccurrenceResults, output_path: Optional[Path] = None) -> Path:
        """Generate HTML report from word occurrence results."""
        if output_path is None:
            output_path = self._get_default_output_path("html", "word_occurrence")
        
        self._ensure_output_dir(output_path)
        
        # Generate HTML content
        html_content = self._generate_html_content(results)
        
        with open(output_path, 'w', encoding='utf-8') as htmlfile:
            htmlfile.write(html_content)
        
        return output_path

    def _generate_html_content(self, results: WordOccurrenceResults) -> str:
        """Generate HTML content for the report."""
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Word Occurrence Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .summary {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .keyword-table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
        .keyword-table th, .keyword-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .keyword-table th {{ background-color: #f2f2f2; }}
        .results-table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
        .results-table th, .results-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 12px; }}
        .results-table th {{ background-color: #f2f2f2; }}
        .stats {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Word Occurrence Analysis Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="summary">
        <h2>Analysis Summary</h2>
        <p><strong>Keywords:</strong> {', '.join(results.keywords)}</p>
        <p><strong>Total Documents:</strong> {results.total_documents}</p>
    </div>
    
    <div class="stats">
        <h2>Keyword Statistics</h2>
        <table class="keyword-table">
            <thead>
                <tr>
                    <th>Keyword</th>
                    <th>Documents Mentioning</th>
                    <th>Percentage</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for keyword in results.keywords:
            count = results.keyword_counts.get(keyword, 0)
            percentage = results.keyword_percentages.get(keyword, 0.0)
            html += f"""
                <tr>
                    <td>{keyword}</td>
                    <td>{count}</td>
                    <td>{percentage:.2f}%</td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
    </div>
    
    <div class="stats">
        <h2>Detailed Results</h2>
        <table class="results-table">
            <thead>
                <tr>
                    <th>Document ID</th>
                    <th>URL</th>
                    <th>Title</th>
                    <th>Date</th>
"""
        
        for keyword in results.keywords:
            html += f"                    <th>{keyword} Count</th>\n"
        
        html += """
                    <th>Total Keywords</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for result in results.results:
            html += f"""
                <tr>
                    <td>{result.document_id}</td>
                    <td><a href="{result.url}" target="_blank">{result.url}</a></td>
                    <td>{result.title}</td>
                    <td>{result.date}</td>
"""
            
            for keyword in results.keywords:
                count = result.keyword_counts.get(keyword, 0)
                html += f"                    <td>{count}</td>\n"
            
            html += f"                    <td>{result.total_keywords}</td>\n                </tr>\n"
        
        html += """
            </tbody>
        </table>
    </div>
    
    <div class="stats">
        <h2>Analysis Metadata</h2>
        <p><strong>Case Sensitive:</strong> {results.metadata.get('case_sensitive', False)}</p>
        <p><strong>Whole Word Only:</strong> {results.metadata.get('whole_word_only', True)}</p>
        <p><strong>Allow Hyphenation:</strong> {results.metadata.get('allow_hyphenation', True)}</p>
        <p><strong>Processing Time:</strong> {results.timing_stats.get('total_time', 0):.2f} seconds</p>
    </div>
</body>
</html>
"""
        
        return html
