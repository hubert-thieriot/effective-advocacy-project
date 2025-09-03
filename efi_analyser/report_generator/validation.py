"""
Report generator for validation results.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from .base import BaseReportGenerator
from ..validation import EvaluationResult, ValidationDataset


class ValidationReportGenerator(BaseReportGenerator):
    """Generates reports from validation results."""

    def __init__(self, config):
        super().__init__(config)

    def generate_csv(self, results: List[EvaluationResult], output_path: Optional[Path] = None) -> Path:
        """Generate CSV report with scorer comparison."""
        if output_path is None:
            output_path = self._get_default_output_path("csv", "validation")

        self._ensure_output_dir(output_path)

        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'task_type', 'scorer_name', 'dataset_name',
                'accuracy', 'macro_f1', 'macro_precision', 'macro_recall',
                'entails_f1', 'contradicts_f1', 'neutral_f1', 'pro_f1', 'anti_f1', 'uncertain_f1'
            ]

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                row = {
                    'task_type': result.task_type.value,
                    'scorer_name': result.scorer_name,
                    'dataset_name': result.dataset_name,
                    'accuracy': result.metrics['accuracy'],
                    'macro_f1': result.metrics.get('macro_f1', 0.0),
                    'macro_precision': result.metrics.get('macro_precision', 0.0),
                    'macro_recall': result.metrics.get('macro_recall', 0.0),
                }

                # Add per-class F1 scores for both NLI and stance
                per_class = result.metrics.get('per_class', {})
                for label in ['entails', 'contradicts', 'neutral', 'pro', 'anti', 'uncertain']:
                    row[f'{label}_f1'] = per_class.get(label, {}).get('f1', 0.0)

                writer.writerow(row)

        return output_path

    def generate_json(self, results: List[EvaluationResult], output_path: Optional[Path] = None) -> Path:
        """Generate JSON report."""
        if output_path is None:
            output_path = self._get_default_output_path("json", "validation")

        self._ensure_output_dir(output_path)

        # Convert results to serializable format
        data = {
            'timestamp': datetime.now().isoformat(),
            'results': []
        }

        for result in results:
            result_dict = {
                'scorer_name': result.scorer_name,
                'dataset_name': result.dataset_name,
                'task_type': result.task_type.value,
                'metrics': result.metrics,
                'sample_count': len(result.sample_results)
            }
            data['results'].append(result_dict)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return output_path

    def generate_html(self, results: List[EvaluationResult], output_path: Optional[Path] = None) -> Path:
        """Generate comprehensive HTML report."""
        if output_path is None:
            output_path = self._get_default_output_path("html", "validation")

        self._ensure_output_dir(output_path)

        # Sort results by accuracy (best first)
        sorted_results = sorted(results, key=lambda x: x.metrics['accuracy'], reverse=True)

        html_content = self._generate_html_header()
        html_content += self._generate_summary_section(sorted_results)
        html_content += self._generate_detailed_results_section(sorted_results)
        html_content += self._generate_sample_predictions_section(sorted_results)
        html_content += self._generate_footer()

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return output_path

    def _generate_html_header(self) -> str:
        """Generate HTML header with styling."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EFI Scorer Validation Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .summary-card {{
            background: white;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #28a745;
        }}
        .metric-label {{
            font-size: 14px;
            color: #6c757d;
            margin-top: 5px;
        }}
        .table-container {{
            background: white;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow-x: auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
            color: #495057;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        tr:hover {{
            background-color: #e9ecef;
        }}
        .best-score {{
            background-color: #d4edda !important;
            font-weight: bold;
        }}
        .accuracy-bar {{
            height: 20px;
            background: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin-top: 5px;
        }}
        .accuracy-fill {{
            height: 100%;
            background: linear-gradient(90deg, #28a745 0%, #20c997 100%);
            border-radius: 10px;
        }}
        .section-title {{
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 20px;
            color: #495057;
            border-bottom: 2px solid #dee2e6;
            padding-bottom: 10px;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
        }}
        .badge-success {{
            background-color: #d4edda;
            color: #155724;
        }}
        .badge-warning {{
            background-color: #fff3cd;
            color: #856404;
        }}
        .badge-danger {{
            background-color: #f8d7da;
            color: #721c24;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 EFI Scorer Validation Report</h1>
        <p>Comprehensive evaluation of scorer performance across different models and backends</p>
        <p><strong>Generated:</strong> {timestamp}</p>
    </div>
"""

    def _generate_summary_section(self, results: List[EvaluationResult]) -> str:
        """Generate summary section with key metrics, grouped by task type."""
        if not results:
            return '<div class="summary-card"><p>No results to display</p></div>'

        # Group results by task type
        task_groups = {}
        for result in results:
            task_type = result.task_type.value
            if task_type not in task_groups:
                task_groups[task_type] = []
            task_groups[task_type].append(result)

        html = '''
    <div class="summary-card">
        <h2 class="section-title">📊 Executive Summary</h2>
'''

        for task_type, task_results in task_groups.items():
            best_result = max(task_results, key=lambda x: x.metrics['accuracy'])
            total_samples = len(best_result.sample_results)
            avg_accuracy = sum(r.metrics['accuracy'] for r in task_results) / len(task_results)

            html += f'''
        <div style="margin-bottom: 30px; padding: 20px; background: #f8f9fa; border-radius: 8px;">
            <h3 style="color: #495057; margin-bottom: 15px;">{task_type.upper()} Dataset Results</h3>

            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{len(task_results)}</div>
                    <div class="metric-label">Scorers Evaluated</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{total_samples}</div>
                    <div class="metric-label">Validation Samples</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{best_result.metrics['accuracy']:.1%}</div>
                    <div class="metric-label">Best Accuracy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{avg_accuracy:.1%}</div>
                    <div class="metric-label">Average Accuracy</div>
                </div>
            </div>

            <h4 style="color: #28a745; margin-bottom: 10px;">🏆 Top Performer: {best_result.scorer_name}</h4>
            <p><strong>Dataset:</strong> {best_result.dataset_name}</p>

            <div class="accuracy-bar">
                <div class="accuracy-fill" style="width: {best_result.metrics['accuracy'] * 100}%"></div>
            </div>
            <p style="text-align: center; margin-top: 5px; font-size: 14px;">
                {best_result.metrics['accuracy']:.1%}
            </p>
        </div>
'''
        html += '    </div>'
        return html

    def _generate_detailed_results_section(self, results: List[EvaluationResult]) -> str:
        """Generate detailed results table, grouped by task type."""
        # Group results by task type
        task_groups = {}
        for result in results:
            task_type = result.task_type.value
            if task_type not in task_groups:
                task_groups[task_type] = []
            task_groups[task_type].append(result)

        html = '''
    <div class="table-container">
        <h2 class="section-title">📈 Detailed Results</h2>
'''

        for task_type, task_results in task_groups.items():
            # Sort by accuracy within each task type
            task_results = sorted(task_results, key=lambda x: x.metrics['accuracy'], reverse=True)

            html += f'''
        <h3 style="color: #495057; margin-top: 30px; margin-bottom: 15px;">{task_type.upper()} Dataset Results</h3>

        <table>
            <thead>
                <tr>
                    <th>Scorer</th>
                    <th>Accuracy</th>
                    <th>Macro F1</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>Entails F1</th>
                    <th>Contradicts F1</th>
                    <th>Neutral F1</th>
                </tr>
            </thead>
            <tbody>
'''

            for i, result in enumerate(task_results):
                metrics = result.metrics
                per_class = metrics.get('per_class', {})

                # Determine performance tier
                accuracy = metrics['accuracy']
                if accuracy >= 0.8:
                    badge_class = "badge-success"
                    badge_text = "Excellent"
                elif accuracy >= 0.6:
                    badge_class = "badge-warning"
                    badge_text = "Good"
                else:
                    badge_class = "badge-danger"
                    badge_text = "Needs Improvement"

                is_best = i == 0
                row_class = "best-score" if is_best else ""

                html += f'''
                <tr class="{row_class}">
                    <td>
                        <strong>{result.scorer_name}</strong>
                        <br><small style="color: #6c757d;">{result.dataset_name}</small>
                        {"<span class='badge " + badge_class + "'>" + badge_text + "</span>" if is_best else ""}
                    </td>
                    <td><strong>{metrics['accuracy']:.3f}</strong></td>
                    <td>{metrics.get('macro_f1', 0.0):.3f}</td>
                    <td>{metrics.get('macro_precision', 0.0):.3f}</td>
                    <td>{metrics.get('macro_recall', 0.0):.3f}</td>
                    <td>{per_class.get('entails', {}).get('f1', 0.0):.3f}</td>
                    <td>{per_class.get('contradicts', {}).get('f1', 0.0):.3f}</td>
                    <td>{per_class.get('neutral', {}).get('f1', 0.0):.3f}</td>
                </tr>
'''

            html += '''
            </tbody>
        </table>
'''

        html += '''
    </div>
'''
        return html

    def _generate_sample_predictions_section(self, results: List[EvaluationResult]) -> str:
        """Generate sample predictions section showing actual examples from all scorers."""
        if not results:
            return ''

        # Group sample results by sample_id across all scorers
        sample_groups = {}

        for result in results:
            for sample_result in result.sample_results:
                sample_id = sample_result.get('sample_id', f"{sample_result['text_a'][:50]}_{sample_result['text_b'][:50]}")
                if sample_id not in sample_groups:
                    sample_groups[sample_id] = {
                        'text_a': sample_result['text_a'],
                        'text_b': sample_result['text_b'],
                        'gold_label': sample_result['gold_label'],
                        'predictions': {}
                    }
                sample_groups[sample_id]['predictions'][result.scorer_name] = {
                    'predicted_label': sample_result['predicted_label'],
                    'probabilities': sample_result.get('probabilities', {}),
                    'correct': sample_result['correct']
                }

        # Select diverse samples (mix of easy, hard, and mixed performance)
        selected_samples = self._select_diverse_samples(sample_groups)

        if not selected_samples:
            return ''

        html = '''
    <div class="table-container">
        <h2 class="section-title">🔍 Sample Predictions Comparison</h2>
        <p style="color: #6c757d; margin-bottom: 20px;">
            Comparing all {} scorers on the same samples to see performance differences
        </p>
'''.format(len(results))

        for i, sample_data in enumerate(selected_samples):
            # Truncate long text for display
            text_a = sample_data['text_a'][:100] + "..." if len(sample_data['text_a']) > 100 else sample_data['text_a']
            text_b = sample_data['text_b'][:100] + "..." if len(sample_data['text_b']) > 100 else sample_data['text_b']

            html += f'''
        <div style="background: #f8f9fa; border-radius: 8px; padding: 20px; margin-bottom: 20px;">
            <div style="margin-bottom: 15px;">
                <h4 style="margin: 0 0 10px 0; color: #495057;">Sample {i+1}</h4>
                <div style="margin-bottom: 8px;">
                    <strong style="color: #007bff;">Premise:</strong>
                    <span style="font-family: 'Courier New', monospace; background: #e9ecef; padding: 2px 6px; border-radius: 3px;">"{text_a}"</span>
                </div>
                <div style="margin-bottom: 8px;">
                    <strong style="color: #28a745;">Hypothesis:</strong>
                    <span style="font-family: 'Courier New', monospace; background: #e9ecef; padding: 2px 6px; border-radius: 3px;">"{text_b}"</span>
                </div>
                <div>
                    <strong style="color: #6c757d;">Gold Label:</strong>
                    <span style="background: #d1ecf1; color: #0c5460; padding: 4px 8px; border-radius: 4px; font-weight: bold;">{sample_data['gold_label']}</span>
                </div>
            </div>

            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
'''

            # Show predictions from each scorer
            for scorer_name, prediction in sample_data['predictions'].items():
                status_icon = "✅" if prediction['correct'] else "❌"
                status_color = "#28a745" if prediction['correct'] else "#dc3545"

                html += f'''
                <div style="background: white; padding: 12px; border-radius: 6px; border: 2px solid {status_color}20;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <strong style="font-size: 14px; color: #495057;">{scorer_name}</strong>
                        <span style="font-size: 16px;">{status_icon}</span>
                    </div>
                    <div style="margin-bottom: 4px;">
                        <span style="background: {'#d4edda' if prediction['correct'] else '#f8d7da'}; color: {'#155724' if prediction['correct'] else '#721c24'}; padding: 2px 6px; border-radius: 3px; font-weight: bold; font-size: 13px;">{prediction['predicted_label']}</span>
                    </div>
                    {f'<div style="font-size: 12px; color: #6c757d;">{self._format_probabilities(prediction.get("probabilities", {}))}</div>' if prediction.get('probabilities') else ''}
                </div>
'''

            html += '''
            </div>
        </div>
'''

        html += '''
    </div>
'''
        return html

    def _select_diverse_samples(self, sample_groups: Dict) -> List[Dict]:
        """Select diverse samples that show different performance patterns."""
        if not sample_groups:
            return []

        # Calculate performance patterns for each sample
        sample_scores = []
        for sample_id, sample_data in sample_groups.items():
            predictions = sample_data['predictions']

            # Count correct predictions per scorer
            correct_count = sum(1 for pred in predictions.values() if pred['correct'])
            total_scorers = len(predictions)

            # Calculate agreement (how many scorers agree)
            labels = [pred['predicted_label'] for pred in predictions.values()]
            agreement = len(set(labels)) == 1  # All agree

            # Calculate difficulty (inverse of average correctness)
            difficulty = 1.0 - (correct_count / total_scorers)

            sample_scores.append({
                'sample_id': sample_id,
                'data': sample_data,
                'correct_count': correct_count,
                'total_scorers': total_scorers,
                'agreement': agreement,
                'difficulty': difficulty
            })

        # Sort by diversity criteria
        # Prioritize: mixed performance, disagreements, then difficulty
        sample_scores.sort(key=lambda x: (
            not x['agreement'],  # Prefer disagreements first
            -abs(x['correct_count'] - x['total_scorers']/2),  # Prefer mixed performance
            -x['difficulty']  # Then by difficulty
        ), reverse=True)

        # Select up to 8 diverse samples
        selected = sample_scores[:8]

        return [score['data'] for score in selected]

    def _format_probabilities(self, probabilities: Dict[str, float]) -> str:
        """Format probability dictionary for display."""
        if not probabilities:
            return "N/A"

        prob_items = []
        for label, prob in probabilities.items():
            prob_items.append(f"{label}: {prob:.3f}")

        return " | ".join(prob_items)

    def _generate_footer(self) -> str:
        """Generate HTML footer."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f'''
    <div class="summary-card">
        <p style="text-align: center; color: #6c757d; margin: 0;">
            Generated by EFI Validation Framework • {timestamp}
        </p>
    </div>
</body>
</html>
'''

    def _get_default_output_path(self, extension: str, prefix: str) -> Path:
        """Get default output path for reports."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.output_dir / f"{prefix}_report_{timestamp}.{extension}"

    def _ensure_output_dir(self, output_path: Path) -> None:
        """Ensure output directory exists."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
