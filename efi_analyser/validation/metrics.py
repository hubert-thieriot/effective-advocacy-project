"""
Performance metrics for scorer evaluation.
"""

from __future__ import annotations

from typing import Dict, List, Any
from collections import defaultdict
import numpy as np


class ClassificationMetrics:
    """Calculate classification performance metrics."""

    @staticmethod
    def accuracy(predictions: List[str], gold_labels: List[str]) -> float:
        """Calculate accuracy."""
        if len(predictions) != len(gold_labels):
            raise ValueError("Predictions and gold labels must have same length")

        correct = sum(1 for pred, gold in zip(predictions, gold_labels) if pred == gold)
        return correct / len(gold_labels) if gold_labels else 0.0

    @staticmethod
    def precision_recall_f1(predictions: List[str], gold_labels: List[str],
                          average: str = 'macro') -> Dict[str, float]:
        """Calculate precision, recall, and F1 scores.

        Args:
            predictions: Predicted labels
            gold_labels: Gold standard labels
            average: 'macro', 'micro', or 'weighted'

        Returns:
            Dict with precision, recall, f1, and per-class metrics
        """
        if len(predictions) != len(gold_labels):
            raise ValueError("Predictions and gold labels must have same length")

        # Get unique labels
        labels = list(set(gold_labels + predictions))

        # Calculate per-class metrics
        metrics = {}
        for label in labels:
            tp = sum(1 for p, g in zip(predictions, gold_labels) if p == label and g == label)
            fp = sum(1 for p, g in zip(predictions, gold_labels) if p == label and g != label)
            fn = sum(1 for p, g in zip(predictions, gold_labels) if p != label and g == label)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics[label] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': tp + fn
            }

        # Calculate aggregate metrics
        if average == 'macro':
            precision_avg = np.mean([m['precision'] for m in metrics.values()])
            recall_avg = np.mean([m['recall'] for m in metrics.values()])
            f1_avg = np.mean([m['f1'] for m in metrics.values()])
        elif average == 'micro':
            total_tp = sum(m['support'] for m in metrics.values() if m['support'] > 0)
            total_pred = sum(tp + fp for tp, fp, fn in
                           [(sum(1 for p, g in zip(predictions, gold_labels) if p == label and g == label),
                             sum(1 for p, g in zip(predictions, gold_labels) if p == label and g != label),
                             sum(1 for p, g in zip(predictions, gold_labels) if p != label and g == label))
                            for label in labels])
            total_gold = sum(tp + fn for tp, fp, fn in
                           [(sum(1 for p, g in zip(predictions, gold_labels) if p == label and g == label),
                             sum(1 for p, g in zip(predictions, gold_labels) if p == label and g != label),
                             sum(1 for p, g in zip(predictions, gold_labels) if p != label and g == label))
                            for label in labels])

            precision_avg = total_tp / total_pred if total_pred > 0 else 0.0
            recall_avg = total_tp / total_gold if total_gold > 0 else 0.0
            f1_avg = 2 * precision_avg * recall_avg / (precision_avg + recall_avg) if (precision_avg + recall_avg) > 0 else 0.0
        else:
            # weighted average
            total_support = sum(m['support'] for m in metrics.values())
            precision_avg = sum(m['precision'] * m['support'] for m in metrics.values()) / total_support if total_support > 0 else 0.0
            recall_avg = sum(m['recall'] * m['support'] for m in metrics.values()) / total_support if total_support > 0 else 0.0
            f1_avg = sum(m['f1'] * m['support'] for m in metrics.values()) / total_support if total_support > 0 else 0.0

        return {
            f'{average}_precision': precision_avg,
            f'{average}_recall': recall_avg,
            f'{average}_f1': f1_avg,
            'per_class': metrics
        }

    @staticmethod
    def confusion_matrix(predictions: List[str], gold_labels: List[str]) -> Dict[str, Dict[str, int]]:
        """Calculate confusion matrix."""
        if len(predictions) != len(gold_labels):
            raise ValueError("Predictions and gold labels must have same length")

        labels = list(set(gold_labels + predictions))
        matrix = defaultdict(lambda: defaultdict(int))

        for pred, gold in zip(predictions, gold_labels):
            matrix[gold][pred] += 1

        # Convert to regular dict for all labels
        return {gold: {pred: matrix[gold][pred] for pred in labels} for gold in labels}

    @staticmethod
    def calculate_all_metrics(predictions: List[str], gold_labels: List[str]) -> Dict[str, Any]:
        """Calculate all available metrics."""
        metrics = {
            'accuracy': ClassificationMetrics.accuracy(predictions, gold_labels),
            'confusion_matrix': ClassificationMetrics.confusion_matrix(predictions, gold_labels)
        }

        prf = ClassificationMetrics.precision_recall_f1(predictions, gold_labels, 'macro')
        metrics.update(prf)

        prf_micro = ClassificationMetrics.precision_recall_f1(predictions, gold_labels, 'micro')
        metrics.update({k: v for k, v in prf_micro.items() if k != 'per_class'})

        prf_weighted = ClassificationMetrics.precision_recall_f1(predictions, gold_labels, 'weighted')
        metrics.update({k: v for k, v in prf_weighted.items() if k != 'per_class'})

        return metrics
