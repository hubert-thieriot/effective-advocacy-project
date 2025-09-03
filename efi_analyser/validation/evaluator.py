"""
Evaluator for running scorers on validation datasets.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from efi_core.types import PairScorer, Task

from .types import ValidationDataset, EvaluationResult, TaskType
from .metrics import ClassificationMetrics


class ScorerEvaluator:
    """Evaluates scorers on validation datasets."""

    def __init__(self, scorer: PairScorer):
        """Initialize evaluator with a scorer."""
        self.scorer = scorer

    def evaluate_dataset(self, dataset: ValidationDataset) -> EvaluationResult:
        """Evaluate scorer on a validation dataset.

        Args:
            dataset: The validation dataset to evaluate on

        Returns:
            EvaluationResult with predictions and metrics
        """
        # Get text pairs from dataset
        texts_a, texts_b = dataset.get_texts()
        gold_labels = dataset.get_gold_labels()

        # Run scorer on all pairs
        predictions = self.scorer.batch_score(texts_a, texts_b)

        # Convert probability distributions to predicted labels
        predicted_labels = self._probabilities_to_labels(predictions, dataset.task_type)

        # Calculate metrics
        metrics = ClassificationMetrics.calculate_all_metrics(predicted_labels, gold_labels)

        # Create detailed sample results
        sample_results = []
        for i, (sample, pred_probs, pred_label, gold_label) in enumerate(
            zip(dataset.samples, predictions, predicted_labels, gold_labels)
        ):
            sample_results.append({
                'sample_id': sample.sample_id,
                'text_a': sample.text_a,
                'text_b': sample.text_b,
                'gold_label': gold_label,
                'predicted_label': pred_label,
                'probabilities': pred_probs,
                'correct': pred_label == gold_label,
                'metadata': sample.metadata
            })

        return EvaluationResult(
            scorer_name=self.scorer.name,
            dataset_name=dataset.name,
            task_type=dataset.task_type,
            predictions=predictions,
            gold_labels=gold_labels,
            metrics=metrics,
            sample_results=sample_results
        )

    def _probabilities_to_labels(self, predictions: List[Dict[str, float]],
                               task_type: TaskType) -> List[str]:
        """Convert probability distributions to predicted labels using argmax."""
        predicted_labels = []

        for probs in predictions:
            # Use argmax to find the label with highest probability
            best_label = max(probs.items(), key=lambda x: x[1])[0]
            predicted_labels.append(best_label)

        return predicted_labels


class EvaluationRunner:
    """Runs multiple scorers on multiple datasets."""

    def __init__(self, scorers: List[PairScorer], verbose: bool = False):
        """Initialize with multiple scorers."""
        self.scorers = scorers
        self.evaluators = [ScorerEvaluator(scorer) for scorer in scorers]
        self.verbose = verbose

    def evaluate_all(self, datasets: List[ValidationDataset]) -> List[EvaluationResult]:
        """Evaluate all scorers on all datasets."""
        results = []
        total_evaluations = sum(1 for evaluator in self.evaluators for dataset in datasets
                               if self._task_matches(evaluator.scorer.task, dataset.task_type))
        current_evaluation = 0

        if self.verbose:
            print(f"\n🔬 Starting evaluation of {len(self.evaluators)} scorers on {len(datasets)} datasets...")
            print(f"   Total evaluations to perform: {total_evaluations}")

        for evaluator in self.evaluators:
            for dataset in datasets:
                # Only evaluate if scorer task matches dataset task
                if self._task_matches(evaluator.scorer.task, dataset.task_type):
                    current_evaluation += 1
                    if self.verbose:
                        print(f"\n📊 [{current_evaluation}/{total_evaluations}] Evaluating {evaluator.scorer.name} on {dataset.name}...")

                    result = evaluator.evaluate_dataset(dataset)
                    results.append(result)

                    print(f"✅ Evaluated {result.scorer_name} on {result.dataset_name}: "
                          f"Accuracy={result.accuracy:.3f}, Macro-F1={result.macro_f1:.3f}")
                else:
                    if self.verbose:
                        print(f"⏭️  Skipping {evaluator.scorer.name} on {dataset.name}: "
                              f"Task mismatch ({evaluator.scorer.task.value} vs {dataset.task_type.value})")

        if self.verbose:
            print(f"\n🎉 Evaluation completed! Processed {len(results)} scorer-dataset combinations.")

        return results

    def _task_matches(self, scorer_task: Task, dataset_task: TaskType) -> bool:
        """Check if scorer task matches dataset task type."""
        return (scorer_task.value == "nli" and dataset_task == TaskType.NLI) or \
               (scorer_task.value == "stance" and dataset_task == TaskType.STANCE)
