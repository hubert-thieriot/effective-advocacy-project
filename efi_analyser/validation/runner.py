"""
Validation runner for evaluating scorers on datasets.
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import asdict

from efi_core.types import Task
from efi_analyser.scorers import NLIHFScorer, NLIHFScorerConfig

from .datasets import NLIDataset, StanceDataset
from .evaluator import EvaluationRunner
from .types import ValidationDataset, EvaluationResult


def create_sample_scorers() -> List:
    """Create sample scorers for demonstration."""
    # NLI Scorer
    nli_config = NLIHFScorerConfig(
        model_name="typeform/distilbert-base-uncased-mnli",
        device=-1,  # CPU
        batch_size=4
    )
    nli_scorer = NLIHFScorer(name="nli_distilbert", task=Task.NLI, config=nli_config)

    return [nli_scorer]


def load_dataset_from_file(file_path: Path) -> ValidationDataset:
    """Load a validation dataset from file."""
    if "nli" in file_path.name.lower():
        dataset = NLIDataset(file_path.stem)
    elif "stance" in file_path.name.lower():
        dataset = StanceDataset(file_path.stem)
    else:
        # Try to infer from content
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
            if "entails" in first_line or "contradicts" in first_line:
                dataset = NLIDataset(file_path.stem)
            else:
                dataset = StanceDataset(file_path.stem)

    dataset.load_from_file(file_path)
    return dataset


def save_evaluation_results(results: List[EvaluationResult], output_dir: Path) -> None:
    """Save evaluation results to files."""
    output_dir.mkdir(exist_ok=True)

    # Save summary
    summary = []
    for result in results:
        summary.append({
            'scorer_name': result.scorer_name,
            'dataset_name': result.dataset_name,
            'task_type': result.task_type.value,
            'accuracy': result.accuracy,
            'macro_f1': result.macro_f1,
            'metrics': result.metrics
        })

    summary_file = output_dir / "evaluation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Save detailed results for each evaluation
    for result in results:
        result_file = output_dir / f"{result.scorer_name}_{result.dataset_name}_results.json"
        result_dict = asdict(result)
        # Convert task type enum to string
        result_dict['task_type'] = result.task_type.value

        with open(result_file, 'w') as f:
            json.dump(result_dict, f, indent=2)

    print(f"\nResults saved to {output_dir}")


def print_evaluation_summary(results: List[EvaluationResult]) -> None:
    """Print a summary of evaluation results."""
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)

    for result in results:
        print(f"\nScorer: {result.scorer_name}")
        print(f"Dataset: {result.dataset_name} ({result.task_type.value})")
        print(".3f")
        print(".3f")
        print(f"Macro Precision: {result.metrics.get('macro_precision', 0):.3f}")
        print(f"Macro Recall: {result.metrics.get('macro_recall', 0):.3f}")

        # Show per-class metrics
        if 'per_class' in result.metrics:
            print("Per-class F1:")
            for label, metrics in result.metrics['per_class'].items():
                print(f"  {label}: {metrics['f1']:.3f}")

        print("-" * 40)


def main():
    """Main validation runner function."""
    parser = argparse.ArgumentParser(description="Run scorer validation")
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Paths to validation dataset files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/validation",
        help="Output directory for results"
    )
    parser.add_argument(
        "--scorer-configs",
        nargs="+",
        help="Paths to scorer config files (optional)"
    )

    args = parser.parse_args()

    # Load datasets
    datasets = []
    for dataset_path in args.datasets:
        path = Path(dataset_path)
        if path.exists():
            print(f"Loading dataset: {path}")
            dataset = load_dataset_from_file(path)
            datasets.append(dataset)
            print(f"  Loaded {len(dataset.samples)} samples")
        else:
            print(f"Warning: Dataset file not found: {path}")

    if not datasets:
        print("No valid datasets found. Exiting.")
        return

    # Create scorers
    scorers = create_sample_scorers()
    print(f"\nUsing {len(scorers)} scorer(s): {[s.name for s in scorers]}")

    # Create evaluation runner
    runner = EvaluationRunner(scorers)

    # Run evaluations
    print("\nRunning evaluations...")
    results = runner.evaluate_all(datasets)

    # Print summary
    print_evaluation_summary(results)

    # Save results
    output_dir = Path(args.output_dir)
    save_evaluation_results(results, output_dir)


if __name__ == "__main__":
    main()
