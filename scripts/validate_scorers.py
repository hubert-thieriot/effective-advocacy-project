#!/usr/bin/env python3
"""
CLI script for validating scorers on datasets with report generation.

Usage:
    # Run with default scorers on sample dataset
    python scripts/validate_scorers.py

    # Run specific scorers
    python scripts/validate_scorers.py --scorers nli_hf nli_llm

    # Use custom dataset
    python scripts/validate_scorers.py --dataset path/to/dataset.json

    # Generate reports
    python scripts/validate_scorers.py --generate-reports --output-dir my_results
"""

import sys
import argparse
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from efi_analyser.scorers import NLIHFScorer, NLILLMScorer, LLMScorerConfig, StanceHFScorer, StanceLLMScorer
from efi_analyser.validation import NLIDataset, StanceDataset, EvaluationRunner
from efi_analyser.report_generator import ValidationReportGenerator
from efi_analyser.types import ReportConfig


def create_scorers(scorer_names: List[str], verbose: bool = False):
    """Create scorers based on names."""
    scorers = []

    for name in scorer_names:
        if name == "nli_hf":
            scorers.append(NLIHFScorer(name="nli_hf"))
        elif name == "nli_llm_phi3":
            # Disable cache for validation to ensure fresh results
            config = LLMScorerConfig(model="phi3:3.8b", ignore_cache=True, verbose=verbose)
            scorers.append(NLILLMScorer(name="nli_llm_phi3", config=config))
        elif name == "nli_llm_gemma":
            # Disable cache for validation to ensure fresh results
            config = LLMScorerConfig(model="gemma3:4b", ignore_cache=True, verbose=verbose)
            scorers.append(NLILLMScorer(name="nli_llm_gemma", config=config))
        elif name == "stance_hf":
            scorers.append(StanceHFScorer(name="stance_hf"))
        elif name == "stance_llm":
            # Disable cache for validation to ensure fresh results
            config = LLMScorerConfig(model="phi3:3.8b", ignore_cache=True, verbose=verbose)
            scorers.append(StanceLLMScorer(name="stance_llm", config=config))
        else:
            print(f"Warning: Unknown scorer '{name}', skipping")

    return scorers


def main():
    parser = argparse.ArgumentParser(description="Validate scorers on datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(project_root / "efi_analyser" / "validation" / "data" / "sample_nli_dataset.json"),
        help="Path to validation dataset JSON file"
    )
    parser.add_argument(
        "--scorers",
        nargs="+",
        default=["nli_hf", "nli_llm_phi3", "nli_llm_gemma"],
        choices=["nli_hf", "nli_llm_phi3", "nli_llm_gemma", "stance_hf", "stance_llm"],
        help="Scorers to evaluate"
    )
    parser.add_argument(
        "--no-generate-reports",
        action="store_true",
        help="Do not generate HTML reports after evaluation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(project_root / "results" / "validation"),
        help="Output directory for reports"
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["html"],
        choices=["html", "csv", "json"],
        help="Report formats to generate"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output showing inference progress"
    )

    args = parser.parse_args()

    print("🔬 EFI Scorer Validation")
    print("=" * 40)

    # Load dataset
    dataset_path = Path(args.dataset)
    print(f"Loading dataset from: {dataset_path}")

    if dataset_path.name.endswith("_nli_") or "nli" in str(dataset_path):
        dataset = NLIDataset("validation_dataset")
    elif dataset_path.name.endswith("_stance_") or "stance" in str(dataset_path):
        dataset = StanceDataset("validation_dataset")
    else:
        # Could extend for other dataset types
        print(f"Error: Unsupported dataset type for {dataset_path}")
        return

    dataset.load_from_file(dataset_path)
    print(f"Loaded {len(dataset.samples)} validation samples")

    # Create scorers
    print(f"\nSetting up scorers: {args.scorers}")
    scorers = create_scorers(args.scorers, verbose=args.verbose)

    if not scorers:
        print("Error: No valid scorers specified")
        return

    for scorer in scorers:
        print(f"✓ {scorer.name}")

    # Run evaluation
    print("\nRunning validation...")
    runner = EvaluationRunner(scorers, verbose=args.verbose)
    results = runner.evaluate_all([dataset])

    # Print summary
    print("\n📊 RESULTS SUMMARY:")
    print("-" * 30)
    for result in results:
        print(f"{result.scorer_name}: Accuracy={result.metrics['accuracy']:.3f}, F1={result.metrics.get('macro_f1', 0.0):.3f}")

    # Find best performer
    if results:
        best_result = max(results, key=lambda x: x.metrics['accuracy'])
        print(f"\n🏆 BEST PERFORMER: {best_result.scorer_name} (Accuracy: {best_result.metrics['accuracy']:.3f})")

    # Generate reports if requested
    if not args.no_generate_reports:
        print(f"\nGenerating reports to: {args.output_dir}")
        report_config = ReportConfig(
            output_dir=Path(args.output_dir),
            formats=args.formats,
            include_metadata=True,
            include_timing=True
        )

        report_generator = ValidationReportGenerator(report_config)
        report_files = report_generator.generate_all(results)

        print("\n📋 REPORTS GENERATED:")
        for format_name, file_path in report_files.items():
            print(f"   {format_name.upper()}: {file_path}")

    print("\n✅ Validation completed!")


if __name__ == "__main__":
    main()
