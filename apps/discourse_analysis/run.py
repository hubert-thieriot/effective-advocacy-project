#!/usr/bin/env python3
"""Command-line entry point for discourse analysis."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from apps.discourse_analysis.config import DiscourseAnalysisConfig, load_config
from apps.discourse_analysis.pipeline import DiscourseAnalysisPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discourse analysis pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML configuration file",
        required=True,
    )
    return parser.parse_args()


def run_pipeline(config: DiscourseAnalysisConfig, output_dir: Path | None = None):
    if output_dir is None:
        output_dir = config.results_dir or Path("results/discourse_analysis")
    pipeline = DiscourseAnalysisPipeline(config, output_dir)
    return pipeline.run()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    config = load_config(args.config)
    run_pipeline(config)


if __name__ == "__main__":
    main()
