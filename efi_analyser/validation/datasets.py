"""
Concrete implementations of validation datasets.
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Union, Dict, Any, List
from dataclasses import dataclass

from .types import ValidationDataset, ValidationSample, TaskType


@dataclass
class NLIDataset(ValidationDataset):
    """Dataset for NLI (Natural Language Inference) validation."""

    def __init__(self, name: str = "nli_validation"):
        super().__init__(name, TaskType.NLI)

    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save NLI dataset to JSON file."""
        import json
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        data = []
        for sample in self.samples:
            data.append({
                'sample_id': sample.sample_id,
                'text_a': sample.text_a,
                'text_b': sample.text_b,
                'gold_label': sample.gold_label,
                'metadata': sample.metadata
            })

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_from_file(self, file_path: Union[str, Path]) -> None:
        """Load NLI dataset from JSON or CSV file.

        Expected formats:
        JSON: [{"text_a": "...", "text_b": "...", "gold_label": "entails|contradicts|neutral", ...}, ...]
        CSV: text_a,text_b,gold_label columns
        """
        file_path = Path(file_path)

        if file_path.suffix.lower() == '.json':
            self._load_from_json(file_path)
        elif file_path.suffix.lower() == '.csv':
            self._load_from_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def _load_from_json(self, file_path: Path) -> None:
        """Load from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for i, item in enumerate(data):
            sample = ValidationSample(
                sample_id=item.get('sample_id', f'sample_{i}'),
                text_a=item['text_a'],
                text_b=item['text_b'],
                gold_label=item['gold_label'],
                metadata=item.get('metadata', {})
            )
            self.add_sample(sample)

    def _load_from_csv(self, file_path: Path) -> None:
        """Load from CSV file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for i, row in enumerate(reader):
                sample = ValidationSample(
                    sample_id=row.get('sample_id', f'sample_{i}'),
                    text_a=row['text_a'],
                    text_b=row['text_b'],
                    gold_label=row['gold_label'],
                    metadata={k: v for k, v in row.items()
                             if k not in ['sample_id', 'text_a', 'text_b', 'gold_label']}
                )
                self.add_sample(sample)


@dataclass
class StanceDataset(ValidationDataset):
    """Dataset for Stance detection validation."""

    def __init__(self, name: str = "stance_validation"):
        super().__init__(name, TaskType.STANCE)

    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save Stance dataset to JSON file."""
        import json
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        data = []
        for sample in self.samples:
            data.append({
                'sample_id': sample.sample_id,
                'text_a': sample.text_a,
                'text_b': sample.text_b,
                'gold_label': sample.gold_label,
                'metadata': sample.metadata
            })

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_from_file(self, file_path: Union[str, Path]) -> None:
        """Load Stance dataset from JSON or CSV file.

        Expected formats:
        JSON: [{"text_a": "...", "text_b": "...", "gold_label": "pro|anti|neutral|uncertain", ...}, ...]
        CSV: text_a,text_b,gold_label columns
        """
        file_path = Path(file_path)

        if file_path.suffix.lower() == '.json':
            self._load_from_json(file_path)
        elif file_path.suffix.lower() == '.csv':
            self._load_from_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def _load_from_json(self, file_path: Path) -> None:
        """Load from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for i, item in enumerate(data):
            sample = ValidationSample(
                sample_id=item.get('sample_id', f'sample_{i}'),
                text_a=item['text_a'],
                text_b=item['text_b'],
                gold_label=item['gold_label'],
                metadata=item.get('metadata', {})
            )
            self.add_sample(sample)

    def _load_from_csv(self, file_path: Path) -> None:
        """Load from CSV file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for i, row in enumerate(reader):
                sample = ValidationSample(
                    sample_id=row.get('sample_id', f'sample_{i}'),
                    text_a=row['text_a'],
                    text_b=row['text_b'],
                    gold_label=row['gold_label'],
                    metadata={k: v for k, v in row.items()
                             if k not in ['sample_id', 'text_a', 'text_b', 'gold_label']}
                )
                self.add_sample(sample)


# Convenience functions for creating datasets
def create_nli_dataset_from_dicts(data: List[Dict[str, Any]], name: str = "nli_dataset") -> NLIDataset:
    """Create NLI dataset from list of dictionaries."""
    dataset = NLIDataset(name)
    for i, item in enumerate(data):
        sample = ValidationSample(
            sample_id=item.get('sample_id', f'sample_{i}'),
            text_a=item['text_a'],
            text_b=item['text_b'],
            gold_label=item['gold_label'],
            metadata=item.get('metadata', {})
        )
        dataset.add_sample(sample)
    return dataset


def create_stance_dataset_from_dicts(data: List[Dict[str, Any]], name: str = "stance_dataset") -> StanceDataset:
    """Create Stance dataset from list of dictionaries."""
    dataset = StanceDataset(name)
    for i, item in enumerate(data):
        sample = ValidationSample(
            sample_id=item.get('sample_id', f'sample_{i}'),
            text_a=item['text_a'],
            text_b=item['text_b'],
            gold_label=item['gold_label'],
            metadata=item.get('metadata', {})
        )
        dataset.add_sample(sample)
    return dataset
