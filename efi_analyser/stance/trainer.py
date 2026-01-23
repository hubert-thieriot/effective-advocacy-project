"""Training utilities for stance classifiers."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import json

import numpy as np
import torch
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from efi_analyser.stance.types import StanceLabelSet


@dataclass
class StanceClassifierSpec:
    """Configuration for training/inference of the stance classifier."""

    model_name: str = "distilbert-base-uncased"
    max_length: int = 384
    learning_rate: float = 5e-5
    num_train_epochs: float = 3.0
    batch_size: int = 16
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    fp16: bool = False
    seed: int = 13
    output_dir: str = "stance_classifier_runs"
    freeze_base_model: bool = False
    # Logging/reporting
    report_to: List[str] = field(default_factory=list)
    logging_dir: Optional[str] = None
    run_name: Optional[str] = None


class _StanceDataset(torch.utils.data.Dataset):  # type: ignore[name-defined]
    def __init__(self, texts: Sequence[str], labels: np.ndarray, tokenizer, max_length: int) -> None:
        self.texts = list(texts)
        self.labels = torch.from_numpy(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        item = {k: torch.tensor(v) for k, v in encoded.items()}
        item["labels"] = self.labels[idx]
        return item


class StanceClassifierModel:
    """Wrapper around a trained stance classifier."""

    def __init__(
        self,
        label_order: Sequence[str],
        model: AutoModelForSequenceClassification,
        tokenizer,
        spec: StanceClassifierSpec,
    ) -> None:
        self.label_order = list(label_order)
        self.model = model
        self.tokenizer = tokenizer
        self.spec = spec

    def predict_proba_batch(
        self,
        texts: Sequence[str],
        batch_size: int = 8,
        device: Optional[str] = None,
    ) -> List[Dict[str, float]]:
        self.model.eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        outputs: List[Dict[str, float]] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]
                encoded = self.tokenizer(
                    list(batch_texts),
                    padding=True,
                    truncation=True,
                    max_length=self.spec.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                logits = self.model(**encoded).logits
                probs = softmax(logits, dim=-1).cpu().numpy()
                for row in probs:
                    outputs.append({label: float(score) for label, score in zip(self.label_order, row)})
        return outputs

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        spec_dict = self.spec.__dict__
        valid_spec_fields = {f.name for f in fields(StanceClassifierSpec)}
        filtered = {k: v for k, v in spec_dict.items() if k in valid_spec_fields}
        payload = {"label_order": self.label_order, "spec": filtered}
        (output_dir / "stance_classifier.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, output_dir: Path) -> "StanceClassifierModel":
        payload = json.loads((output_dir / "stance_classifier.json").read_text(encoding="utf-8"))
        label_order = payload.get("label_order", [])
        spec_dict = payload.get("spec", {})
        valid_spec_fields = {f.name for f in fields(StanceClassifierSpec)}
        filtered = {k: v for k, v in spec_dict.items() if k in valid_spec_fields}
        spec = StanceClassifierSpec(**filtered)
        model = AutoModelForSequenceClassification.from_pretrained(output_dir, use_safetensors=True)
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        return cls(label_order=label_order, model=model, tokenizer=tokenizer, spec=spec)


class StanceClassifierTrainer:
    """Trainer for stance classifiers."""

    def __init__(self, spec: StanceClassifierSpec, label_order: Sequence[str]) -> None:
        self.spec = spec
        self.label_order = list(label_order)
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_name)

    def train(self, label_set: StanceLabelSet) -> StanceClassifierModel:
        labels = label_set.to_numpy()
        texts = label_set.texts()
        dataset = _StanceDataset(texts=texts, labels=labels, tokenizer=self.tokenizer, max_length=self.spec.max_length)

        model = AutoModelForSequenceClassification.from_pretrained(
            self.spec.model_name,
            num_labels=len(self.label_order),
            problem_type="single_label_classification",
            ignore_mismatched_sizes=True,
            use_safetensors=True,
        )
        if self.spec.freeze_base_model:
            for name, param in model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False

        training_kwargs = dict(
            output_dir=self.spec.output_dir,
            num_train_epochs=self.spec.num_train_epochs,
            per_device_train_batch_size=self.spec.batch_size,
            learning_rate=self.spec.learning_rate,
            weight_decay=self.spec.weight_decay,
            warmup_ratio=self.spec.warmup_ratio,
            gradient_accumulation_steps=self.spec.gradient_accumulation_steps,
            fp16=self.spec.fp16,
            seed=self.spec.seed,
            logging_steps=10,
            report_to=list(self.spec.report_to or []),
            save_strategy="no",
            evaluation_strategy="no",
        )
        if self.spec.run_name:
            training_kwargs["run_name"] = self.spec.run_name
        if self.spec.logging_dir:
            training_kwargs["logging_dir"] = self.spec.logging_dir

        training_args = TrainingArguments(**training_kwargs)
        trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
        trainer.train()

        return StanceClassifierModel(
            label_order=self.label_order,
            model=model,
            tokenizer=self.tokenizer,
            spec=self.spec,
        )


__all__ = ["StanceClassifierSpec", "StanceClassifierModel", "StanceClassifierTrainer"]
