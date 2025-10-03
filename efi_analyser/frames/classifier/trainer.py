"""Training utilities for frame classifiers."""

from __future__ import annotations

from dataclasses import asdict
from typing import Callable, Dict, Optional, Sequence

import numpy as np

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from efi_analyser.frames.classifier.dataset import FrameLabelSet
from efi_analyser.frames.classifier.model import FrameClassifierModel, FrameClassifierSpec


class _HFDataset(torch.utils.data.Dataset):  # type: ignore[name-defined]
    def __init__(
        self,
        texts: Sequence[str],
        labels: np.ndarray,
        tokenizer,
        max_length: int,
    ) -> None:
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


class FrameClassifierTrainer:
    """Train Hugging Face multi-label classifiers on frame datasets."""

    def __init__(self, spec: FrameClassifierSpec) -> None:
        self.spec = spec
        self.tokenizer = AutoTokenizer.from_pretrained(spec.model_name)

    def _build_model(self, num_labels: int) -> AutoModelForSequenceClassification:
        model = AutoModelForSequenceClassification.from_pretrained(
            self.spec.model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True,
        )
        if self.spec.freeze_base_model:
            for name, param in model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False
        return model

    def _build_dataset(self, label_set: FrameLabelSet) -> _HFDataset:
        labels = label_set.to_numpy()[0]
        texts = [passage.text for passage in label_set.passages]
        return _HFDataset(texts=texts, labels=labels, tokenizer=self.tokenizer, max_length=self.spec.max_length)

    def train(
        self,
        label_set: FrameLabelSet,
        eval_set: Optional[FrameLabelSet] = None,
        compute_metrics: Optional[Callable[[Dict[str, np.ndarray]], Dict[str, float]]] = None,
    ) -> FrameClassifierModel:
        train_dataset = self._build_dataset(label_set)
        eval_dataset = self._build_dataset(eval_set) if eval_set and len(eval_set.passages) > 0 else None

        model = self._build_model(num_labels=label_set.num_frames)

        training_kwargs = dict(
            output_dir=self.spec.output_dir,
            num_train_epochs=self.spec.num_train_epochs,
            per_device_train_batch_size=self.spec.batch_size,
            per_device_eval_batch_size=self.spec.batch_size,
            learning_rate=self.spec.learning_rate,
            weight_decay=self.spec.weight_decay,
            warmup_ratio=self.spec.warmup_ratio,
            gradient_accumulation_steps=self.spec.gradient_accumulation_steps,
            fp16=self.spec.fp16,
            seed=self.spec.seed,
            logging_steps=10,
            report_to=[],
        )
        if eval_dataset is not None:
            training_kwargs["evaluation_strategy"] = "epoch"
        else:
            training_kwargs["evaluation_strategy"] = "no"
        training_kwargs["save_strategy"] = "no"
        training_kwargs["load_best_model_at_end"] = False

        try:
            training_args = TrainingArguments(**training_kwargs)
        except TypeError:  # pragma: no cover - older transformers compatibility
            training_kwargs.pop("evaluation_strategy", None)
            training_kwargs.pop("save_strategy", None)
            if eval_dataset is not None:
                training_kwargs["evaluate_during_training"] = True
            training_args = TrainingArguments(**training_kwargs)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )
        trainer.train()

        return FrameClassifierModel(
            schema=label_set.schema,
            label_order=label_set.label_order,
            model=trainer.model,
            tokenizer=self.tokenizer,
            spec=self.spec,
        )


__all__ = ["FrameClassifierTrainer"]
