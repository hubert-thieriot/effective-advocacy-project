"""Training utilities for frame classifiers."""

from __future__ import annotations

from dataclasses import asdict
from typing import Callable, Dict, Optional, Sequence, Any, List

import numpy as np

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

from efi_analyser.frames.classifier.dataset import FrameLabelSet
from efi_analyser.frames.classifier.model import FrameClassifierModel, FrameClassifierSpec
from efi_analyser.frames.classifier.reporting import (
    build_default_compute_metrics,
    PeriodicEvaluatorCallback,
    EpochEndEvaluateCallback,
    PerFrameMetricsCallback,
)


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
            use_safetensors=True,  # Use safetensors to avoid torch version requirements
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

        # If no compute_metrics provided, use the shared implementation from reporting.py
        # that supports threshold- and top-k-based multi-label metrics.
        if compute_metrics is None and eval_dataset is not None:
            compute_metrics = build_default_compute_metrics(
                label_order=list(label_set.label_order),
                eval_threshold=getattr(self.spec, "eval_threshold", 0.5),
                eval_top_k=getattr(self.spec, "eval_top_k", None),
            )

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
            report_to=list(getattr(self.spec, "report_to", []) or []),
        )
        # Optional run name (used by WandB/TensorBoard integrations)
        if getattr(self.spec, "run_name", None):
            training_kwargs["run_name"] = self.spec.run_name  # type: ignore[index]
        # Optional explicit logging directory
        if getattr(self.spec, "logging_dir", None):
            training_kwargs["logging_dir"] = self.spec.logging_dir  # type: ignore[index]
        if eval_dataset is not None:
            # Prefer steps-based eval when eval_steps is configured, otherwise at each epoch
            eval_steps_val = getattr(self.spec, "eval_steps", None)
            if isinstance(eval_steps_val, int) and eval_steps_val > 0:
                training_kwargs["evaluation_strategy"] = "steps"
                training_kwargs["eval_steps"] = int(eval_steps_val)
            else:
                training_kwargs["evaluation_strategy"] = "epoch"
        else:
            training_kwargs["evaluation_strategy"] = "no"
        training_kwargs["save_strategy"] = "no"
        training_kwargs["load_best_model_at_end"] = False

        # Build TrainingArguments with compatibility for older transformers versions
        try:
            training_args = TrainingArguments(**training_kwargs)
        except TypeError:
            # Fallback for older versions: drop newer keys and try legacy flag
            fallback_kwargs = dict(training_kwargs)
            # Remove keys older versions might not recognize
            for k in ("evaluation_strategy", "eval_steps", "save_strategy", "report_to", "load_best_model_at_end"):
                fallback_kwargs.pop(k, None)
            # Some very old versions used 'evaluate_during_training'
            if eval_dataset is not None:
                fallback_kwargs["evaluate_during_training"] = True
            try:
                training_args = TrainingArguments(**fallback_kwargs)
            except TypeError:
                # Final attempt: remove legacy flag if also unsupported
                fallback_kwargs.pop("evaluate_during_training", None)
                training_args = TrainingArguments(**fallback_kwargs)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )
        # Attach periodic eval callback for mid-epoch evaluation if requested
        if eval_dataset is not None and getattr(self.spec, "eval_steps", None):
            try:
                steps = int(self.spec.eval_steps) if self.spec.eval_steps else 0
            except Exception:
                steps = 0
            if steps > 0:
                print(f"[FrameTrainer] Adding periodic evaluation every {steps} steps")
                try:
                    trainer.add_callback(PeriodicEvaluatorCallback(steps))
                except Exception:
                    pass
        # Ensure eval at each epoch end when using epoch-level evaluation
        if eval_dataset is not None and not getattr(self.spec, "eval_steps", None):
            try:
                print("[FrameTrainer] Enabling evaluation at end of each epoch via callback")
                trainer.add_callback(EpochEndEvaluateCallback())
            except Exception:
                pass
        # Attach callback to export per-epoch metrics and charts when we have eval
        if eval_dataset is not None and compute_metrics is not None:
            try:
                id_to_name_cb: Dict[str, str] = {}
                for frame in label_set.schema.frames:
                    id_to_name_cb[frame.frame_id] = (frame.short_name or frame.name or frame.frame_id).strip()
                trainer.add_callback(
                    PerFrameMetricsCallback(
                        out_dir=self.spec.output_dir,
                        label_ids=label_set.label_order,
                        id_to_name=id_to_name_cb,
                    )
                )
            except Exception:
                # Non-fatal: if logging callback fails, keep training working.
                pass
        print(
            "[FrameTrainer] Starting training with settings: "
            f"epochs={self.spec.num_train_epochs}, batch={self.spec.batch_size}, lr={self.spec.learning_rate}, "
            f"eval={'enabled' if eval_dataset is not None else 'disabled'}, eval_steps={getattr(self.spec, 'eval_steps', None)}"
        )
        trainer.train()
        # Ensure at least one evaluation runs to emit metrics/charts in environments
        # where per-epoch evaluation isn't triggered by TrainingArguments compatibility.
        if eval_dataset is not None:
            try:
                trainer.evaluate()
            except Exception:
                pass

        return FrameClassifierModel(
            schema=label_set.schema,
            label_order=label_set.label_order,
            model=trainer.model,
            tokenizer=self.tokenizer,
            spec=self.spec,
        )


__all__ = ["FrameClassifierTrainer"]
