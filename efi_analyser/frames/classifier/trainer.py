"""Training utilities for frame classifiers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
import json

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
from efi_analyser.frames.types import FrameAssignment


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


@dataclass
class FrameClassifierArtifacts:
    """Trained classifier and optional predictions for downstream use."""

    model: Optional[FrameClassifierModel]
    predictions: List[Dict[str, object]]

    def save_predictions(self, path: Path) -> None:
        """Persist predictions to JSON at the given path."""
        path.write_text(json.dumps(list(self.predictions), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_predictions(cls, path: Path) -> "FrameClassifierArtifacts":
        """Load predictions from JSON; model is not restored here."""
        data = json.loads(path.read_text(encoding="utf-8"))
        preds: List[Dict[str, object]] = list(data)
        return cls(model=None, predictions=preds)


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

    def _build_hf_trainer(
        self,
        *,
        model: AutoModelForSequenceClassification,
        train_dataset: torch.utils.data.Dataset,  # type: ignore[name-defined]
        eval_dataset: Optional[torch.utils.data.Dataset],  # type: ignore[name-defined]
        compute_metrics: Optional[Callable[[Dict[str, np.ndarray]], Dict[str, float]]],
    ) -> Trainer:
        """Construct a Hugging Face Trainer with our TrainingArguments and callbacks.

        Centralizes Trainer setup so both `train` and `cross_validate` can reuse it.
        """
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
        if getattr(self.spec, "run_name", None):
            training_kwargs["run_name"] = self.spec.run_name  # type: ignore[index]
        if getattr(self.spec, "logging_dir", None):
            training_kwargs["logging_dir"] = self.spec.logging_dir  # type: ignore[index]
        if eval_dataset is not None:
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

        try:
            training_args = TrainingArguments(**training_kwargs)
        except TypeError:
            fallback_kwargs = dict(training_kwargs)
            for k in ("evaluation_strategy", "eval_steps", "save_strategy", "report_to", "load_best_model_at_end"):
                fallback_kwargs.pop(k, None)
            if eval_dataset is not None:
                fallback_kwargs["evaluate_during_training"] = True
            try:
                training_args = TrainingArguments(**fallback_kwargs)
            except TypeError:
                fallback_kwargs.pop("evaluate_during_training", None)
                training_args = TrainingArguments(**fallback_kwargs)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

        # Optional periodic evaluation mid-epoch
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
        return trainer

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

        trainer = self._build_hf_trainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )
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

    def cross_validate(
        self,
        label_set: FrameLabelSet,
        *,
        k_folds: int = 5,
        seed: int = 13,
        shuffle: bool = True,
        compute_metrics: Optional[Callable[[Dict[str, np.ndarray]], Dict[str, float]]] = None,
    ) -> Dict[str, Any]:
        """Run K-fold cross-validation and aggregate metrics.

        Returns a summary dict with mean/std for key metrics and per-fold gaps to help detect overfitting.
        """
        import math
        from pathlib import Path
        from dataclasses import replace

        n = len(label_set.passages)
        if n == 0:
            raise ValueError("Label set is empty; cannot run cross-validation")
        k = int(max(2, min(k_folds, n)))

        indices = list(range(n))
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)

        # Build fold index blocks
        fold_sizes = [n // k] * k
        for i in range(n % k):
            fold_sizes[i] += 1
        folds: List[List[int]] = []
        start = 0
        for size in fold_sizes:
            end = start + size
            folds.append(indices[start:end])
            start = end

        # Build default compute_metrics if not supplied
        if compute_metrics is None:
            compute_metrics = build_default_compute_metrics(
                label_order=list(label_set.label_order),
                eval_threshold=getattr(self.spec, "eval_threshold", 0.5),
                eval_top_k=getattr(self.spec, "eval_top_k", None),
            )

        # Prepare output directories
        out_root = Path(self.spec.output_dir) if self.spec.output_dir else Path("frame_classifier_runs")
        cv_dir = out_root / "cv"
        cv_dir.mkdir(parents=True, exist_ok=True)

        def subset(idxs: Sequence[int]) -> FrameLabelSet:
            return FrameLabelSet(
                schema=label_set.schema,
                passages=[label_set.passages[i] for i in idxs],
                source=label_set.source,
                metadata=label_set.metadata.copy(),
            )

        fold_records: List[Dict[str, Any]] = []
        for i in range(k):
            val_idx = folds[i]
            train_idx = [idx for j, f in enumerate(folds) if j != i for idx in f]
            train_set = subset(train_idx)
            val_set = subset(val_idx)

            fold_out = cv_dir / f"fold_{i+1}_of_{k}"
            # Copy spec and isolate outputs per fold
            try:
                new_spec = replace(self.spec, output_dir=str(fold_out))  # type: ignore[call-arg]
            except Exception:
                new_spec = FrameClassifierSpec(**asdict(self.spec))  # type: ignore[name-defined]
                new_spec.output_dir = str(fold_out)
            fold_trainer = FrameClassifierTrainer(new_spec)

            # Build model/datasets and trainer
            model = fold_trainer._build_model(num_labels=label_set.num_frames)
            train_ds = fold_trainer._build_dataset(train_set)
            val_ds = fold_trainer._build_dataset(val_set)
            hf_trainer = fold_trainer._build_hf_trainer(
                model=model, train_dataset=train_ds, eval_dataset=val_ds, compute_metrics=compute_metrics
            )

            print(f"[FrameTrainer][CV] Fold {i+1}/{k}: train={len(train_set.passages)} val={len(val_set.passages)}")
            hf_trainer.train()

            # Evaluate on validation and training to estimate generalization gap
            # Use predict() to apply same compute_metrics consistently
            pred_val = hf_trainer.predict(val_ds)
            val_metrics = compute_metrics((pred_val.predictions, pred_val.label_ids)) or {}
            pred_train = hf_trainer.predict(train_ds)
            train_metrics = compute_metrics((pred_train.predictions, pred_train.label_ids)) or {}

            # Extract key numbers
            macro_f1_val = float(val_metrics.get("macro_f1", val_metrics.get("eval_macro_f1", 0.0)))
            macro_f1_train = float(train_metrics.get("macro_f1", train_metrics.get("eval_macro_f1", 0.0)))
            gap = macro_f1_train - macro_f1_val

            record = {
                "fold": i + 1,
                "k": k,
                "n_train": len(train_set.passages),
                "n_val": len(val_set.passages),
                "macro_f1_train": macro_f1_train,
                "macro_f1_val": macro_f1_val,
                "generalization_gap": gap,
            }
            # Include a few other summary metrics if present
            for key in (
                "micro_f1",
                "macro_precision",
                "macro_recall",
                "micro_precision",
                "micro_recall",
            ):
                if key in val_metrics:
                    record[f"val_{key}"] = float(val_metrics[key])
                if key in train_metrics:
                    record[f"train_{key}"] = float(train_metrics[key])

            fold_records.append(record)

        # Aggregate
        def mean_std(values: List[float]) -> Dict[str, float]:
            arr = np.asarray(values, dtype=np.float32)
            return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)) if len(arr) > 1 else 0.0}

        macro_train = [r["macro_f1_train"] for r in fold_records]
        macro_val = [r["macro_f1_val"] for r in fold_records]
        gaps = [r["generalization_gap"] for r in fold_records]
        summary: Dict[str, Any] = {
            "k_folds": k,
            "n_examples": n,
            "macro_f1_train": mean_std(macro_train),
            "macro_f1_val": mean_std(macro_val),
            "generalization_gap": mean_std(gaps),
            "folds": fold_records,
        }

        # Persist summary
        try:
            import json, csv

            (cv_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            with open(cv_dir / "fold_metrics.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                header = list(fold_records[0].keys()) if fold_records else [
                    "fold",
                    "k",
                    "n_train",
                    "n_val",
                    "macro_f1_train",
                    "macro_f1_val",
                    "generalization_gap",
                ]
                writer.writerow(header)
                for rec in fold_records:
                    writer.writerow([rec.get(col, "") for col in header])
        except Exception:
            pass

        print(
            "[FrameTrainer][CV] macroF1 (val): "
            f"{summary['macro_f1_val']['mean']:.3f} ± {summary['macro_f1_val']['std']:.3f} | "
            "gap (train - val): "
            f"{summary['generalization_gap']['mean']:.3f}"
        )
        return summary

    # ------------------------------------------------------------------ high-level API
    def run(
        self,
        label_set: FrameLabelSet,
        *,
        assignments: Optional[Sequence[FrameAssignment]] = None,
        cv_folds: Optional[int] = None,
    ) -> FrameClassifierArtifacts:
        """Train, optionally cross-validate, and infer on provided samples.

        - Splits ``label_set`` into train/dev using a simple 90/10 policy
          (with a special case to ensure at least one dev example when possible).
        - Trains a classifier on the train split and evaluates on the dev split.
        - Optionally runs K-fold cross-validation for overfitting diagnostics.
        - If ``inference_samples`` are provided, runs inference over them and
          returns predictions keyed by passage_id.
        """
        if not label_set.passages:
            print("[FrameTrainer] Label set is empty; skipping training.")
            return FrameClassifierArtifacts(model=None, predictions=[])

        # Train/dev split
        n_total = len(label_set.passages)
        base_train_ratio = 0.9
        base_dev_ratio = 0.1
        if n_total >= 2 and int(n_total * base_dev_ratio) == 0:
            dev_ratio = 1.0 / n_total
            train_ratio = max(0.0, 1.0 - dev_ratio)
        else:
            train_ratio = base_train_ratio
            dev_ratio = base_dev_ratio
        train_set, dev_set, _ = label_set.split(train_ratio=train_ratio, dev_ratio=dev_ratio, seed=self.spec.seed)

        print(
            f"[FrameTrainer] Training classifier: {self.spec.model_name} "
            f"(epochs={self.spec.num_train_epochs}, lr={self.spec.learning_rate})\n"
            f"  Using {len(train_set.passages)} train / {len(dev_set.passages)} dev passages for evaluation"
        )
        model = self.train(train_set, eval_set=dev_set)
        print("[FrameTrainer] Training complete.")

        # Optional cross-validation
        cv_value = cv_folds if cv_folds is not None else getattr(self.spec, "cv_folds", None)
        try:
            k = int(cv_value) if cv_value is not None else 0
        except Exception:
            k = 0
        if k >= 2:
            print(f"[FrameTrainer] Running {k}-fold cross-validation for overfitting check…")
            try:
                self.cross_validate(label_set, k_folds=k, seed=self.spec.seed)
            except Exception as exc:
                print(f"[FrameTrainer] ⚠️ Cross-validation failed: {exc}")

        # Inference
        predictions: List[Dict[str, object]] = []
        if assignments:
            texts = [a.passage_text for a in assignments]
            if not texts:
                print("[FrameTrainer] ⚠️ No passages available for inference; skipping predictions.")
            else:
                batch_size = self.spec.batch_size
                print(
                    f"[FrameTrainer] Running classifier inference on {len(texts)} passages "
                    f"(batch size {batch_size})..."
                )
                probs = model.predict_proba_batch(texts, batch_size=batch_size)
                for assignment, prob_dict in zip(assignments, probs):
                    ordered = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
                    predictions.append(
                        {
                            "passage_id": assignment.passage_id,
                            "probabilities": prob_dict,
                            "top_frames": [fid for fid, _ in ordered[:3]],
                        }
                    )

        return FrameClassifierArtifacts(model=model, predictions=predictions)


__all__ = ["FrameClassifierTrainer", "FrameClassifierArtifacts"]
