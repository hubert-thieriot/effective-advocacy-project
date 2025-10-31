"""Reporting utilities and callbacks for frame classifier training.

Separates metric computation and logging from the trainer logic to keep
efi_analyser.frames.classifier.trainer lean.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import numpy as np

try:
    from transformers import TrainerCallback  # type: ignore
except Exception:  # pragma: no cover - transformers not available in some contexts
    class TrainerCallback:  # type: ignore
        pass


def build_default_compute_metrics(
    *,
    label_order: Sequence[str],
    eval_threshold: float = 0.5,
    eval_top_k: int | None = None,
) -> Callable[[Any], Dict[str, float]]:
    """Factory producing a compute_metrics function for multi-label problems.

    Computes per-frame precision/recall/F1 using:
    - Thresholded predictions vs labels (threshold applies to both, default 0.5)
    - Optional top-k predictions vs top-k labels (if eval_top_k is provided)
    Returns a flat scalar dict suitable for HF Trainer logging.
    """

    labels = list(label_order)

    def compute_from_binary(pred_bin: np.ndarray, gold_bin: np.ndarray, prefix: str = "") -> Dict[str, float]:
        tp_total = 0
        fp_total = 0
        fn_total = 0
        precisions: List[float] = []
        recalls: List[float] = []
        f1s: List[float] = []
        out: Dict[str, float] = {}
        for j, fid in enumerate(labels):
            pred_col = pred_bin[:, j]
            gold_col = gold_bin[:, j]
            tp = int(np.sum((pred_col == 1) & (gold_col == 1)))
            fp = int(np.sum((pred_col == 1) & (gold_col == 0)))
            fn = int(np.sum((pred_col == 0) & (gold_col == 1)))
            support = int(np.sum(gold_col == 1))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            tp_total += tp
            fp_total += fp
            fn_total += fn
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

            key = (prefix + "/") if prefix else ""
            out[f"{key}per_frame/precision/{fid}"] = float(precision)
            out[f"{key}per_frame/recall/{fid}"] = float(recall)
            out[f"{key}per_frame/f1/{fid}"] = float(f1)
            out[f"{key}per_frame/support/{fid}"] = float(support)

        macro_precision = float(np.mean(precisions)) if precisions else 0.0
        macro_recall = float(np.mean(recalls)) if recalls else 0.0
        macro_f1 = float(np.mean(f1s)) if f1s else 0.0
        micro_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
        micro_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if (micro_precision + micro_recall) > 0
            else 0.0
        )
        key = (prefix + "/") if prefix else ""
        out.update(
            {
                f"{key}macro_precision": float(macro_precision),
                f"{key}macro_recall": float(macro_recall),
                f"{key}macro_f1": float(macro_f1),
                f"{key}micro_precision": float(micro_precision),
                f"{key}micro_recall": float(micro_recall),
                f"{key}micro_f1": float(micro_f1),
            }
        )
        return out

    def _compute_metrics(eval_pred: Any) -> Dict[str, float]:
        # eval_pred may be a tuple or EvalPrediction
        if isinstance(eval_pred, tuple) and len(eval_pred) >= 2:
            logits = eval_pred[0]
            labels_arr = eval_pred[1]
        else:
            logits = getattr(eval_pred, "predictions")
            labels_arr = getattr(eval_pred, "label_ids")

        logits = np.asarray(logits)
        labels_a = np.asarray(labels_arr)

        # Convert logits to probabilities
        probs = 1.0 / (1.0 + np.exp(-logits))

        result: Dict[str, float] = {}
        # Threshold-based
        thr = float(eval_threshold)
        thr_pred = (probs >= thr).astype(np.int32)
        thr_gold = (labels_a >= thr).astype(np.int32)
        result.update(compute_from_binary(thr_pred, thr_gold))
        result["threshold"] = float(thr)

        # Top-k based (optional)
        if isinstance(eval_top_k, int) and eval_top_k > 0:
            k = min(eval_top_k, probs.shape[1])
            pred_topk = np.zeros_like(probs, dtype=np.int32)
            gold_topk = np.zeros_like(labels_a, dtype=np.int32)
            pred_idx = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
            gold_idx = np.argpartition(-labels_a, kth=k - 1, axis=1)[:, :k]
            for i in range(probs.shape[0]):
                pred_topk[i, pred_idx[i]] = 1
                gold_topk[i, gold_idx[i]] = 1
            result.update(compute_from_binary(pred_topk, gold_topk, prefix=f"topk{k}"))
        return result

    return _compute_metrics


class PeriodicEvaluatorCallback(TrainerCallback):
    """Trigger evaluation every N steps regardless of TrainingArguments support.

    This is useful on older transformers versions or when you want mid-epoch
    metrics without switching the global evaluation strategy.
    """

    def __init__(self, every_n_steps: int) -> None:
        if every_n_steps <= 0:
            raise ValueError("every_n_steps must be positive")
        self.every = int(every_n_steps)

    def on_step_end(self, args, state, control, **kwargs):  # type: ignore[override]
        try:
            step = int(getattr(state, "global_step", 0))
            if step > 0 and (step % self.every == 0):
                print(f"[FrameTrainer] Periodic evaluation at step {step}…")
                # Return control to trigger evaluation
                control.should_evaluate = True
        except Exception:
            pass
        return control


class PerFrameMetricsCallback(TrainerCallback):
    """Persist per-epoch metrics and produce simple charts/CSV.

    Handles HF Trainer's eval prefixing (e.g., `eval_macro_f1`) by
    transparently reading with/without the `eval_` prefix.
    """

    def __init__(self, out_dir: str, label_ids: Sequence[str], id_to_name: Dict[str, str]):
        from datetime import datetime
        import csv
        import json

        self.Path = Path
        self.json = json
        self.csv = csv
        self.datetime = datetime

        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.label_ids = list(label_ids)
        self.id_to_name = dict(id_to_name)
        self.history: List[Dict[str, float]] = []
        self.csv_path = self.out_dir / "metrics_per_epoch.csv"
        self.jsonl_path = self.out_dir / "metrics_history.jsonl"
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                header = [
                    "timestamp",
                    "epoch",
                    "macro_precision",
                    "macro_recall",
                    "macro_f1",
                    "micro_precision",
                    "micro_recall",
                    "micro_f1",
                ]
                header += [f"f1_{fid}" for fid in self.label_ids]
                writer.writerow(header)
        self.topk_csv_paths: Dict[str, Path] = {}

        def _get(metrics: Dict[str, float], key: str) -> float:
            if key in metrics:
                return float(metrics[key])
            eval_key = f"eval_{key}"
            return float(metrics.get(eval_key, 0.0))

        self._get_metric = _get

        # Optional plotting
        try:
            import matplotlib.pyplot as plt  # type: ignore

            self._plt = plt
        except Exception:
            self._plt = None

    def _append_records(self, epoch: int, metrics: Dict[str, float]) -> None:
        record = {"epoch": float(epoch)}
        for k, v in metrics.items():
            try:
                record[k] = float(v)
            except Exception:
                continue
        self.history.append(record)
        payload = {"timestamp": self.datetime.utcnow().isoformat(), **record}
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(self.json.dumps(payload) + "\n")
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = self.csv.writer(f)
            row = [
                self.datetime.utcnow().isoformat(),
                epoch,
                self._get_metric(metrics, "macro_precision"),
                self._get_metric(metrics, "macro_recall"),
                self._get_metric(metrics, "macro_f1"),
                self._get_metric(metrics, "micro_precision"),
                self._get_metric(metrics, "micro_recall"),
                self._get_metric(metrics, "micro_f1"),
            ]
            row += [self._get_metric(metrics, f"per_frame/f1/{fid}") for fid in self.label_ids]
            writer.writerow(row)

        # Additionally export top-k CSVs if present
        prefixes = set()
        for key in metrics.keys():
            if key.startswith("topk"):
                prefix = key.split("/", 1)[0]
                prefixes.add(prefix)
        for prefix in sorted(prefixes):
            csv_path = self.topk_csv_paths.get(prefix)
            if csv_path is None:
                csv_path = self.out_dir / f"metrics_per_epoch_{prefix}.csv"
                self.topk_csv_paths[prefix] = csv_path
                if not csv_path.exists():
                    with open(csv_path, "w", newline="", encoding="utf-8") as f:
                        writer = self.csv.writer(f)
                        header = [
                            "timestamp",
                            "epoch",
                            f"{prefix}/macro_precision",
                            f"{prefix}/macro_recall",
                            f"{prefix}/macro_f1",
                            f"{prefix}/micro_precision",
                            f"{prefix}/micro_recall",
                            f"{prefix}/micro_f1",
                        ]
                        header += [f"{prefix}/per_frame/f1/{fid}" for fid in self.label_ids]
                        writer.writerow(header)
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = self.csv.writer(f)
                row = [
                    self.datetime.utcnow().isoformat(),
                    epoch,
                    self._get_metric(metrics, f"{prefix}/macro_precision"),
                    self._get_metric(metrics, f"{prefix}/macro_recall"),
                    self._get_metric(metrics, f"{prefix}/macro_f1"),
                    self._get_metric(metrics, f"{prefix}/micro_precision"),
                    self._get_metric(metrics, f"{prefix}/micro_recall"),
                    self._get_metric(metrics, f"{prefix}/micro_f1"),
                ]
                row += [self._get_metric(metrics, f"{prefix}/per_frame/f1/{fid}") for fid in self.label_ids]
                writer.writerow(row)

    def _plot(self) -> None:
        if self._plt is None or not self.history:
            return
        plt = self._plt
        epochs = sorted({int(round(item.get("epoch", 0))) for item in self.history})
        for metric_name, fname in (
            ("f1", "per_frame_f1_by_epoch.png"),
            ("precision", "per_frame_precision_by_epoch.png"),
            ("recall", "per_frame_recall_by_epoch.png"),
        ):
            # Adjust figure size based on number of frames
            num_frames = len(self.label_ids)
            fig_width = max(10, num_frames * 1.2)
            fig_height = max(6, 4 + (num_frames * 0.3))
            plt.figure(figsize=(fig_width, fig_height))
            
            for fid in self.label_ids:
                ys: List[float] = []
                for e in epochs:
                    rec = None
                    for item in self.history:
                        if int(round(item.get("epoch", 0))) == e:
                            rec = item
                            break
                    val = 0.0
                    if rec is not None:
                        if f"per_frame/{metric_name}/{fid}" in rec:
                            val = float(rec.get(f"per_frame/{metric_name}/{fid}", 0.0))
                        elif f"eval_per_frame/{metric_name}/{fid}" in rec:
                            val = float(rec.get(f"eval_per_frame/{metric_name}/{fid}", 0.0))
                    ys.append(val)
                label = self.id_to_name.get(fid, fid)
                plt.plot(epochs, ys, marker="o", label=label)
            
            plt.title(f"Per-frame {metric_name} over epochs", fontsize=14, fontweight='bold')
            plt.xlabel("Epoch", fontsize=12)
            plt.ylabel(metric_name.capitalize(), fontsize=12)
            plt.ylim(0.0, 1.0)
            plt.grid(True, alpha=0.3)
            
            # Adjust legend based on number of frames
            if num_frames <= 4:
                ncol = 1
                loc = 'best'
            elif num_frames <= 8:
                ncol = 2
                loc = 'upper left'
            else:
                ncol = 3
                loc = 'upper left'
            
            # Place legend outside if it would overlap too much
            if num_frames > 6:
                plt.legend(fontsize=9, ncol=ncol, loc=loc, bbox_to_anchor=(1.02, 1), borderaxespad=0)
            else:
                plt.legend(fontsize=9, ncol=ncol, loc=loc)
            
            plt.tight_layout()
            plt.savefig(self.out_dir / fname, bbox_inches='tight')
            plt.close()

    def on_evaluate(self, args, state, control, metrics, **kwargs):  # type: ignore[override]
        try:
            import math

            epoch_val = state.epoch if state.epoch is not None else 0.0
            epoch = int(math.floor(epoch_val + 1e-9))
            self._append_records(epoch, metrics or {})
            self._plot()
            # Console summary with top/bottom frames by F1
            per_f1 = [(fid, self._get_metric(metrics, f"per_frame/f1/{fid}")) for fid in self.label_ids]
            per_f1.sort(key=lambda x: x[1], reverse=True)
            top = ", ".join(f"{self.id_to_name.get(fid, fid)}={score:.2f}" for fid, score in per_f1[:3])
            bottom = ", ".join(f"{self.id_to_name.get(fid, fid)}={score:.2f}" for fid, score in per_f1[-3:])
            print(
                f"[FrameTrainer] Eval epoch {epoch}: macroF1={self._get_metric(metrics, 'macro_f1'):.3f} | top F1: {top} | bottom F1: {bottom}"
            )
        except Exception:
            pass


__all__ = [
    "build_default_compute_metrics",
    "PeriodicEvaluatorCallback",
    "EpochEndEvaluateCallback",
    "PerFrameMetricsCallback",
]

class EpochEndEvaluateCallback(TrainerCallback):
    """Force an evaluation at the end of each epoch.

    Useful for environments/transformers versions where evaluation_strategy='epoch'
    is not supported or ignored. This guarantees evals at epoch boundaries.
    """

    def on_epoch_end(self, args, state, control, **kwargs):  # type: ignore[override]
        try:
            control.should_evaluate = True
        except Exception:
            pass
        return control
