"""Frame classifier model wrapper built on top of Hugging Face transformers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

import torch
from torch.nn.functional import sigmoid

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from efi_analyser.frames.types import FrameSchema


@dataclass
class FrameClassifierSpec:
    """Configuration for training/inference of the frame classifier."""

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
    output_dir: str = "frame_classifier_runs"
    freeze_base_model: bool = False
    # Logging/reporting
    report_to: list[str] = field(default_factory=list)
    logging_dir: str | None = None
    run_name: str | None = None
    # Evaluation options
    eval_threshold: float = 0.5
    eval_top_k: int | None = None
    eval_steps: int | None = None


class FrameClassifierModel:
    """Thin wrapper around a trained Hugging Face sequence classifier."""

    def __init__(
        self,
        schema: FrameSchema,
        label_order: Sequence[str],
        model: AutoModelForSequenceClassification,
        tokenizer,
        spec: FrameClassifierSpec,
    ) -> None:
        self.schema = schema
        self.label_order = list(label_order)
        self.model = model
        self.tokenizer = tokenizer
        self.spec = spec

    def predict_proba_batch(
        self,
        texts: Sequence[str],
        batch_size: int = 8,
        device: Optional[str] = None,
        progress_callback: Optional[callable] = None,
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
                probs = sigmoid(logits).cpu().numpy()
                for row in probs:
                    outputs.append({fid: float(score) for fid, score in zip(self.label_order, row)})
                
                # Update progress callback if provided
                if progress_callback:
                    progress_callback(len(batch_texts))
        return outputs

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        # Filter spec dict to only include FrameClassifierSpec fields
        # This handles cases where ClassifierSettings (which extends FrameClassifierSpec)
        # is passed with extra fields like 'enabled' and 'cv_folds'
        spec_dict = self.spec.__dict__
        valid_spec_fields = {f.name for f in fields(FrameClassifierSpec)}
        filtered_spec_dict = {k: v for k, v in spec_dict.items() if k in valid_spec_fields}
        payload = {
            "schema": {
                "domain": self.schema.domain,
                "notes": self.schema.notes,
                "frames": [
                    {
                        "frame_id": frame.frame_id,
                        "short_name": frame.short_name,
                        "name": frame.name,
                        "description": frame.description,
                        "keywords": frame.keywords,
                        "examples": frame.examples,
                    }
                    for frame in self.schema.frames
                ],
            },
            "label_order": self.label_order,
            "spec": filtered_spec_dict,
        }
        (output_dir / "frame_classifier.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, output_dir: Path) -> "FrameClassifierModel":
        payload = json.loads((output_dir / "frame_classifier.json").read_text(encoding="utf-8"))
        schema_payload = payload["schema"]
        from efi_analyser.frames.types import Frame  # Local import to avoid cycles.

        frames = []
        for item in schema_payload.get("frames", []):
            short_raw = item.get("short_name") or (
                item.get("name", "") if item.get("name") else item.get("frame_id", "")
            )
            frames.append(
                Frame(
                    frame_id=item["frame_id"],
                    name=item["name"],
                    description=item.get("description", ""),
                    keywords=item.get("keywords", []),
                    examples=item.get("examples", []),
                    short_name=str(short_raw).strip(),
                )
            )

        schema = FrameSchema(
            domain=schema_payload["domain"],
            frames=frames,
            notes=schema_payload.get("notes", ""),
        )
        label_order = payload["label_order"]
        # Filter spec dict to only include fields that belong to FrameClassifierSpec
        # This handles cases where ClassifierSettings (which extends FrameClassifierSpec)
        # was saved with extra fields like 'enabled' and 'cv_folds'
        spec_dict = payload.get("spec", {})
        valid_spec_fields = {f.name for f in fields(FrameClassifierSpec)}
        filtered_spec_dict = {k: v for k, v in spec_dict.items() if k in valid_spec_fields}
        spec = FrameClassifierSpec(**filtered_spec_dict)
        model = AutoModelForSequenceClassification.from_pretrained(output_dir, use_safetensors=True)
        tokenizer = AutoTokenizer.from_pretrained(output_dir)
        return cls(schema=schema, label_order=label_order, model=model, tokenizer=tokenizer, spec=spec)


__all__ = ["FrameClassifierModel", "FrameClassifierSpec"]
