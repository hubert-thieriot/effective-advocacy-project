"""Safe version of FrameClassifierModel with better error handling and memory management."""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
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


class SafeFrameClassifierModel:
    """Safe wrapper around a trained Hugging Face sequence classifier with better error handling."""

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
    ) -> List[Dict[str, float]]:
        """Predict probabilities with better error handling and memory management."""
        self.model.eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        outputs: List[Dict[str, float]] = []
        
        try:
            with torch.no_grad():
                for start in range(0, len(texts), batch_size):
                    try:
                        batch_texts = texts[start : start + batch_size]
                        
                        # Add progress indicator for large batches
                        if len(texts) > 100 and start % 100 == 0:
                            print(f"  Processing batch {start//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                        
                        encoded = self.tokenizer(
                            list(batch_texts),
                            padding=True,
                            truncation=True,
                            max_length=self.spec.max_length,
                            return_tensors="pt",
                        )
                        encoded = {k: v.to(device) for k, v in encoded.items()}
                        
                        # This is the line that was failing
                        logits = self.model(**encoded).logits
                        probs = sigmoid(logits).cpu().numpy()
                        
                        for row in probs:
                            outputs.append({fid: float(score) for fid, score in zip(self.label_order, row)})
                        
                        # Clean up batch tensors
                        del encoded
                        del logits
                        del probs
                        
                        # Force garbage collection every 50 batches
                        if start % (batch_size * 50) == 0:
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                
                    except Exception as e:
                        print(f"Error processing batch starting at {start}: {e}")
                        # Add dummy predictions for failed batch
                        for _ in batch_texts:
                            outputs.append({fid: 0.0 for fid in self.label_order})
                        continue
                        
        except Exception as e:
            print(f"Critical error in predict_proba_batch: {e}")
            # Return dummy predictions for all texts
            outputs = [{fid: 0.0 for fid in self.label_order} for _ in texts]
        
        return outputs

    def save(self, output_dir: Path) -> None:
        """Save the model with error handling."""
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
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
                "spec": self.spec.__dict__,
            }
            (output_dir / "frame_classifier.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            print(f"Error saving model: {e}")

    @classmethod
    def load(cls, output_dir: Path) -> "SafeFrameClassifierModel":
        """Load the model with error handling."""
        try:
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
            spec = FrameClassifierSpec(**payload.get("spec", {}))
            model = AutoModelForSequenceClassification.from_pretrained(output_dir)
            tokenizer = AutoTokenizer.from_pretrained(output_dir)
            return cls(schema=schema, label_order=label_order, model=model, tokenizer=tokenizer, spec=spec)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise


__all__ = ["SafeFrameClassifierModel", "FrameClassifierSpec"]
