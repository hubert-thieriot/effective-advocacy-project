"""LLM-based stance annotation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
import json
import re

from efi_analyser.frames.types import Candidate
from efi_analyser.stance.types import STANCE_LABELS, StanceAssignments, StanceResult


class LLMStanceAnnotator:
    """Annotate stance labels for (chunk, target) pairs using an LLM."""

    def __init__(
        self,
        llm_client: object,
        *,
        labels: Sequence[str] = STANCE_LABELS,
        system_template: Optional[str] = None,
        user_template: Optional[str] = None,
        resolved_messages_dir: Optional[Path] = None,
        resolved_messages_prefix: str = "stance_annotation",
    ) -> None:
        self.llm_client = llm_client
        self.labels = list(labels)
        self._system_template = system_template
        self._user_template = user_template
        self._resolved_messages_dir = resolved_messages_dir
        self._resolved_messages_prefix = resolved_messages_prefix
        self._call_index = 0

    def annotate(
        self,
        candidates: Sequence[Candidate],
        targets: Sequence[str],
        *,
        show_progress: bool = False,
    ) -> StanceAssignments:
        results = StanceAssignments()
        targets_list = [t for t in targets if str(t).strip()]
        if not targets_list:
            return results

        iterator = candidates
        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(candidates, desc="Annotating stance", unit="passage")
            except Exception:
                iterator = candidates

        for candidate in iterator:
            instructions, input = self._build_messages(candidate.text, targets_list)
            self._maybe_save_messages(instructions, input)
            raw = self.llm_client.infer(instructions, input)
            label_map = self._parse_label_map(raw)
            for target in targets_list:
                label = label_map.get(target, "neutral")
                scores = {lab: 0.0 for lab in self.labels}
                scores[label] = 1.0
                results.append(
                    StanceResult(
                        chunk_id=candidate.item_id,
                        target=target,
                        text=candidate.text,
                        scores=scores,
                        label=label,
                        metadata=dict(candidate.meta or {}),
                    )
                )

        return results

    def _build_messages(self, text: str, targets: Sequence[str]) -> Tuple[str, str]:
        labels_str = ", ".join(self.labels)
        if self._system_template and self._user_template:
            ctx = {
                "labels": labels_str,
                "targets": list(targets),
                "text": text,
            }
            instructions = self._system_template.format(**ctx)
            input = self._user_template.format(**ctx)
        else:
            instructions = (
                "You are a stance annotator. For each target, label the stance as one of: "
                f"{labels_str}. Return only valid JSON."
            )
            input = (
                "Text:\n"
                f"{text}\n\n"
                "Targets:\n"
                + "\n".join(f"- {t}" for t in targets)
                + "\n\n"
                "Output JSON mapping target -> label."
            )
        return (instructions, input)

    def _parse_label_map(self, raw: str) -> Dict[str, str]:
        try:
            return self._normalize_labels(json.loads(raw))
        except Exception:
            pass
        # Attempt to extract JSON object from text
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            try:
                return self._normalize_labels(json.loads(match.group(0)))
            except Exception:
                return {}
        return {}

    def _normalize_labels(self, payload: object) -> Dict[str, str]:
        if not isinstance(payload, dict):
            return {}
        normalized: Dict[str, str] = {}
        allowed = {lab.lower(): lab for lab in self.labels}
        for target, label in payload.items():
            key = str(target).strip()
            label_str = str(label).strip().lower()
            if label_str in allowed:
                normalized[key] = allowed[label_str]
        return normalized

    def _maybe_save_messages(self, instructions: str, input: str) -> None:
        if not self._resolved_messages_dir:
            return
        self._resolved_messages_dir.mkdir(parents=True, exist_ok=True)
        self._call_index += 1
        out_path = self._resolved_messages_dir / f"{self._resolved_messages_prefix}_{self._call_index:03d}.json"
        out_path.write_text(json.dumps({"instructions": instructions, "input": input}, indent=2, ensure_ascii=False), encoding="utf-8")


__all__ = ["LLMStanceAnnotator"]
