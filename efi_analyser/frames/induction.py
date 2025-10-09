"""Frame induction implementation for generating frame schemas from passages."""

from __future__ import annotations

import json
from inspect import Parameter, signature
from typing import Any, Dict, Iterable, List, Sequence, Optional

try:
    from jinja2 import Template  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Template = None  # type: ignore

from .types import Frame, FrameSchema


class FrameInducer:
    """Generate a frame schema for a given domain using an LLM backend."""

    MAX_PASSAGES: int = 400
    MAX_CHARS_PER_PASSAGE: int = 800
    SYSTEM_PROMPT: str = (
        "Discover distinct, non-overlapping media frames (lenses/angles) for the given domain. "
        "Deliverables per frame: a crisp definition; positive triggers (explicit phrases/keywords and semantic cues); anti-triggers (words/contexts that should not fire this frame); near-misses (confusable content with rationale why it's not this frame); decision rules (boolean logic, e.g., MUST/SHOULD/NEVER conditions); hard evidence cues (verbatim text patterns); soft cues (semantic signals); regex seeds; 3 positive examples and 3 counter-examples from the passages; and a scoring rubric (0–3 scale) you'd apply to a passage. "
        "Expand keywords with common synonyms and near-variants. Ensure triggers include both lexical cues (quoted phrases) and semantic cues (paraphrasable conditions). Provide anti-triggers that disambiguate closely related frames to minimize overlap. "
        "Avoid sentiment or policy prescriptions. Prefer generalizable, domain-portable language. Frames must be mutually exclusive on triggers (if two share triggers, split or refine). When narratives oppose (e.g., jobs vs. health harms), split them. "
        "For each frame, provide a short_name that is a human-friendly 1–3 word label (≤20 characters), using full words and no mid-word truncation. "
        "Output valid JSON only, matching the provided schema. No prose."
    )

    def __init__(
        self,
        llm_client: Any,
        domain: str,
        frame_target: int | str = 8,
        infer_timeout: float | None = 600,
        max_chars_per_passage: int | None = None,
        chunk_overlap_chars: int = 80,
        max_passages_per_call: int | None = None,
        max_total_passages: int | None = None,
        frame_guidance: str | None = None,
        system_template: Optional[str] = None,
        user_template: Optional[str] = None,
    ) -> None:
        """Initialize the inducer with an LLM client and target domain."""
        self.llm_client = llm_client
        self.domain = domain
        self.frame_target = frame_target
        self.infer_timeout = infer_timeout
        self._infer_supports_timeout = self._llm_accepts_timeout()
        self._frame_target_text = self._format_frame_target(frame_target)

        if max_chars_per_passage is None:
            max_chars_per_passage = self.MAX_CHARS_PER_PASSAGE
        if max_chars_per_passage is not None and max_chars_per_passage <= 0:
            raise ValueError("max_chars_per_passage must be positive when provided.")

        self.max_chars_per_passage = max_chars_per_passage
        self.chunk_overlap_chars = max(0, chunk_overlap_chars)

        if max_passages_per_call is None:
            max_passages_per_call = self.MAX_PASSAGES
        if max_passages_per_call <= 0:
            raise ValueError("max_passages_per_call must be positive.")
        self.max_passages_per_call = max_passages_per_call

        if max_total_passages is None:
            max_total_passages = max_passages_per_call * 5
        if max_total_passages < max_passages_per_call:
            raise ValueError("max_total_passages must be >= max_passages_per_call.")
        self.max_total_passages = max_total_passages
        self.frame_guidance = frame_guidance.strip() if frame_guidance else ""
        # Optional Jinja templates for system/user prompts
        self._system_template = system_template
        self._user_template = user_template
        # Expose last-built messages for provenance/debugging
        self.last_built_messages: Optional[List[Dict[str, str]]] = None
        self.emitted_messages: List[List[Dict[str, str]]] = []

    def induce(self, passages: Iterable[str]) -> FrameSchema:
        """Produce a frame schema for the configured domain and passages."""
        prepared = self._prepare_passages(passages)
        if not prepared:
            raise ValueError("Frame induction requires at least one passage string.")

        if len(prepared) <= self.max_passages_per_call:
            return self._induce_single(prepared)

        partial_schemas: List[FrameSchema] = []
        for chunk in self._chunk_passages(prepared):
            partial_schemas.append(self._induce_single(chunk))

        return self._merge_schemas(prepared, partial_schemas)

    # ------------------------------------------------------------------ utils
    def _prepare_passages(self, passages: Iterable[str]) -> List[str]:
        seen = set()
        unique: List[str] = []
        for passage in passages:
            normalized = passage.strip()
            if not normalized or normalized in seen:
                continue
            if (
                self.max_chars_per_passage is not None
                and len(normalized) > self.max_chars_per_passage
            ):
                truncated = normalized[: self.max_chars_per_passage].rstrip()
                normalized = f"{truncated}..."
            seen.add(normalized)
            unique.append(normalized)
            if len(unique) >= self.max_total_passages:
                break
        return unique

    def _build_messages(self, passages: Sequence[str]) -> List[dict[str, str]]:
        # Require Jinja templates; do not fallback to built-in strings
        if Template is None:
            raise RuntimeError("Jinja2 is required for template-based prompts but is not installed.")
        if not (self._system_template and self._user_template):
            raise RuntimeError("Induction prompts must be provided via templates (system + user).")

        passage_lines = [f"{idx + 1}. {text}" for idx, text in enumerate(passages)]
        passages_joined = "\n".join(passage_lines)
        ctx = {
            "domain": self.domain,
            "frame_target": self._frame_target_text,
            "frame_guidance": self.frame_guidance,
            "passages": list(passages),
            "passages_joined": passages_joined,
            "max_passages_per_call": self.max_passages_per_call,
        }
        sys_content = Template(self._system_template).render(**ctx)
        usr_content = Template(self._user_template).render(**ctx)
        return [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": usr_content},
        ]

    def _induce_single(self, passages: Sequence[str]) -> FrameSchema:
        messages = self._build_messages(passages)
        self.last_built_messages = list(messages)
        self.emitted_messages.append(list(messages))
        infer_kwargs: dict[str, Any] = {}
        if self.infer_timeout is not None and self._infer_supports_timeout:
            infer_kwargs["timeout"] = self.infer_timeout
        raw_response = self.llm_client.infer(messages, **infer_kwargs)
        return self._parse_response(raw_response)

    def _chunk_passages(self, passages: Sequence[str]) -> List[Sequence[str]]:
        chunked: List[Sequence[str]] = []
        for index in range(0, len(passages), self.max_passages_per_call):
            chunked.append(passages[index : index + self.max_passages_per_call])
        return chunked

    def _merge_schemas(
        self, all_passages: Sequence[str], partial_schemas: Sequence[FrameSchema]
    ) -> FrameSchema:
        if not partial_schemas:
            raise ValueError("Frame induction produced no partial schemas.")
        if len(partial_schemas) == 1:
            return partial_schemas[0]

        messages = self._build_merge_messages(all_passages, partial_schemas)
        self.last_built_messages = list(messages)
        self.emitted_messages.append(list(messages))
        infer_kwargs: dict[str, Any] = {}
        if self.infer_timeout is not None and self._infer_supports_timeout:
            infer_kwargs["timeout"] = self.infer_timeout
        raw_response = self.llm_client.infer(messages, **infer_kwargs)
        return self._parse_response(raw_response)

    def _build_merge_messages(
        self, passages: Sequence[str], partial_schemas: Sequence[FrameSchema]
    ) -> List[dict[str, str]]:
        frames_summary = []
        for batch_idx, schema in enumerate(partial_schemas, start=1):
            for frame in schema.frames:
                frames_summary.append(
                    {
                        "batch": batch_idx,
                        "frame_id": frame.frame_id,
                        "name": frame.name,
                        "short_name": frame.short_name,
                        "description": frame.description[:300],
                        "keywords": frame.keywords[:6],
                        "examples": frame.examples[:2],
                    }
                )

        frames_json = json.dumps(frames_summary, ensure_ascii=False)
        sample_limit = min(40, len(passages))
        sampled_passages = "\n".join(
            f"{idx + 1}. {text}" for idx, text in enumerate(passages[:sample_limit])
        )
        schema_instruction = self._schema_instruction()

        guidance_line = f"GUIDANCE: {self.frame_guidance}\n" if self.frame_guidance else ""
        frame_target_line = f"Frame target: {self._frame_target_text}\n" if self._frame_target_text else ""

        user_prompt = (
            f"DOMAIN: {self.domain}\n"
            f"{frame_target_line}"
            f"Goal: Induce a compact, mutually-exclusive frame set for this domain and derive operational detection rules.\n\n"
            f"Constraints:\n"
            "- Start from the guidance seed frames if provided, but refine/split/merge as needed for non-overlap.\n"
            "- Triggers must include both lexical (quoted phrases) and semantic cues (paraphrasable conditions).\n"
            "- Anti-triggers must explicitly name *false positives* (e.g., \"capacity addition\" is NOT \"overcapacity\").\n"
            "- Decision rules must include at least one MUST, at most three SHOULD, and at least two NEVER conditions per frame.\n"
            "- Regex seeds should be robust but conservative (lower FP > higher FN).\n"
            "- Examples/counter-examples MUST be exact quotes from the provided passages when possible; if not available, construct minimal plausible snippets.\n\n"
            + (f"Seed frames (optional):\n{self.frame_guidance}\n\n" if self.frame_guidance else "")
            + f"Passages (≤ {self.max_passages_per_call}, sampled & deduped):\n"
            f"{sampled_passages}\n\n"
            f"Return JSON only using this schema:\n"
            f"{schema_instruction}"
        )

        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def _schema_instruction(self) -> str:
        instruction = (
            '{"domain":"' + self.domain.replace('"', "'") + '",'
            '"frames":[{'
            '"frame_id":"...",'
            '"short_name":"...",'
            '"name":"...",'
            '"description":"...",'
            '"keywords":["..."],'
            '"triggers":{"positive":["..."],"anti":["..."]},'
            '"examples":["..."],'
            '"counter_examples":["..."]'
            '},...],'
            '"notes":""}'
        )
        return instruction

    def _llm_accepts_timeout(self) -> bool:
        infer = getattr(self.llm_client, "infer", None)
        if not callable(infer):
            return False

        try:
            params = signature(infer).parameters.values()
        except (ValueError, TypeError):
            return False

        for param in params:
            if param.kind == Parameter.VAR_KEYWORD:
                return True
        return any(param.name == "timeout" for param in params)

    def _format_frame_target(self, target: int | str) -> str:
        if isinstance(target, str):
            return target.strip()
        if target is None:
            return ""
        return f"{int(target)} frames"

    def _parse_response(self, raw_response: str) -> FrameSchema:
        content = raw_response.strip()

        if content.startswith("```"):
            lines = content.splitlines()
            if len(lines) >= 3:
                content = "\n".join(lines[1:-1]).strip()

        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            snippet = content[:200].replace("\n", " ")
            raise ValueError(f"Frame inducer returned invalid JSON: {exc}: {snippet}") from exc

        if not isinstance(payload, dict):
            raise ValueError("Frame inducer expected JSON object at top level.")

        domain = payload.get("domain", self.domain)
        frames_payload = payload.get("frames", [])
        notes = payload.get("notes", "")

        if not isinstance(frames_payload, list):
            raise ValueError("Frame inducer response 'frames' must be a list.")

        frames: List[Frame] = []
        for entry in frames_payload:
            if not isinstance(entry, dict):
                continue
            frame_id = entry.get("frame_id")
            name = entry.get("name")
            description = entry.get("description", "")
            keywords = entry.get("keywords", [])
            examples = entry.get("examples", [])
            short_name = entry.get("short_name")

            if not frame_id or not name:
                continue

            base_short = (
                str(short_name).strip()
                if short_name
                else (str(name).strip() if str(name).strip() else str(frame_id))
            )
            resolved_short = base_short

            frames.append(
                Frame(
                    frame_id=str(frame_id),
                    name=str(name),
                    description=str(description),
                    keywords=[str(keyword) for keyword in keywords if isinstance(keyword, str)],
                    examples=[str(example) for example in examples if isinstance(example, str)],
                    short_name=resolved_short,
                )
            )

        if not frames:
            raise ValueError("Frame inducer did not return any valid frames.")

        return FrameSchema(domain=str(domain), frames=frames, notes=str(notes))
