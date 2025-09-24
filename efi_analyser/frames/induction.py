"""Frame induction implementation for generating frame schemas from passages."""

from __future__ import annotations

import json
from inspect import Parameter, signature
from typing import Any, Iterable, List, Sequence

from .types import Frame, FrameSchema


class FrameInducer:
    """Generate a frame schema for a given domain using an LLM backend."""

    MAX_PASSAGES: int = 400
    MAX_CHARS_PER_PASSAGE: int = 800
    SYSTEM_PROMPT: str = (
        "Discover distinct media frames (lenses/angles) for the given domain. "
        "Output valid JSON matching the schema; avoid sentiment labels; ensure frames "
        "are generalizable and non-overlapping."
    )

    def __init__(
        self,
        llm_client: Any,
        domain: str,
        frame_target: int | str = 8,
        infer_timeout: float | None = 600,
    ) -> None:
        """Initialize the inducer with an LLM client and target domain."""
        self.llm_client = llm_client
        self.domain = domain
        self.frame_target = frame_target
        self.infer_timeout = infer_timeout
        self._infer_supports_timeout = self._llm_accepts_timeout()
        self._frame_target_text = self._format_frame_target(frame_target)

    def induce(self, passages: Iterable[str]) -> FrameSchema:
        """Produce a frame schema for the configured domain and passages.

        Args:
            passages: Iterable of passage strings (list, generator, etc.).
        """
        prepared = self._prepare_passages(passages)
        if not prepared:
            raise ValueError("Frame induction requires at least one passage string.")

        messages = self._build_messages(prepared)
        infer_kwargs: dict[str, Any] = {}
        if self.infer_timeout is not None and self._infer_supports_timeout:
            infer_kwargs["timeout"] = self.infer_timeout

        raw_response = self.llm_client.infer(messages, **infer_kwargs)
        return self._parse_response(raw_response)

    # ------------------------------------------------------------------ utils
    def _prepare_passages(self, passages: Iterable[str]) -> List[str]:
        seen = set()
        unique: List[str] = []
        for passage in passages:
            if not passage:
                continue
            normalized = passage.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            if len(normalized) > self.MAX_CHARS_PER_PASSAGE:
                truncated = normalized[: self.MAX_CHARS_PER_PASSAGE].rstrip()
                normalized = f"{truncated}..."
            unique.append(normalized)
            if len(unique) >= self.MAX_PASSAGES:
                break
        return unique

    def _build_messages(self, passages: Sequence[str]) -> List[dict[str, str]]:
        passage_lines = [f"{idx + 1}. {text}" for idx, text in enumerate(passages)]
        passages_joined = "\n".join(passage_lines)
        frame_target_line = ""
        if self._frame_target_text:
            frame_target_line = f"Frame target: {self._frame_target_text}\n"
        schema_instruction = (
            '{"domain":"' + self.domain.replace('"', "'") + '",'  # Avoid double quotes in domain
            '"frames":[{"frame_id":"...","name":"...","description":"...",'
            '"keywords":["..."],"examples":["..."]},...],"notes":""}'
        )

        user_prompt = (
            f"DOMAIN: {self.domain}\n"
            f"{frame_target_line}Passages (≤{self.MAX_PASSAGES}, sampled & deduped):\n"
            f"{passages_joined}\n\n"
            f"Return JSON: {schema_instruction}"
        )

        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

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

            if not frame_id or not name:
                continue

            frames.append(
                Frame(
                    frame_id=str(frame_id),
                    name=str(name),
                    description=str(description),
                    keywords=[str(keyword) for keyword in keywords if isinstance(keyword, str)],
                    examples=[str(example) for example in examples if isinstance(example, str)],
                )
            )

        if not frames:
            raise ValueError("Frame inducer did not return any valid frames.")

        return FrameSchema(domain=str(domain), frames=frames, notes=str(notes))
