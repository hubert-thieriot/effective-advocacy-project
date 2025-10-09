"""LLM-based frame application for assigning frame probabilities to passages."""

from __future__ import annotations

import copy
import hashlib
import json
from inspect import Parameter, signature
from typing import Any, Dict, Iterable, List, MutableMapping, Sequence, Tuple, Optional

try:
    from jinja2 import Template  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Template = None  # type: ignore

from .types import FrameAssignment, FrameSchema
from .identifiers import make_global_doc_id, split_passage_id


class LLMFrameApplicator:
    """Apply an induced frame schema to passages using an LLM backend."""

    DEFAULT_BATCH_SIZE: int = 8

    def __init__(
        self,
        llm_client: Any,
        batch_size: int = DEFAULT_BATCH_SIZE,
        rationale_threshold: float = 0.45,
        infer_timeout: float | None = 600.0,
        cache: MutableMapping[str, FrameAssignment] | None = None,
        max_chars_per_passage: int | None = 1200,
        chunk_overlap_chars: int = 120,
        system_template: Optional[str] = None,
        user_template: Optional[str] = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("Frame applicator batch_size must be positive.")
        if max_chars_per_passage is not None and max_chars_per_passage <= 0:
            raise ValueError("max_chars_per_passage must be positive when provided.")
        if (
            max_chars_per_passage is not None
            and chunk_overlap_chars >= max_chars_per_passage
            and chunk_overlap_chars > 0
        ):
            raise ValueError("chunk_overlap_chars must be smaller than max_chars_per_passage.")

        self.llm_client = llm_client
        self.batch_size = batch_size
        self.rationale_threshold = rationale_threshold
        self.infer_timeout = infer_timeout
        self.max_chars_per_passage = max_chars_per_passage
        self.chunk_overlap_chars = max(0, chunk_overlap_chars)
        self._cache: MutableMapping[str, FrameAssignment] = cache if cache is not None else {}
        self._llm_spec_key = self._compute_model_spec_key()
        self._infer_supports_timeout = self._llm_accepts_timeout()
        # Optional Jinja templates for system/user prompts
        self._system_template = system_template
        self._user_template = user_template
        # Expose last built messages (last batch) for provenance
        self.last_built_messages: Optional[List[Dict[str, str]]] = None
        # Collect all batch messages for this run
        self.emitted_messages: List[List[Dict[str, str]]] = []

    # ------------------------------------------------------------------ public
    def batch_assign(
        self,
        schema: FrameSchema,
        passages: Sequence[str] | Sequence[Tuple[str, str]],
        *,
        top_k: int = 3,
    ) -> List[FrameAssignment]:
        """Assign frame probabilities for each passage in order of input."""
        prepared = self._prepare_passages(passages)
        if not prepared:
            return []

        schema_hash = self._schema_hash(schema)
        cached: Dict[str, FrameAssignment] = {}
        pending: List[Tuple[str, str, str]] = []

        for passage_id, text in prepared:
            cache_key = self._cache_key(schema_hash, text)
            cached_assignment = self._cache.get(cache_key)
            if cached_assignment is not None:
                cached[passage_id] = self._clone_assignment(cached_assignment)
            else:
                pending.append((passage_id, text, cache_key))

        results: Dict[str, FrameAssignment] = dict(cached)
        skipped_passages: List[str] = []

        if pending:
            for batch in self._chunk(pending, self.batch_size):
                messages = self._build_messages(schema, batch, top_k)
                self.last_built_messages = list(messages)
                self.emitted_messages.append(list(messages))
                infer_kwargs: Dict[str, Any] = {}
                if self.infer_timeout is not None and self._infer_supports_timeout:
                    infer_kwargs["timeout"] = self.infer_timeout
                try:
                    raw_response = self.llm_client.infer(messages, **infer_kwargs)
                    parsed = self._parse_batch_response(schema, batch, raw_response, top_k)
                except ValueError as exc:
                    print(
                        f"⚠️ Skipping frame application batch of {len(batch)} passages due to parse error: {exc}"
                    )
                    skipped_passages.extend(pid for pid, _, _ in batch)
                    continue

                for passage_id, assignment in parsed.items():
                    cache_key = next(key for pid, _, key in batch if pid == passage_id)
                    self._cache[cache_key] = self._clone_assignment(assignment)
                    results[passage_id] = assignment

        if skipped_passages:
            print(
                f"⚠️ Dropped {len(skipped_passages)} passages due to malformed LLM responses."
            )

        ordered = [results[pid] for pid, _ in prepared if pid in results]
        return ordered

    # ------------------------------------------------------------------ helpers
    def _prepare_passages(
        self,
        passages: Sequence[str] | Sequence[Tuple[str, str]],
    ) -> List[Tuple[str, str]]:
        prepared: List[Tuple[str, str]] = []
        for idx, item in enumerate(passages):
            if isinstance(item, tuple):
                passage_id, text = item
            else:
                passage_id, text = f"passage_{idx:05d}", str(item)
            segments = self._split_passage(passage_id, text)
            prepared.extend(segments)
        return prepared

    def _split_passage(self, passage_id: str, text: str) -> List[Tuple[str, str]]:
        text = text.strip()
        if not text:
            return []

        if self.max_chars_per_passage is None:
            return [(passage_id, text)]

        chunks = self._chunk_text(text)
        if len(chunks) == 1:
            return [(passage_id, chunks[0])]

        segmented: List[Tuple[str, str]] = []
        for index, chunk in enumerate(chunks, start=1):
            chunk_id = passage_id if index == 1 else f"{passage_id}::chunk{index:02d}"
            segmented.append((chunk_id, chunk))
        return segmented

    def _chunk_text(self, text: str) -> List[str]:
        limit = self.max_chars_per_passage
        if limit is None or len(text) <= limit:
            return [text.strip()]

        chunks: List[str] = []
        start = 0
        length = len(text)
        while start < length:
            end = min(length, start + limit)
            if end < length:
                whitespace = text.rfind(" ", start, end)
                if whitespace > start + int(limit * 0.4):
                    end = whitespace
            segment = text[start:end].strip()
            if segment:
                chunks.append(segment)
            if end >= length:
                break
            next_start = end - self.chunk_overlap_chars if self.chunk_overlap_chars else end
            if next_start <= start:
                next_start = end
            start = next_start
        return chunks if chunks else [text.strip()]

    def _build_messages(
        self,
        schema: FrameSchema,
        batch: Sequence[Tuple[str, str, str]],
        top_k: int,
    ) -> List[Dict[str, str]]:
        frame_lines = []
        for frame in schema.frames:
            keywords = ", ".join(frame.keywords) if frame.keywords else "(no keywords)"
            examples = "; ".join(frame.examples[:2]) if frame.examples else "—"
            frame_lines.append(
                f"- {frame.frame_id}: {frame.name}\n"
                f"  Description: {frame.description or '(none)'}\n"
                f"  Keywords: {keywords}\n"
                f"  Example: {examples}"
            )

        passage_lines = []
        for passage_id, text, _ in batch:
            passage_lines.append(
                f"- passage_id: {passage_id}\n"
                f"  TEXT: {text}"
            )

        schema_lines = "\n".join(frame_lines)
        batch_lines = "\n".join(passage_lines)

        # Require Jinja templates; do not fallback to built-in strings
        if Template is None:
            raise RuntimeError("Jinja2 is required for template-based prompts but is not installed.")
        if not (self._system_template and self._user_template):
            raise RuntimeError("Application prompts must be provided via templates (system + user).")

        ctx_schema_frames = [
            {
                "frame_id": f.frame_id,
                "name": f.name,
                "short_name": f.short_name,
                "description": f.description,
                "keywords": list(f.keywords),
                "examples": list(f.examples),
            }
            for f in schema.frames
        ]
        ctx_passages = [
            {"passage_id": pid, "text": text}
            for (pid, text, _key) in batch
        ]
        ctx = {
            "schema": {
                "schema_id": schema.schema_id,
                "domain": schema.domain,
                "frames": ctx_schema_frames,
            },
            "frames_text": schema_lines,
            "passages": ctx_passages,
            "passages_text": batch_lines,
            "top_k": top_k,
        }
        sys_content = Template(self._system_template).render(**ctx)
        usr_content = Template(self._user_template).render(**ctx)
        return [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": usr_content},
        ]

    def _parse_batch_response(
        self,
        schema: FrameSchema,
        batch: Sequence[Tuple[str, str, str]],
        raw_response: str,
        top_k: int,
    ) -> Dict[str, FrameAssignment]:
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if len(lines) >= 3:
                cleaned = "\n".join(lines[1:-1]).strip()

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Frame applicator returned invalid JSON: {exc}") from exc

        if not isinstance(payload, list):
            raise ValueError("Frame applicator expected a JSON array response.")

        batch_lookup = {passage_id: text for passage_id, text, _ in batch}
        frame_ids = {frame.frame_id for frame in schema.frames}

        assignments: Dict[str, FrameAssignment] = {}
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            passage_id = entry.get("passage_id")
            if not passage_id or passage_id not in batch_lookup:
                continue

            probs_raw = entry.get("probs", {})
            if not isinstance(probs_raw, dict):
                probs_raw = {}

            probabilities = {}
            for fid, value in probs_raw.items():
                if fid not in frame_ids:
                    continue
                try:
                    prob = float(value)
                except (TypeError, ValueError):
                    continue
                probabilities[fid] = max(0.0, min(prob, 1.0))

            for fid in frame_ids:
                probabilities.setdefault(fid, 0.0)

            total = sum(probabilities.values())
            if total > 0:
                probabilities = {fid: value / total for fid, value in probabilities.items()}

            top_frames = entry.get("top_frames")
            if not isinstance(top_frames, list) or not all(isinstance(fid, str) for fid in top_frames):
                top_frames = self._derive_top_frames(probabilities, top_k)
            else:
                top_frames = [fid for fid in top_frames if fid in frame_ids][:top_k]
                if not top_frames:
                    top_frames = self._derive_top_frames(probabilities, top_k)

            rationale = entry.get("rationale")
            if not isinstance(rationale, str):
                rationale = ""
            rationale = rationale.strip()

            evidence_spans = entry.get("evidence_spans", [])
            if not isinstance(evidence_spans, list):
                evidence_spans = []
            evidence_clean = [str(span).strip() for span in evidence_spans if str(span).strip()]

            corpus_name, local_doc_id, _ = split_passage_id(passage_id)
            metadata = {
                "doc_id": local_doc_id,
                "global_doc_id": make_global_doc_id(corpus_name, local_doc_id) if corpus_name else local_doc_id,
            }
            if corpus_name:
                metadata["corpus"] = corpus_name
            assignment = FrameAssignment(
                passage_id=passage_id,
                passage_text=batch_lookup[passage_id],
                probabilities=probabilities,
                top_frames=top_frames,
                rationale=rationale,
                evidence_spans=evidence_clean[:3],
                metadata=metadata,
            )
            assignments[passage_id] = assignment

        missing_ids = {pid for pid, _, _ in batch} - assignments.keys()
        if missing_ids:
            print(
                f"⚠️ Frame applicator response missing {len(missing_ids)} passages; dropping them from this batch."
            )

        return assignments

    def _derive_top_frames(self, probabilities: Dict[str, float], top_k: int) -> List[str]:
        ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        return [fid for fid, prob in ordered[:top_k] if prob > 0]

    def _chunk(self, items: Sequence[Tuple[str, str, str]], size: int) -> Iterable[List[Tuple[str, str, str]]]:
        batch: List[Tuple[str, str, str]] = []
        for item in items:
            batch.append(item)
            if len(batch) == size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _schema_hash(self, schema: FrameSchema) -> str:
        schema_dict = {
            "domain": schema.domain,
            "notes": schema.notes,
            "schema_id": schema.schema_id,
            "frames": [
                {
                    "frame_id": frame.frame_id,
                    "name": frame.name,
                    "description": frame.description,
                    "keywords": frame.keywords,
                    "examples": frame.examples,
                }
                for frame in schema.frames
            ],
        }
        encoded = json.dumps(schema_dict, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _cache_key(self, schema_hash: str, passage_text: str) -> str:
        passage_hash = hashlib.sha256(passage_text.encode("utf-8")).hexdigest()
        return f"{self._llm_spec_key}:{schema_hash}:{passage_hash}"

    def _compute_model_spec_key(self) -> str:
        candidate_attrs = (
            "spec_key",
            "model_spec_key",
            "model_key",
            "key",
        )

        for attr_name in candidate_attrs:
            attr = getattr(self.llm_client, attr_name, None)
            if attr is None:
                continue
            try:
                value = attr() if callable(attr) else attr
            except TypeError:
                continue
            if isinstance(value, str):
                return value
            try:
                serialized = json.dumps(value, sort_keys=True, default=str)
            except TypeError:
                serialized = str(value)
            return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

        config = getattr(self.llm_client, "config", None)
        if config is not None:
            try:
                mapping = config if isinstance(config, dict) else config.__dict__
                serialized = json.dumps(mapping, sort_keys=True, default=str)
                return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
            except Exception:
                pass

        return f"{self.llm_client.__class__.__module__}.{self.llm_client.__class__.__name__}"

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

    def _clone_assignment(self, assignment: FrameAssignment) -> FrameAssignment:
        return FrameAssignment(
            passage_id=assignment.passage_id,
            passage_text=assignment.passage_text,
            probabilities=dict(assignment.probabilities),
            top_frames=list(assignment.top_frames),
            rationale=assignment.rationale,
            evidence_spans=list(assignment.evidence_spans),
            metadata=copy.deepcopy(assignment.metadata),
        )
