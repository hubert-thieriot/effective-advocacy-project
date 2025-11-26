"""LLM-based frame application for assigning frame probabilities to passages."""

from __future__ import annotations

import copy
import hashlib
import json
from inspect import Parameter, signature
from typing import Any, Dict, Iterable, List, MutableMapping, Sequence, Tuple, Optional, Union

from pathlib import Path

from jinja2 import Template
from tqdm import tqdm

from .types import Candidate, FrameAssignment, FrameSchema
from .identifiers import make_global_doc_id, split_passage_id


class LLMFrameAnnotator:
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
        resolved_messages_dir: Optional[Path | str] = None,
        resolved_messages_prefix: str = "batch",
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
        # Optional per-passage metadata attached by upstream samplers
        self._current_metadata: Dict[str, Dict[str, Any]] = {}
        # Optional directory for saving resolved messages
        self._resolved_messages_dir: Optional[Path] = (
            Path(resolved_messages_dir) if resolved_messages_dir else None
        )
        self._resolved_messages_prefix = resolved_messages_prefix

    # ------------------------------------------------------------------ public
    def batch_assign(
        self,
        schema: FrameSchema,
        passages: Sequence[str] | Sequence[Tuple[str, str]] | Sequence[Candidate],
        *,
        top_k: int = 3,
        show_progress: bool = False,
        progress_desc: Optional[str] = None,
        relevance_keywords: Optional[Union[List[str], Dict[str, List[str]]]] = None,
        guidance: Optional[str] = None,
    ) -> List[FrameAssignment]:
        """Assign frame probabilities for each passage in order of input.
        
        Args:
            schema: The frame schema to apply.
            passages: Passages to assign frames to.
            top_k: Number of top frames to return per passage.
            show_progress: Whether to display a progress bar during processing.
            progress_desc: Optional description for the progress bar.
            relevance_keywords: If provided, passages without these keywords get
                zero frame scores without calling the LLM (cost optimization).
                Can be a flat list (applies to all languages) or a dict mapping
                language codes to keyword lists.
            guidance: Additional guidance text to help the annotator avoid false positives.
        """
        self._current_guidance = guidance  # Store for use in _build_messages
        self._current_metadata = {}
        prepared = self._prepare_passages(passages)
        if not prepared:
            return []

        frame_ids = [frame.frame_id for frame in schema.frames]
        
        # Partition passages: those needing annotation vs auto-zero
        to_annotate, zero_annotated = self._partition_by_relevance(
            prepared, relevance_keywords, frame_ids
        )
        
        if zero_annotated:
            print(f"ℹ️ Bypassed LLM for {len(zero_annotated)} passages without relevance keywords (auto-zero).")

        # Check cache for passages that need annotation
        schema_hash = self._schema_hash(schema)
        cached: Dict[str, FrameAssignment] = {}
        pending: List[Tuple[str, str, str]] = []

        for passage_id, text in to_annotate:
            cache_key = self._cache_key(schema_hash, text)
            cached_assignment = self._cache.get(cache_key)
            if cached_assignment is not None:
                cached[passage_id] = self._clone_assignment(cached_assignment, new_passage_id=passage_id)
            else:
                pending.append((passage_id, text, cache_key))

        # Combine zero-annotated with cached results
        results: Dict[str, FrameAssignment] = dict(cached)
        results.update(zero_annotated)
        skipped_passages: List[str] = []

        # Create progress bar if requested
        pbar = None
        if show_progress and pending:
            desc = progress_desc or "Annotating frames"
            pbar = tqdm(
                total=len(pending),
                desc=desc,
                unit="passages",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
            )

        try:
            if pending:
                for batch in self._chunk(pending, self.batch_size):
                    messages = self._build_messages(schema, batch, top_k)
                    self.last_built_messages = list(messages)
                    self.emitted_messages.append(list(messages))
                    # Save messages if directory is configured
                    if self._resolved_messages_dir is not None:
                        self._save_resolved_messages_batch(list(messages), len(self.emitted_messages))
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
                        if pbar is not None:
                            pbar.update(len(batch))
                        continue

                    for passage_id, assignment in parsed.items():
                        cache_key = next(key for pid, _, key in batch if pid == passage_id)
                        self._cache[cache_key] = self._clone_assignment(assignment)
                        results[passage_id] = assignment
                    
                    # Update progress bar after batch is successfully processed
                    if pbar is not None:
                        pbar.update(len(batch))
        finally:
            if pbar is not None:
                pbar.close()

        if skipped_passages:
            print(
                f"⚠️ Dropped {len(skipped_passages)} passages due to malformed LLM responses."
            )

        ordered = [results[pid] for pid, _ in prepared if pid in results]
        return ordered

    # ------------------------------------------------------------------ helpers
    def _prepare_passages(
        self,
        passages: Sequence[str] | Sequence[Tuple[str, str]] | Sequence[Candidate],
    ) -> List[Tuple[str, str]]:
        prepared: List[Tuple[str, str]] = []
        for idx, item in enumerate(passages):
            # Support raw text, (id, text) tuples, and Candidate objects
            if isinstance(item, Candidate):
                passage_id, text = item.item_id, item.text
                base_meta = dict(item.meta or {})
                segments = self._split_passage(passage_id, text)
                for seg_id, seg_text in segments:
                    prepared.append((seg_id, seg_text))
                    self._current_metadata[seg_id] = dict(base_meta)
            else:
                if isinstance(item, tuple):
                    passage_id, text = item
                else:
                    passage_id, text = f"passage_{idx:05d}", str(item)
                segments = self._split_passage(passage_id, text)
                prepared.extend(segments)
        return prepared

    def _partition_by_relevance(
        self,
        prepared: List[Tuple[str, str]],
        relevance_keywords: Optional[Union[List[str], Dict[str, List[str]]]],
        frame_ids: List[str],
    ) -> Tuple[List[Tuple[str, str]], Dict[str, FrameAssignment]]:
        """Partition passages into those needing annotation vs auto-zero.
        
        Args:
            prepared: List of (passage_id, text) tuples
            relevance_keywords: Either a flat list of keywords (language-agnostic)
                or a dict mapping language codes to keyword lists
            frame_ids: List of frame IDs for zero assignments
        
        Returns:
            (to_annotate, zero_annotated): Passages to send to LLM, and 
            pre-built zero assignments for passages without relevance keywords.
        """
        if not relevance_keywords:
            return prepared, {}
        
        # Normalize keywords - either flat list or dict by language
        if isinstance(relevance_keywords, dict):
            # Dict by language: normalize each language's keywords
            keywords_by_lang = {
                lang: [kw.lower().strip() for kw in kws if kw]
                for lang, kws in relevance_keywords.items()
            }
            is_multilingual = True
        else:
            # Flat list
            normalized_keywords = [kw.lower().strip() for kw in relevance_keywords if kw]
            if not normalized_keywords:
                return prepared, {}
            is_multilingual = False
        
        to_annotate: List[Tuple[str, str]] = []
        zero_annotated: Dict[str, FrameAssignment] = {}
        
        for passage_id, text in prepared:
            text_lower = text.lower()
            
            if is_multilingual:
                # Look up language from stored metadata
                meta = self._current_metadata.get(passage_id, {})
                lang = meta.get("language", "").lower()
                # Try exact match, then partial match (e.g., "serbian-cyrillic" -> "serbian")
                keywords = keywords_by_lang.get(lang)
                if keywords is None:
                    # Try language prefix (e.g., "sr" from "serbian-cyrillic")
                    lang_prefix = lang.split("-")[0] if "-" in lang else lang[:2]
                    keywords = keywords_by_lang.get(lang_prefix)
                if keywords is None:
                    # No keywords for this language - annotate it (don't skip)
                    to_annotate.append((passage_id, text))
                    continue
            else:
                keywords = normalized_keywords
            
            has_keyword = any(kw in text_lower for kw in keywords)
            
            if has_keyword:
                to_annotate.append((passage_id, text))
            else:
                zero_annotated[passage_id] = self._create_zero_assignment(
                    passage_id, text, frame_ids
                )
        
        return to_annotate, zero_annotated

    def _create_zero_assignment(
        self,
        passage_id: str,
        text: str,
        frame_ids: List[str],
    ) -> FrameAssignment:
        """Create a zero-probability frame assignment for a passage."""
        zero_probs = {fid: 0.0 for fid in frame_ids}
        corpus_name, local_doc_id, _ = split_passage_id(passage_id)
        metadata = {
            "doc_id": local_doc_id,
            "global_doc_id": make_global_doc_id(corpus_name, local_doc_id) if corpus_name else local_doc_id,
            "bypassed_relevance": True,
        }
        if corpus_name:
            metadata["corpus"] = corpus_name
        
        return FrameAssignment(
            passage_id=passage_id,
            passage_text=text,
            probabilities=zero_probs,
            top_frames=[],
            rationale="",
            evidence_spans=[],
            metadata=metadata,
        )

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
            anti_triggers_str = ", ".join(frame.anti_triggers[:5]) if frame.anti_triggers else "(none)"
            boundary_notes_str = "; ".join(frame.boundary_notes[:2]) if frame.boundary_notes else "(none)"
            frame_lines.append(
                f"- {frame.frame_id}: {frame.name}\n"
                f"  Description: {frame.description or '(none)'}\n"
                f"  Keywords: {keywords}\n"
                f"  Example: {examples}\n"
                f"  Anti-triggers: {anti_triggers_str}\n"
                f"  Boundary notes: {boundary_notes_str}"
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
                "anti_triggers": list(f.anti_triggers),
                "boundary_notes": list(f.boundary_notes),
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
            "guidance": getattr(self, "_current_guidance", None) or "",
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

        # Handle both list and dict payloads
        if isinstance(payload, dict):
            # Single dict response - wrap it in a list
            payload = [payload]
        elif not isinstance(payload, list):
            raise ValueError("Frame applicator expected a JSON array or object response.")

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
            # Merge any upstream metadata attached to this passage_id
            extra_meta = self._current_metadata.get(passage_id)
            if isinstance(extra_meta, dict):
                for key, value in extra_meta.items():
                    if key not in metadata or not metadata[key]:
                        metadata[key] = value
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

    def _save_resolved_messages_batch(self, messages: List[Dict[str, str]], batch_index: int) -> None:
        """Save a single batch of resolved messages to disk."""
        if self._resolved_messages_dir is None:
            return
        self._resolved_messages_dir.mkdir(parents=True, exist_ok=True)
        for msg in messages:
            role = str(msg.get("role", "unknown")).lower()
            content = str(msg.get("content", ""))
            out_path = self._resolved_messages_dir / f"{self._resolved_messages_prefix}_{batch_index:03d}_{role}.txt"
            out_path.write_text(content, encoding="utf-8")

    def _clone_assignment(self, assignment: FrameAssignment, new_passage_id: str | None = None) -> FrameAssignment:
        passage_id_to_use = new_passage_id if new_passage_id is not None else assignment.passage_id
        
        # Update metadata if we have a new passage_id
        metadata = copy.deepcopy(assignment.metadata)
        if new_passage_id is not None:
            corpus_name, local_doc_id, _ = split_passage_id(new_passage_id)
            metadata["doc_id"] = local_doc_id
            metadata["global_doc_id"] = make_global_doc_id(corpus_name, local_doc_id) if corpus_name else local_doc_id
            if corpus_name:
                metadata["corpus"] = corpus_name
            elif "corpus" in metadata:
                del metadata["corpus"]
        
        return FrameAssignment(
            passage_id=passage_id_to_use,
            passage_text=assignment.passage_text,
            probabilities=dict(assignment.probabilities),
            top_frames=list(assignment.top_frames),
            rationale=assignment.rationale,
            evidence_spans=list(assignment.evidence_spans),
            metadata=metadata,
        )
