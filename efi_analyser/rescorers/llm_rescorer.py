"""
LLM-based re-scorer base class with disk caching and hybrid prompt.

This module defines a reusable `LLMReScorer` that implements:
- Prompt assembly (system+user) for 0..1 derive-from scoring
- Strict JSON parsing and clamping
- Deterministic decoding defaults (temperature=0.0)
- Disk cache under `cache/rescorer/`
- Batch processing with parallel workers
- Automatic timing metadata

Concrete subclasses must implement `_inference(messages: list) -> str`.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import os
import time

from efi_core.protocols import ReScorer
from efi_core.retrieval.retriever import SearchResult


@dataclass
class LLMReScorerConfig:
    model: str = "llama3"
    prompt_version: str = "v1"
    temperature: float = 0.0
    top_p: float = 1.0
    max_rationale_words: int = 30
    cache_dir: Path = Path("cache") / "rescorer"
    timeout_s: float = 60.0
    batch_size: int = 50
    max_workers: int = 4
    show_progress: bool = True
    ignore_cache: bool = False  # New parameter to bypass cache


class LLMReScorer(ReScorer[SearchResult]):
    """Base class for LLM-backed rescoring to [0,1] using derive-from prompt.

    Subclasses must implement `_inference(messages)` to call the underlying LLM
    and return a raw string response.
    """

    def __init__(self, config: Optional[LLMReScorerConfig] = None):
        self.config = config or LLMReScorerConfig()
        # Ensure cache directory exists
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def name(self) -> str:
        """Get a unique name for this rescorer instance."""
        return self.config.model

    # -------- Public API --------
    def rescore(self, query: str, matches: List[SearchResult]) -> List[SearchResult]:
        if not matches:
            return matches

        # Process in batches if configured
        if self.config.batch_size > 1 and len(matches) > self.config.batch_size:
            return self._rescore_in_batches(query, matches)
        
        return self._rescore_single_batch(query, matches)

    def _rescore_single_batch(self, query: str, matches: List[SearchResult]) -> List[SearchResult]:
        """Rescore a single batch of matches sequentially."""
        rescored: List[SearchResult] = []
        batch_start_time = time.time()

        for match in matches:
            chunk_text = match.metadata.get("text", "")
            if not chunk_text or not query:
                rescored.append(match)
                continue

            messages = self._build_messages(query, chunk_text)
            params = {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "prompt_version": self.config.prompt_version,
                "model": self.config.model,
            }

            # Time the scoring process
            start_time = time.time()
            score, rationale, raw_text = self._score_with_cache(messages, params)
            scoring_time = time.time() - start_time

            new_metadata = dict(match.metadata)
            new_metadata.update({
                "llm_model": self.config.model,
                "llm_prompt_version": self.config.prompt_version,
                "llm_score": score,
                "llm_rationale": rationale,
                "llm_scoring_time": scoring_time,
            })

            rescored.append(
                SearchResult(item_id=match.item_id, score=float(score), metadata=new_metadata)
            )

        batch_time = time.time() - batch_start_time
        # Only print timing for non-cached results (when it takes significant time)
        if batch_time > 0.1:  # Only show if it took more than 0.1 seconds
            print(f"⏱️ {len(matches)} matches in {batch_time:.1f}s")
        
        rescored.sort(key=lambda x: x.score, reverse=True)
        return rescored

    def _rescore_in_batches(self, query: str, matches: List[SearchResult]) -> List[SearchResult]:
        """Rescore matches in parallel batches."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        all_rescored = []
        batch_size = self.config.batch_size
        max_workers = self.config.max_workers
        
        # Create batches
        batches = [matches[i:i + batch_size] for i in range(0, len(matches), batch_size)]
        
        print(f"🔄 Processing {len(matches)} matches in {len(batches)} batches with {max_workers} workers")
        
        # Progress tracking
        if self.config.show_progress:
            try:
                from tqdm import tqdm
                progress_bar = tqdm(total=len(batches), desc="Rescoring batches")
            except ImportError:
                progress_bar = None
        else:
            progress_bar = None
        
        # Track overall timing
        overall_start = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit batch processing tasks
            future_to_batch = {
                executor.submit(self._rescore_single_batch, query, batch): i 
                for i, batch in enumerate(batches)
            }
            
            # Process completed batches
            for future in as_completed(future_to_batch):
                batch_index = future_to_batch[future]
                try:
                    batch_results = future.result()
                    all_rescored.extend(batch_results)
                except Exception as exc:
                    print(f"⚠️ Error processing batch {batch_index}: {exc}")
                    # Fallback: add original matches for failed batch
                    batch = batches[batch_index]
                    all_rescored.extend(batch)
                
                # Update progress
                if progress_bar:
                    progress_bar.update(1)
        
        if progress_bar:
            progress_bar.close()
        
        overall_time = time.time() - overall_start
        # Only print if it took significant time
        if overall_time > 1.0:
            print(f"⏱️ Total batch processing: {overall_time:.1f}s")
        
        # Final sort across all batches
        all_rescored.sort(key=lambda x: x.score, reverse=True)
        return all_rescored

    # -------- To be implemented by subclasses --------
    def _inference(self, messages: List[Dict[str, str]]) -> str:  # pragma: no cover - abstract
        """Execute a chat completion against an LLM and return raw text.

        Subclasses should raise a clear exception if the model/client is unavailable.
        """
        raise NotImplementedError

    # -------- Internals --------
    def _build_messages(self, finding_text: str, chunk_text: str) -> List[Dict[str, str]]:
        system = (
            "Rate chunk alignment with finding. Output ONLY JSON: {\"score\": <0-1>, \"rationale\": \"<30 words>\"}.\n\n"
            "RULES: NO text before/after JSON. Start with {, end with }.\n\n"
            "Scoring: 0.8-1.0=strong match, 0.5-0.79=medium, 0.2-0.49=weak, 0.0-0.19=none/contradiction."
        )

        user = (
            f"finding:\n\"{finding_text}\"\n\n"
            f"chunk:\n\"{chunk_text}\"\n\n"
            "Score alignment. Examples:\n"
            "- exact match: {\"score\": 0.98, \"rationale\": \"Perfect match\"}\n"
            "- partial: {\"score\": 0.60, \"rationale\": \"Similar but different scope\"}\n"
            "- weak: {\"score\": 0.05, \"rationale\": \"Vague similarity\"}\n"
            "- none: {\"score\": 0.00, \"rationale\": \"No relation\"}\n\n"
            "Output ONLY JSON, nothing else."
        )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _score_with_cache(self, messages: List[Dict[str, str]], params: Dict[str, Any]) -> Tuple[float, str, str]:
        key = self._make_cache_key(messages, params)
        path = self._cache_path(self.config.model, self.config.prompt_version, key)

        cached = self._read_cache(path)
        if cached is not None and not self.config.ignore_cache:
            return cached.get("score", 0.0), cached.get("rationale", ""), cached.get("raw_text", "")

        # Time the inference
        start_time = time.time()
        raw_text = self._safe_infer(messages)
        inference_time = time.time() - start_time
        
        score, rationale = self._parse_output(raw_text)

        payload = {
            "model": self.config.model,
            "prompt_version": self.config.prompt_version,
            "params": {"temperature": self.config.temperature, "top_p": self.config.top_p},
            "messages_hash": key,
            "created_at": time.time(),
            "score": score,
            "rationale": rationale,
            "raw_text": raw_text,
            "inference_time": inference_time,  # Store timing in cache
        }
        self._write_cache(path, payload)
        return score, rationale, raw_text

    def _safe_infer(self, messages: List[Dict[str, str]]) -> str:
        try:
            return self._inference(messages)
        except Exception as exc:
            return json.dumps({"score": 0.0, "rationale": f"Error: {exc}"})

    def _parse_output(self, text: str) -> Tuple[float, str]:
        try:
            # Try to find JSON in the text - look for the first { and last }
            text = text.strip()
            start = text.find('{')
            end = text.rfind('}')
            
            if start != -1 and end != -1 and end > start:
                json_text = text[start:end+1]
                obj = json.loads(json_text)
            else:
                # Fallback to original parsing
                obj = json.loads(text)
                
            score = float(obj.get("score", 0.0))
            if score < 0.0:
                score = 0.0
            if score > 1.0:
                score = 1.0
            rationale = str(obj.get("rationale", ""))
            if len(rationale.split()) > self.config.max_rationale_words:
                # keep first N words
                rationale = " ".join(rationale.split()[: self.config.max_rationale_words])
            return score, rationale
        except Exception:
            return 0.0, "Unparseable output"

    # -------- Disk cache helpers --------
    def _make_cache_key(self, messages: List[Dict[str, str]], params: Dict[str, Any]) -> str:
        payload = json.dumps(
            {"model": self.config.model, "prompt_version": self.config.prompt_version, "messages": messages, "params": params},
            sort_keys=True,
            ensure_ascii=False,
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, model: str, prompt_version: str, key: str) -> Path:
        dir_path = self.config.cache_dir / model / prompt_version / key[:2]
        return dir_path / f"{key}.json"

    def _read_cache(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            return None
        return None

    def _write_cache(self, path: Path, payload: Dict[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.parent / f".{path.name}.tmp"
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
            os.replace(tmp_path, path)
        except Exception:
            # Best-effort caching; ignore failures
            pass


