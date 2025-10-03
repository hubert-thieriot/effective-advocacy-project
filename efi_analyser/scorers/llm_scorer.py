"""
LLM scorer interface and implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import os
import time
import requests

from efi_core.types import PairScorer, Task
import sys
from pathlib import Path

# Add project root to path for cache manager
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from cache.llm_cache_manager import get_cache_manager


@dataclass
class LLMScorerConfig:
    model: str = "llama3"
    prompt_version: str = "v1"
    temperature: float = 0.0
    top_p: float = 1.0
    max_rationale_words: int = 30
    cache_dir: Path = Path("cache") / "scorer"
    timeout_s: float = 60.0
    batch_size: int = 50
    max_workers: int = 4
    show_progress: bool = True
    ignore_cache: bool = False  # New parameter to bypass cache
    base_url: Optional[str] = None  # Ollama/OpenAI base URL
    api_key: Optional[str] = None  # API key for authentication
    verbose: bool = False  # Enable verbose output for inference progress


class LLMInterface:
    """Pure LLM interface for sending messages and getting responses.

    This class provides a clean interface for LLM interactions with caching.
    Subclasses must implement `_inference(messages)` to call the underlying LLM.
    """

    def __init__(self, name: str = "llm_scorer", config: Optional[LLMScorerConfig] = None):
        self.name = name
        self.config = config or LLMScorerConfig()
        # Ensure cache directory exists
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

        # Set up API connection details
        self.base_url = self.config.base_url or os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
        self.api_key = self.config.api_key or os.getenv("LLM_API_KEY", "ollama")
        
        # Initialize cache manager
        self.cache_manager = get_cache_manager()

    # -------- Public API --------
    def infer(self, messages: List[Dict[str, str]]) -> str:
        """Send messages to LLM and get raw response string.

        Args:
            messages: List of message dictionaries with 'role' and 'content'

        Returns:
            Raw response string from LLM
        """
        return self._inference_with_cache(messages)


    def _inference(self, messages: List[Dict[str, str]]) -> str:
        """Execute a chat completion against an LLM and return raw text.

        Uses OpenAI-compatible API (works with Ollama).
        """
        try:
            url = f"{self.base_url}/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            data = {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
            }

            response = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=self.config.timeout_s
            )

            if response.status_code != 200:
                raise Exception(f"LLM API error: {response.status_code} - {response.text}")

            result = response.json()
            if "choices" not in result or not result["choices"]:
                raise Exception(f"Invalid LLM response format: {result}")

            return result["choices"][0]["message"]["content"]

        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to connect to LLM server at {self.base_url}: {e}")
        except Exception as e:
            raise Exception(f"LLM inference failed: {e}")

    # -------- Internals --------

    def _inference_with_cache(self, messages: List[Dict[str, str]]) -> str:
        """Get raw response from LLM with centralized caching."""
        # Prepare parameters for cache lookup
        parameters = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "prompt_version": self.config.prompt_version
        }
        
        # Check cache first
        if not self.config.ignore_cache:
            cached_entry = self.cache_manager.get(self.config.model, messages, parameters)
            if cached_entry is not None:
                if self.config.verbose:
                    print(f"📋 Using cached response for {self.name} ({self.config.model})")
                return cached_entry.response

        # Make inference call
        if self.config.verbose:
            print(f"🚀 Making inference call to {self.name} ({self.config.model})...")
        start_time = time.time()
        raw_text = self._safe_infer(messages)
        inference_time = time.time() - start_time
        if self.config.verbose:
            print(f"✅ Inference completed in {inference_time:.2f}s for {self.name} ({self.config.model})")

        # Cache the response using centralized cache manager
        self.cache_manager.put(
            model=self.config.model,
            messages=messages,
            parameters=parameters,
            response=raw_text,
            inference_time=inference_time
        )
        
        return raw_text

    def _safe_infer(self, messages: List[Dict[str, str]]) -> str:
        try:
            return self._inference(messages)
        except Exception as exc:
            return json.dumps({"score": 0.0, "rationale": f"Error: {exc}"})

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


