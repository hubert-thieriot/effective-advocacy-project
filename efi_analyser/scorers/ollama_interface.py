"""Minimal Ollama chat interface used in tests and local workflows."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import requests
from dotenv import load_dotenv

from cache.llm_cache_manager import get_cache_manager

load_dotenv()

DEFAULT_PRIORITY_MODELS: tuple[str, ...] = (
    "llama3.2:1b",
    "llama3.2:3b",
    "phi3:3.8b",
    "mistral:7b",
    "gemma3:4b",
)


@dataclass
class OllamaConfig:
    """Configuration for the Ollama interface."""

    model: Optional[str] = None
    base_url: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    temperature: float = 0.0
    timeout: float = 60.0
    max_retries: int = 2
    retry_delay: float = 1.5
    cache_dir: Path = Path("cache") / "ollama"
    ignore_cache: bool = False
    verbose: bool = False
    preferred_models: Sequence[str] = field(default_factory=lambda: DEFAULT_PRIORITY_MODELS)


class OllamaInterface:
    """Very small helper around the Ollama HTTP API (OpenAI-compatible)."""

    def __init__(self, name: str = "ollama_interface", config: Optional[OllamaConfig] = None) -> None:
        self.name = name
        self.config = config or OllamaConfig()
        self.base_url = self.config.base_url.rstrip("/")
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_manager = get_cache_manager()
        self.model = self.config.model or self.select_fast_model(self.base_url, self.config.preferred_models)
        if not self.model:
            raise RuntimeError(
                "No Ollama models available. Run `ollama pull llama3.2:1b` (or similar) and retry."
            )

    # --------------------------------------------------------------------- utils
    @staticmethod
    def list_models(base_url: str) -> List[str]:
        """List models available on the Ollama daemon."""
        try:
            response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
            response.raise_for_status()
            payload = response.json()
            return [entry["name"] for entry in payload.get("models", []) if "name" in entry]
        except Exception:
            return []

    @staticmethod
    def select_fast_model(base_url: str, priority: Sequence[str]) -> Optional[str]:
        """Select the first installed model from the preferred list."""
        available = OllamaInterface.list_models(base_url)
        if not available:
            return None
        for candidate in priority:
            if candidate in available:
                return candidate
        return available[0]

    @staticmethod
    def is_available(base_url: str) -> bool:
        """Quick health check for the Ollama daemon."""
        try:
            response = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=3)
            return response.ok
        except Exception:
            return False

    # --------------------------------------------------------------------- public
    def spec_key(self) -> str:
        payload = {
            "name": self.name,
            "model": self.model,
            "temperature": self.config.temperature,
            "base_url": self.base_url,
        }
        serialized = json.dumps(payload, sort_keys=True)
        import hashlib

        return hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:12]

    def infer(self, messages: List[Dict[str, str]], timeout: Optional[float] = None) -> str:
        """Send a chat completion request to the Ollama daemon."""
        return self._inference_with_cache(messages, timeout=timeout)

    # ------------------------------------------------------------------ internals
    def _inference_with_cache(self, messages: List[Dict[str, str]], timeout: Optional[float]) -> str:
        parameters = {
            "temperature": self.config.temperature,
            "timeout": timeout or self.config.timeout,
        }
        if not self.config.ignore_cache:
            cached_entry = self.cache_manager.get(self.model, messages, parameters)
            if cached_entry is not None:
                if self.config.verbose:
                    print(f"📋 Using cached Ollama response for {self.name} ({self.model})")
                return cached_entry.response

        if self.config.verbose:
            print(f"🫎 Calling Ollama model {self.model} via {self.base_url} ...")

        start = time.time()
        response = self._call_ollama(messages, timeout=timeout)
        elapsed = time.time() - start

        if self.config.verbose:
            print(f"✅ Ollama call finished in {elapsed:.2f}s for {self.model}")

        self.cache_manager.put(
            model=self.model,
            messages=messages,
            parameters=parameters,
            response=response,
            inference_time=elapsed,
        )
        return response

    def _call_ollama(self, messages: List[Dict[str, str]], timeout: Optional[float]) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "stream": False,
            "messages": messages,
            "options": {"temperature": self.config.temperature},
        }
        final_timeout = timeout or self.config.timeout
        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(url, json=payload, timeout=final_timeout)
                response.raise_for_status()
                data = response.json()
                if "message" in data and data["message"].get("content"):
                    return data["message"]["content"]
                if "response" in data:
                    return data["response"]
                raise ValueError("Unexpected Ollama response payload.")
            except Exception as exc:
                if attempt == self.config.max_retries - 1:
                    raise RuntimeError(f"Ollama request failed after {self.config.max_retries} attempts: {exc}") from exc
                if self.config.verbose:
                    print(f"⚠️ Ollama call failed (attempt {attempt + 1}): {exc}")
                time.sleep(self.config.retry_delay * (2**attempt))
        raise RuntimeError("Unreachable due to retry logic.")




