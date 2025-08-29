"""
Concrete OllamaReScorer using Ollama's OpenAI-compatible local server.
"""

from __future__ import annotations

from typing import Dict, List

from .llm_rescorer import LLMReScorer, LLMReScorerConfig


class OllamaReScorer(LLMReScorer):
    """LLM re-scorer using Ollama's OpenAI-compatible API.

    Configure your OpenAI client with base_url pointing to the local server.
    For example (Ollama default): base_url="http://localhost:11434/v1"
    The environment variable OPENAI_BASE_URL can be used by the OpenAI client.
    """

    def __init__(self, config: LLMReScorerConfig | None = None, base_url: str | None = None, api_key: str | None = None):
        super().__init__(config)
        # Lazy import to keep dependency optional
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency optional
            raise RuntimeError("openai package is required for OllamaReScorer") from exc

        # Use sensible defaults for local Ollama
        if base_url is None:
            base_url = "http://localhost:11434/v1"
        elif not base_url.endswith('/v1'):
            base_url = base_url.rstrip('/') + '/v1'
        
        # For local Ollama, we need a dummy API key (OpenAI client requirement)
        if api_key is None and base_url == "http://localhost:11434/v1":
            api_key = "ollama-local"
        
        # Allow caller/env to provide connection details
        self._client = OpenAI(
            base_url=base_url, 
            api_key=api_key,  # Use dummy key for local Ollama
            timeout=120.0  # 120 second timeout for slower models like phi3:3.8b
        )

    def _inference(self, messages: List[Dict[str, str]]) -> str:
        # Deterministic decoding by default; rely on base config
        resp = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        return resp.choices[0].message.content or ""


