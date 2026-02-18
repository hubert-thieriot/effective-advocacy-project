"""
OpenAI interface for making API calls to OpenAI's GPT models.
"""

from __future__ import annotations

import os
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import logging
import openai
from dotenv import load_dotenv

from efi_analyser.cache.llm_cache_manager import get_cache_manager

logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Load environment variables
load_dotenv()


@dataclass
class OpenAIConfig:
    """Configuration for OpenAI interface."""
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.0
    timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 1.0
    cache_dir: Path = Path("cache") / "openai"
    ignore_cache: bool = False
    verbose: bool = False
    flex_processing: bool = False
    
    # Models that don't support custom temperature (must use default or omit parameter)
    _NO_TEMP_MODELS = {"gpt-5", "gpt-5-nano", "gpt-5-mini"}
    _FLEX_PROCESSING_MODELS = {"gpt-5", "gpt-5-nano", "gpt-5-mini"}
    
    def __post_init__(self):
        """Validate and adjust temperature based on model capabilities."""
        if self.model in self._NO_TEMP_MODELS:
            # These models don't support temperature parameter - it will be omitted from API calls
            if self.verbose and self.temperature != 1.0:
                print(f"⚠️ Model {self.model} doesn't support custom temperature, will omit temperature parameter (requested: {self.temperature})")
            # Store the requested value but it won't be used in API calls
    
    @classmethod
    def get_default_temperature(cls, model: str) -> float:
        """Get the default temperature for a given model."""
        return 1.0 if model in cls._NO_TEMP_MODELS else 0.0
    
    @classmethod
    def supports_custom_temperature(cls, model: str) -> bool:
        """Check if a model supports custom temperature values."""
        return model not in cls._NO_TEMP_MODELS
    
    @classmethod
    def supports_flex_processing(cls, model: str) -> bool:
        """Check if a model supports flex processing."""
        return model in cls._FLEX_PROCESSING_MODELS


class OpenAIInterface:
    """Interface for making OpenAI API calls with caching and error handling."""

    def __init__(self, name: str = "openai_interface", config: Optional[OpenAIConfig] = None):
        """Initialize OpenAI interface.
        
        Args:
            name: Name for this interface instance
            config: Configuration object
        """
        self.name = name
        self.config = config or OpenAIConfig()
        
        # Ensure cache directory exists
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = openai.OpenAI(api_key=api_key)
        
        # Initialize cache manager
        self.cache_manager = get_cache_manager()

    def spec_key(self) -> str:
        payload = {
            "name": self.name,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "timeout": self.config.timeout,
        }
        serialized = json.dumps(payload, sort_keys=True)
        import hashlib

        return hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:12]

    def infer(self, instructions: str, input: str) -> str:
        """Send messages to OpenAI and get response.
        
        Args:
            instructions: Instructions for the OpenAI model
            input: Input for the OpenAI model
            
        Returns:
            Raw response string from OpenAI
        """
        return self._inference_with_cache(instructions, input)

    def _inference_with_cache(self, instructions: str, input: str) -> str:
        """Get response from OpenAI with centralized caching."""
        # Prepare parameters for cache lookup
        parameters = {
            "temperature": self.config.temperature,
            "timeout": self.config.timeout
        }
        
        # Check cache first
        if not self.config.ignore_cache:
            cached_entry = self.cache_manager.get(self.config.model, instructions, input, parameters)
            if cached_entry is not None:
                # Skip cached error responses (re-fetch them)
                if self._is_error_response(cached_entry.response):
                    if self.config.verbose:
                        print(f"⚠️ Skipping cached error response, will re-fetch")
                else:
                    if self.config.verbose:
                        print(f"📋 Using cached response for {self.name} ({self.config.model})")
                    return cached_entry.response
        
        # Make API call
        if self.config.verbose:
            print(f"🚀 Making OpenAI API call to {self.name} ({self.config.model})...")
        
        start_time = time.time()
        response = self._safe_infer(instructions, input)
        inference_time = time.time() - start_time
        
        if self.config.verbose:
            print(f"✅ OpenAI API call completed in {inference_time:.2f}s for {self.name} ({self.config.model})")
        
        # Don't cache error responses (429 rate limit, insufficient credits, etc.)
        if not self._is_error_response(response):
            self.cache_manager.put(
                model=self.config.model,
                instructions=instructions,
                input=input,
                parameters=parameters,
                response=response,
                inference_time=inference_time
            )
        else:
            if self.config.verbose:
                print(f"⚠️ Skipping cache for error response")
        
        return response
    
    def _is_error_response(self, response: str) -> bool:
        """Check if response is an error that should not be cached."""
        error_indicators = [
            "rate_limit",
            "insufficient_quota",
            "RateLimitError",
            "API error"
        ]
        return any(indicator in response for indicator in error_indicators)

    def _safe_infer(self, instructions: str, input: str) -> str:
        """Make OpenAI API call with retry logic and error handling."""
        for attempt in range(self.config.max_retries):
            try:
                # Build API call parameters
                api_params = {
                    "model": self.config.model,
                    "instructions": instructions,
                    "input": input
                }
                
                # Only include temperature if the model supports it
                if self.config.supports_custom_temperature(self.config.model):
                    api_params["temperature"] = float(self.config.temperature)
                
                # Add service_tier for flex processing (50% cost reduction)
                if self.config.flex_processing and self.config.supports_flex_processing(self.config.model):
                    api_params["service_tier"] = "flex"
                
                response = self.client.with_options(timeout=self.config.timeout) \
                    .responses.create(**api_params)
                
                return response.output_text
                
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    if self.config.verbose:
                        print(f"⚠️ OpenAI API call failed (attempt {attempt + 1}/{self.config.max_retries}): {e}")
                    time.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    # Last attempt failed, return error response
                    error_msg = f"OpenAI API error after {self.config.max_retries} attempts: {e}"
                    if self.config.verbose:
                        print(f"❌ {error_msg}")
                    return json.dumps({"error": error_msg, "score": 0.0, "rationale": "API call failed"})
        
        # Fallback if max_retries is 0 or somehow loop doesn't execute
        return json.dumps({"error": "OpenAI API call failed: max_retries is 0", "score": 0.0, "rationale": "API call failed"})

    def _make_cache_key(self, instructions: str, input: str) -> str:
        """Generate cache key for messages."""
        import hashlib
        payload = json.dumps({
            "model": self.config.model,
            "instructions": instructions,
            "input": input,
            "temperature": self.config.temperature
        }, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for given key."""
        return self.config.cache_dir / f"{cache_key}.json"
