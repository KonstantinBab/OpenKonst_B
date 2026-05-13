"""vLLM provider for high-performance local inference with multi-model support."""

from __future__ import annotations

import time
from typing import Any

import httpx

from coding_agent.llm.base import BaseLLMProvider, ChatMessage, LLMResponse, ModelComplexity
from coding_agent.util.errors import LLMResponseError


class VLLMProvider(BaseLLMProvider):
    """Calls a vLLM OpenAI-compatible server with retry, warmup and multi-model routing support."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: int = 120,
        max_retries: int = 6,
        retry_backoff_seconds: float = 3.0,
        warmup_timeout_seconds: int = 90,
        api_key: str | None = None,
        simple_model: str | None = None,
        complex_model: str | None = None,
        chat_model: str | None = None,
        auto_switch_enabled: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.simple_model = simple_model or model
        self.complex_model = complex_model or model
        self.chat_model = chat_model or model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.warmup_timeout_seconds = warmup_timeout_seconds
        self.api_key = api_key
        self.auto_switch_enabled = auto_switch_enabled

    def _select_model(self, messages: list[ChatMessage], model_override: str | None = None) -> tuple[str, ModelComplexity]:
        """Select appropriate model based on task complexity."""
        if model_override:
            return model_override, ModelComplexity.MEDIUM
        
        if not self.auto_switch_enabled:
            return self.model, ModelComplexity.MEDIUM
        
        # Analyze the last user message for complexity
        prompt = ""
        for msg in reversed(messages):
            if msg.role == "user":
                prompt = msg.content
                break
        
        complexity = self.estimate_complexity(prompt)
        
        if complexity == ModelComplexity.COMPLEX:
            return self.complex_model, complexity
        elif complexity == ModelComplexity.SIMPLE:
            return self.simple_model, complexity
        else:
            return self.model, complexity

    def chat(self, messages: list[ChatMessage], json_mode: bool = False, model_override: str | None = None) -> LLMResponse:
        selected_model, complexity = self._select_model(messages, model_override)
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload: dict[str, Any] = {
            "model": selected_model,
            "messages": [message.model_dump() for message in messages],
            "temperature": 0.2,
            "max_tokens": 4096,
        }
        
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        
        last_exc: Exception | None = None
        attempts = max(self.max_retries, 1)
        
        for attempt in range(1, attempts + 1):
            try:
                response = httpx.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                data = response.json()
                
                if not data.get("choices"):
                    raise LLMResponseError("vLLM returned empty choices")
                
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                return LLMResponse(
                    content=content,
                    raw=data,
                    model_used=selected_model,
                    tokens_used=usage.get("total_tokens"),
                    complexity_level=complexity,
                )
                
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if exc.response.status_code == 503 and attempt < attempts:
                    # Model is loading, wait and retry
                    time.sleep(self.retry_backoff_seconds * attempt)
                    continue
                if exc.response.status_code == 404:
                    raise LLMResponseError(
                        f"Model '{selected_model}' not found on vLLM server at {self.base_url}. "
                        f"Check that the model is loaded: curl {self.base_url}/v1/models"
                    ) from exc
                raise LLMResponseError(f"vLLM HTTP {exc.response.status_code}: {exc}") from exc
                
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt < attempts:
                    time.sleep(self.retry_backoff_seconds * attempt)
                    continue
                raise LLMResponseError(
                    f"Failed to reach vLLM at {self.base_url}. "
                    "Ensure vLLM server is running: python -m vllm.entrypoints.api_server --model <model>"
                ) from exc
        
        assert last_exc is not None
        raise LLMResponseError(f"vLLM request failed after {attempts} attempts: {last_exc}")

    def warmup(self) -> None:
        """Wait until the vLLM model is ready to accept requests."""
        deadline = time.monotonic() + max(self.warmup_timeout_seconds, 1)
        last_error: str | None = None
        
        while time.monotonic() < deadline:
            try:
                headers = {}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                    
                response = httpx.get(
                    f"{self.base_url}/v1/models",
                    headers=headers,
                    timeout=min(self.timeout_seconds, 30),
                )
                response.raise_for_status()
                data = response.json()
                
                models = data.get("data", [])
                model_ids = {m.get("id") for m in models if isinstance(m, dict)}
                
                if self.model in model_ids or any(self.model.split(":")[0] in mid for mid in model_ids):
                    # Model is available, do a test request
                    self._warmup_request(headers)
                    return
                    
                last_error = f"Model '{self.model}' not in available models: {model_ids}"
                
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 503:
                    last_error = "vLLM server is busy loading models"
                elif exc.response.status_code == 404:
                    raise LLMResponseError(
                        f"vLLM endpoint not found at {self.base_url}/v1/models. "
                        "Ensure vLLM is running with OpenAI-compatible API."
                    ) from exc
                else:
                    last_error = f"vLLM HTTP {exc.response.status_code}"
                    
            except httpx.HTTPError as exc:
                last_error = str(exc)
            
            time.sleep(self.retry_backoff_seconds)
        
        raise LLMResponseError(
            f"vLLM model '{self.model}' did not become ready within {self.warmup_timeout_seconds} seconds. "
            f"Last observed error: {last_error or 'unknown'}. "
            f"Start vLLM with: python -m vllm.entrypoints.api_server --model {self.model}"
        )

    def _warmup_request(self, headers: dict[str, str]) -> None:
        """Send a minimal request to warm up the model."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "OK"}],
            "max_tokens": 10,
        }
        response = httpx.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=min(self.timeout_seconds, 30),
        )
        response.raise_for_status()
