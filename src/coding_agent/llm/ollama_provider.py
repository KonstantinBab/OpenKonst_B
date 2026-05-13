"""Ollama provider."""

from __future__ import annotations

import time
import subprocess
from typing import Any

import httpx

from coding_agent.llm.base import BaseLLMProvider, ChatMessage, LLMResponse
from coding_agent.util.errors import LLMResponseError


class OllamaProvider(BaseLLMProvider):
    """Talks to a local Ollama server."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: int = 90,
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
        warmup_timeout_seconds: int = 90,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.warmup_timeout_seconds = warmup_timeout_seconds

    def chat(self, messages: list[ChatMessage], json_mode: bool = False) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [message.model_dump() for message in messages],
            "stream": False,
        }
        if json_mode:
            payload["format"] = "json"
        try:
            response = self._request_with_retries(payload)
            data = response.json()
            return LLMResponse(content=data["message"]["content"], raw=data)
        except LLMResponseError as exc:
            if not self._should_fallback_to_cli(str(exc)):
                raise
            return self._chat_via_cli(messages, json_mode=json_mode)

    @staticmethod
    def _should_fallback_to_cli(error_text: str) -> bool:
        return "503" in error_text or "HTTP 500" in error_text

    def _request_with_retries(self, payload: dict[str, Any]) -> httpx.Response:
        last_exc: Exception | None = None
        attempts = max(self.max_retries, 1)
        for attempt in range(1, attempts + 1):
            try:
                response = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_seconds)
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if exc.response.status_code == 503 and attempt < attempts:
                    self._warmup_model()
                    time.sleep(self.retry_backoff_seconds * attempt)
                    continue
                raise LLMResponseError(self._format_http_error(exc)) from exc
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt < attempts:
                    time.sleep(self.retry_backoff_seconds * attempt)
                    continue
                raise LLMResponseError(
                    f"Failed to reach Ollama at {self.base_url}. "
                    "Check that Ollama is running and reachable on Windows localhost."
                ) from exc
        assert last_exc is not None
        raise LLMResponseError(f"Ollama request failed after {attempts} attempts: {last_exc}")

    def _chat_via_cli(self, messages: list[ChatMessage], json_mode: bool = False) -> LLMResponse:
        prompt = self._messages_to_prompt(messages)
        if json_mode:
            prompt = (
                "Return valid JSON only. Do not use markdown fences. "
                "Do not add commentary outside JSON.\n\n"
                + prompt
            )
        try:
            timeout_seconds = self._cli_timeout_seconds(prompt)
            result = subprocess.run(
                ["ollama", "run", self.model],
                capture_output=True,
                input=prompt,
                text=True,
                timeout=timeout_seconds,
                encoding="utf-8",
                errors="replace",
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMResponseError(f"Ollama HTTP returned 503 and CLI fallback failed: {exc}") from exc
        if result.returncode != 0:
            error = (result.stderr or result.stdout).strip()
            raise LLMResponseError(f"Ollama HTTP returned 503 and CLI fallback exited {result.returncode}: {error}")
        content = self._strip_cli_spinner(result.stdout)
        return LLMResponse(content=content, raw={"provider": "ollama_cli", "stderr": result.stderr})

    def _cli_timeout_seconds(self, prompt: str) -> int:
        prompt_chars = len(prompt)
        base_timeout = max(self.timeout_seconds * 3, 300)
        if prompt_chars > 40_000:
            return max(base_timeout, 900)
        if prompt_chars > 20_000:
            return max(base_timeout, 720)
        if prompt_chars > 8_000:
            return max(base_timeout, 480)
        return base_timeout

    @staticmethod
    def _messages_to_prompt(messages: list[ChatMessage]) -> str:
        return "\n\n".join(f"{message.role.upper()}:\n{message.content}" for message in messages)

    @staticmethod
    def _strip_cli_spinner(output: str) -> str:
        cleaned = output.replace("\x1b[?2026h", "").replace("\x1b[?2026l", "").replace("\x1b[?25l", "").replace("\x1b[?25h", "")
        return "\n".join(line for line in cleaned.splitlines() if not line.strip().startswith("⠙")).strip()

    def warmup(self) -> None:
        """Wait until the configured Ollama model accepts chat requests."""
        deadline = time.monotonic() + max(self.warmup_timeout_seconds, 1)
        last_error: str | None = None
        while time.monotonic() < deadline:
            try:
                response = self._warmup_request()
                response.raise_for_status()
                return
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    raise LLMResponseError(self._format_http_error(exc)) from exc
                if exc.response.status_code == 503:
                    tag_probe = self._probe_tags()
                    if tag_probe == "missing_model":
                        raise LLMResponseError(
                            f"Ollama model '{self.model}' is not installed on {self.base_url}. "
                            f"Run `ollama pull {self.model}` or override the model with `agent run --model ...`."
                        ) from exc
                    if "/api/tags HTTP 503" in tag_probe:
                        self._warmup_via_cli()
                        return
                    last_error = f"/api/chat HTTP 503; {tag_probe}"
                else:
                    last_error = f"/api/chat HTTP {exc.response.status_code}"
                    raise LLMResponseError(self._format_http_error(exc)) from exc
            except httpx.HTTPError as exc:
                last_error = str(exc)
            time.sleep(self.retry_backoff_seconds)
        if last_error and "503" in last_error:
            self._warmup_via_cli()
            return
        raise LLMResponseError(
            f"Ollama model '{self.model}' did not become ready within {self.warmup_timeout_seconds} seconds. "
            f"Last observed error: {last_error or 'unknown'}."
        )

    def _warmup_via_cli(self) -> None:
        self._chat_via_cli([ChatMessage(role="user", content="Reply with OK only.")])

    def _warmup_model(self) -> None:
        """Try to load the model into memory before retrying the real request."""
        try:
            self.warmup()
        except LLMResponseError:
            return

    def _warmup_request(self) -> httpx.Response:
        warmup_payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "Reply with OK only."}],
            "stream": False,
        }
        return httpx.post(f"{self.base_url}/api/chat", json=warmup_payload, timeout=min(self.timeout_seconds, 30))

    def _probe_tags(self) -> str:
        """Inspect the Ollama tags endpoint for more actionable readiness errors."""
        try:
            response = httpx.get(f"{self.base_url}/api/tags", timeout=min(self.timeout_seconds, 15))
            if response.status_code == 200:
                if self._model_installed(response):
                    return "/api/tags HTTP 200"
                return "missing_model"
            return f"/api/tags HTTP {response.status_code}"
        except httpx.HTTPError as exc:
            return f"/api/tags error: {exc}"

    def _model_installed(self, response: httpx.Response) -> bool:
        try:
            data = response.json()
        except ValueError:
            return True
        models = data.get("models")
        if not isinstance(models, list):
            return True
        installed_names = {
            str(item.get("name") or item.get("model") or "").strip()
            for item in models
            if isinstance(item, dict)
        }
        if not installed_names:
            return True
        return self.model in installed_names

    def _format_http_error(self, exc: httpx.HTTPStatusError) -> str:
        status = exc.response.status_code
        if status == 404:
            return (
                f"Ollama model '{self.model}' was not found on {self.base_url}. "
                f"Run `ollama pull {self.model}` or override the model with `agent run --model ...`."
            )
        if status == 503:
            return (
                f"Ollama returned 503 for model '{self.model}'. "
                "The local runner is available but the chat endpoint is temporarily unavailable. "
                "This usually means the model is still loading, Ollama is overloaded, or the server needs a retry. "
                f"Try `ollama ps`, retry the command, or run with a different installed model via `--model`."
            )
        return f"Ollama returned HTTP {status} for model '{self.model}' at {self.base_url}."
