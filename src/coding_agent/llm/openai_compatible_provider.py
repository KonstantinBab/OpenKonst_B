"""OpenAI-compatible provider."""

from __future__ import annotations

from typing import Any

import httpx

from coding_agent.llm.base import BaseLLMProvider, ChatMessage, LLMResponse


class OpenAICompatibleProvider(BaseLLMProvider):
    """Calls a chat-completions endpoint that follows the OpenAI-compatible schema."""

    def __init__(self, base_url: str, model: str, timeout_seconds: int = 90, api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key

    def chat(self, messages: list[ChatMessage], json_mode: bool = False) -> LLMResponse:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [message.model_dump() for message in messages],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        response = httpx.post(f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=self.timeout_seconds)
        response.raise_for_status()
        data = response.json()
        return LLMResponse(content=data["choices"][0]["message"]["content"], raw=data)

