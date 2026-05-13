"""OpenAI-compatible provider with multi-model support."""

from __future__ import annotations

from typing import Any

import httpx

from coding_agent.llm.base import BaseLLMProvider, ChatMessage, LLMResponse, ModelComplexity


class OpenAICompatibleProvider(BaseLLMProvider):
    """Calls a chat-completions endpoint that follows the OpenAI-compatible schema with multi-model routing."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: int = 90,
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
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        response = httpx.post(f"{self.base_url}/chat/completions", json=payload, headers=headers, timeout=self.timeout_seconds)
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage", {})
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            raw=data,
            model_used=selected_model,
            tokens_used=usage.get("total_tokens"),
            complexity_level=complexity,
        )

