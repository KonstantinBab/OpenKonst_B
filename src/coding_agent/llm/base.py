"""LLM provider abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class LLMResponse(BaseModel):
    content: str
    raw: dict[str, Any] = Field(default_factory=dict)


class BaseLLMProvider(ABC):
    """Abstract chat-completion provider."""

    @abstractmethod
    def chat(self, messages: list[ChatMessage], json_mode: bool = False) -> LLMResponse:
        raise NotImplementedError

    def warmup(self) -> None:
        """Optional provider-specific readiness check."""
        return None
