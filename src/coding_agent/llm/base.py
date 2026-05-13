"""LLM provider abstractions with multi-model support."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from enum import Enum

from pydantic import BaseModel, Field


class ModelComplexity(Enum):
    """Complexity levels for model selection."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class ChatMessage(BaseModel):
    role: str
    content: str


class LLMResponse(BaseModel):
    content: str
    raw: dict[str, Any] = Field(default_factory=dict)
    model_used: str | None = None
    tokens_used: int | None = None
    complexity_level: ModelComplexity | None = None


class BaseLLMProvider(ABC):
    """Abstract chat-completion provider with multi-model support."""

    @abstractmethod
    def chat(self, messages: list[ChatMessage], json_mode: bool = False, model_override: str | None = None) -> LLMResponse:
        raise NotImplementedError

    def warmup(self) -> None:
        """Optional provider-specific readiness check."""
        return None
    
    def estimate_complexity(self, prompt: str) -> ModelComplexity:
        """Estimate task complexity based on prompt analysis."""
        char_count = len(prompt)
        line_count = prompt.count('\n')
        
        # Keywords indicating complex tasks
        complex_keywords = [
            'архитектура', 'architecture', 'рефакторинг', 'refactor',
            'оптимизация', 'optimization', 'масштабирование', 'scalability',
            'деплой', 'deploy', 'интеграция', 'integration',
            'анализ', 'analysis', 'проектирование', 'design',
            'сложн', 'complex', 'multi', 'distributed'
        ]
        
        has_complex_keywords = any(kw in prompt.lower() for kw in complex_keywords)
        
        if char_count > 1000 or line_count > 50 or has_complex_keywords:
            return ModelComplexity.COMPLEX
        elif char_count > 300 or line_count > 20:
            return ModelComplexity.MEDIUM
        else:
            return ModelComplexity.SIMPLE
