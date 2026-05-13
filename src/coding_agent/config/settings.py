"""Pydantic settings models with vLLM and advanced features."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class VLLMSettings(BaseModel):
    """vLLM configuration for production workloads."""
    enabled: bool = False
    base_url: str = "http://127.0.0.1:8000"
    model: str = "Qwen/Qwen2.5-Coder-32B-Instruct"
    tensor_parallel_size: int = 2
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.9


class LLMSettings(BaseModel):
    provider: Literal["ollama", "vllm", "openai_compatible"] = "ollama"
    model: str = "qwen2.5-coder:7b"
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: int = 180
    max_retries: int = 10
    retry_backoff_seconds: float = 5.0
    warmup_timeout_seconds: int = 180
    api_key: str | None = None
    # Модели для разных типов задач
    simple_model: str = "qwen2.5-coder:7b"  # Для простых задач
    complex_model: str = "qwen3.6:27b"  # Для сложных задач
    chat_model: str = "qwen3.5:9b"  # Для чата
    # Автоматическое переключение моделей
    auto_switch_enabled: bool = True
    complexity_threshold: int = 500  # Порог сложности для переключения (символы)
    max_context_tokens: int = 32768
    # vLLM конфигурация
    vllm: VLLMSettings = Field(default_factory=VLLMSettings)


class OrchestratorSettings(BaseModel):
    max_steps: int = 48
    max_tool_calls: int = 192
    retrieval_max_chunks: int = 16
    auto_test_enabled: bool = True
    test_on_command_execution: bool = True
    smart_context_window: bool = True
    max_context_chars: int = 120000


class VerifyProfile(BaseModel):
    tests: list[str] = Field(default_factory=list)
    lint: list[str] = Field(default_factory=list)
    type_check: list[str] = Field(default_factory=list)
    build: list[str] = Field(default_factory=list)


class VerifierSettings(BaseModel):
    default_profile: str = "generic"
    auto_detect_profile: bool = True
    fail_fast: bool = False
    python: VerifyProfile = Field(default_factory=VerifyProfile)
    node: VerifyProfile = Field(default_factory=VerifyProfile)
    ml_project: VerifyProfile = Field(default_factory=VerifyProfile)
    generic: VerifyProfile = Field(default_factory=VerifyProfile)


class AdvancedSettings(BaseModel):
    """Advanced features for Codex-like functionality."""
    enable_deep_analysis: bool = True
    enable_incremental_changes: bool = True
    enable_self_correction: bool = True
    max_retry_on_failure: int = 3
    context_awareness: Literal["low", "medium", "high"] = "high"
    parallel_tool_execution: bool = False


class AppConfig(BaseModel):
    app_name: str = "coding-agent"
    log_level: str = "INFO"
    memory_db_path: Path = Path("./data/agent_memory.db")
    llm: LLMSettings = Field(default_factory=LLMSettings)
    orchestrator: OrchestratorSettings = Field(default_factory=OrchestratorSettings)
    verifier: VerifierSettings = Field(default_factory=VerifierSettings)
    advanced: AdvancedSettings = Field(default_factory=AdvancedSettings)


class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CODING_AGENT_", extra="ignore")

    config: Path = Path("./config/default.yaml")
    policy: Path = Path("./config/policy.yaml")
    memory_db: Path | None = None
    log_level: str | None = None
    ollama_base_url: str | None = None
    ollama_model: str | None = None
    vllm_base_url: str | None = None
    vllm_model: str | None = None
    # Переключение моделей
    complex_model: str | None = None
    chat_model: str | None = None
    auto_switch_enabled: bool | None = None
