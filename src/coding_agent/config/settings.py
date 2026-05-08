"""Pydantic settings models."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseModel):
    provider: Literal["ollama", "openai_compatible"] = "ollama"
    model: str = "qwen2.5-coder:7b"
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: int = 90
    max_retries: int = 6
    retry_backoff_seconds: float = 3.0
    warmup_timeout_seconds: int = 90


class OrchestratorSettings(BaseModel):
    max_steps: int = 8
    max_tool_calls: int = 24
    retrieval_max_chunks: int = 8


class VerifyProfile(BaseModel):
    tests: list[str] = Field(default_factory=list)
    lint: list[str] = Field(default_factory=list)
    type_check: list[str] = Field(default_factory=list)
    build: list[str] = Field(default_factory=list)


class VerifierSettings(BaseModel):
    default_profile: str = "generic"
    python: VerifyProfile = Field(default_factory=VerifyProfile)
    node: VerifyProfile = Field(default_factory=VerifyProfile)
    generic: VerifyProfile = Field(default_factory=VerifyProfile)


class AppConfig(BaseModel):
    app_name: str = "coding-agent"
    log_level: str = "INFO"
    memory_db_path: Path = Path("./data/agent_memory.db")
    llm: LLMSettings = Field(default_factory=LLMSettings)
    orchestrator: OrchestratorSettings = Field(default_factory=OrchestratorSettings)
    verifier: VerifierSettings = Field(default_factory=VerifierSettings)


class EnvSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CODING_AGENT_", extra="ignore")

    config: Path = Path("./config/default.yaml")
    policy: Path = Path("./config/policy.yaml")
    memory_db: Path | None = None
    log_level: str | None = None
    ollama_base_url: str | None = None
    ollama_model: str | None = None
