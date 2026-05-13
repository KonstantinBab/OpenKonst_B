"""Configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from coding_agent.config.settings import AppConfig, EnvSettings


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_app_config(config_path: Path | None = None) -> AppConfig:
    """Load YAML config and apply environment overrides."""
    env = EnvSettings()
    resolved_path = config_path or env.config
    data = _load_yaml(resolved_path)
    config = AppConfig.model_validate(data)
    if env.memory_db:
        config.memory_db_path = env.memory_db
    if env.log_level:
        config.log_level = env.log_level
    # Ollama overrides
    if env.ollama_base_url:
        config.llm.base_url = env.ollama_base_url
    if env.ollama_model:
        config.llm.model = env.ollama_model
    # vLLM overrides (also sets provider if specified)
    if env.vllm_base_url:
        config.llm.vllm.base_url = env.vllm_base_url
        config.llm.provider = "vllm"
    if env.vllm_model:
        config.llm.vllm.model = env.vllm_model
        config.llm.provider = "vllm"
        config.llm.vllm.enabled = True
    # Model switching overrides
    if env.complex_model:
        config.llm.complex_model = env.complex_model
    if env.chat_model:
        config.llm.chat_model = env.chat_model
    if env.auto_switch_enabled is not None:
        config.llm.auto_switch_enabled = env.auto_switch_enabled
    return config


def load_policy_config(policy_path: Path | None = None) -> dict[str, Any]:
    """Load raw policy YAML."""
    env = EnvSettings()
    return _load_yaml(policy_path or env.policy)
