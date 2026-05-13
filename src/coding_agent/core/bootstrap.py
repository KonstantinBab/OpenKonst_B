"""Runtime assembly helpers."""

from __future__ import annotations

from dataclasses import dataclass
import sqlite3
import tempfile
from pathlib import Path

from coding_agent.config.loader import load_app_config, load_policy_config
from coding_agent.config.settings import AppConfig
from coding_agent.core.orchestrator import Orchestrator
from coding_agent.core.verifier import Verifier
from coding_agent.llm.base import BaseLLMProvider
from coding_agent.llm.ollama_provider import OllamaProvider
from coding_agent.llm.openai_compatible_provider import OpenAICompatibleProvider
from coding_agent.llm.vllm_provider import VLLMProvider
from coding_agent.memory.retrieval import RetrievalService
from coding_agent.memory.store import MemoryStore
from coding_agent.sandbox.command_policy import CommandPolicyEngine
from coding_agent.sandbox.shell_runner import ShellRunner
from coding_agent.sandbox.workspace_guard import WorkspaceGuard
from coding_agent.tools.file_tools import FileTools
from coding_agent.tools.git_tools import GitManager
from coding_agent.tools.memory_tools import MemoryTools
from coding_agent.tools.patch_tools import PatchTools
from coding_agent.tools.search_tools import SearchTools


@dataclass
class Runtime:
    config: AppConfig
    guard: WorkspaceGuard
    policy: CommandPolicyEngine
    shell_runner: ShellRunner
    file_tools: FileTools
    patch_tools: PatchTools
    search_tools: SearchTools
    git_manager: GitManager
    memory_store: MemoryStore
    memory_tools: MemoryTools
    retrieval_service: RetrievalService
    verifier: Verifier
    llm: BaseLLMProvider
    orchestrator: Orchestrator


def build_runtime(
    workspace: Path,
    config_path: Path | None = None,
    policy_path: Path | None = None,
    model_override: str | None = None,
) -> Runtime:
    """Build the complete runtime for a workspace."""
    config = load_app_config(config_path)
    workspace = workspace.expanduser().resolve()
    if not config.memory_db_path.is_absolute():
        config.memory_db_path = (workspace / config.memory_db_path).resolve()
    if model_override:
        config.llm.model = model_override
    policy = CommandPolicyEngine(load_policy_config(policy_path))
    guard = WorkspaceGuard(
        workspace,
        deny_patterns=[".git/**/.env", "**/.env"],
        ignore_patterns=[
            ".git/**",
            ".venv/**",
            "**/__pycache__/**",
            "**/*.pyc",
            "**/*.pyo",
            "**/*.egg-info/**",
            ".pytest_cache/**",
            ".pytest-tmp/**",
            "data/**",
            "node_modules/**",
        ],
    )
    shell_runner = ShellRunner(guard, policy)
    file_tools = FileTools(guard, policy)
    patch_tools = PatchTools(guard, file_tools)
    search_tools = SearchTools(guard)
    git_manager = GitManager(guard, shell_runner, policy)
    memory_store = _build_memory_store(config, workspace)
    memory_tools = MemoryTools(memory_store)
    retrieval_service = RetrievalService(memory_store, search_tools)
    verifier = Verifier(config.verifier, shell_runner)
    llm = build_llm_provider(config)
    orchestrator = Orchestrator(
        config=config,
        llm=llm,
        memory_store=memory_store,
        retrieval_service=retrieval_service,
        file_tools=file_tools,
        patch_tools=patch_tools,
        git_manager=git_manager,
        memory_tools=memory_tools,
        search_tools=search_tools,
        shell_runner=shell_runner,
        verifier=verifier,
    )
    return Runtime(
        config=config,
        guard=guard,
        policy=policy,
        shell_runner=shell_runner,
        file_tools=file_tools,
        patch_tools=patch_tools,
        search_tools=search_tools,
        git_manager=git_manager,
        memory_store=memory_store,
        memory_tools=memory_tools,
        retrieval_service=retrieval_service,
        verifier=verifier,
        llm=llm,
        orchestrator=orchestrator,
    )


def _build_memory_store(config: AppConfig, workspace: Path) -> MemoryStore:
    try:
        return MemoryStore(config.memory_db_path)
    except sqlite3.OperationalError as exc:
        if "readonly" not in str(exc).lower():
            raise
        fallback_path = _fallback_memory_db_path(workspace)
        config.memory_db_path = fallback_path
        return MemoryStore(fallback_path)


def _fallback_memory_db_path(workspace: Path) -> Path:
    return Path(tempfile.gettempdir()) / "coding-agent" / "memory" / workspace.name / "agent_memory.db"


def build_llm_provider(config: AppConfig) -> BaseLLMProvider:
    """Create an LLM provider from config with multi-model support."""
    if config.llm.provider == "ollama":
        return OllamaProvider(
            config.llm.base_url,
            config.llm.model,
            config.llm.timeout_seconds,
            max_retries=config.llm.max_retries,
            retry_backoff_seconds=config.llm.retry_backoff_seconds,
            warmup_timeout_seconds=config.llm.warmup_timeout_seconds,
            simple_model=config.llm.simple_model,
            complex_model=config.llm.complex_model,
            chat_model=config.llm.chat_model,
            auto_switch_enabled=config.llm.auto_switch_enabled,
        )
    if config.llm.provider == "vllm":
        return VLLMProvider(
            config.llm.base_url,
            config.llm.model,
            config.llm.timeout_seconds,
            max_retries=config.llm.max_retries,
            retry_backoff_seconds=config.llm.retry_backoff_seconds,
            warmup_timeout_seconds=config.llm.warmup_timeout_seconds,
            api_key=config.llm.api_key,
            simple_model=config.llm.simple_model,
            complex_model=config.llm.complex_model,
            chat_model=config.llm.chat_model,
            auto_switch_enabled=config.llm.auto_switch_enabled,
        )
    return OpenAICompatibleProvider(
        config.llm.base_url,
        config.llm.model,
        config.llm.timeout_seconds,
        api_key=config.llm.api_key,
        simple_model=config.llm.simple_model,
        complex_model=config.llm.complex_model,
        chat_model=config.llm.chat_model,
        auto_switch_enabled=config.llm.auto_switch_enabled,
    )
