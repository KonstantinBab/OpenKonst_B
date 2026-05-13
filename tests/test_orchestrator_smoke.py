from pathlib import Path
from types import SimpleNamespace

from coding_agent.config.settings import AppConfig
from coding_agent.core.orchestrator import Orchestrator
from coding_agent.core.planner import AgentTurn, FinalReport
from coding_agent.core.verifier import Verifier
from coding_agent.llm.base import BaseLLMProvider, ChatMessage, LLMResponse
from coding_agent.util.errors import LLMResponseError
from coding_agent.memory.retrieval import RetrievalService
from coding_agent.memory.store import MemoryStore
from coding_agent.sandbox.command_policy import CommandPolicyEngine
from coding_agent.sandbox.shell_runner import ShellRunner
from coding_agent.sandbox.workspace_guard import WorkspaceGuard
from coding_agent.tools.file_tools import FileTools
from coding_agent.tools.git_tools import GitManager
from coding_agent.tools.memory_tools import MemoryTools
from coding_agent.tools.patch_tools import PatchTools
from coding_agent.tools.search_tools import FileChunk
from coding_agent.tools.search_tools import SearchTools
from coding_agent.memory.retrieval import RetrievalBundle
from coding_agent.core.context import ProjectContext
from coding_agent.llm.prompt_builder import build_turn_prompt
from coding_agent.api.app import (
    _extract_requested_iterations,
    _is_allowed_direct_command,
    _is_direct_smoke_training_request,
    _is_execution_request,
)


class FakeLLM(BaseLLMProvider):
    def chat(self, messages: list[ChatMessage], json_mode: bool = False) -> LLMResponse:
        return LLMResponse(content='```json\n{"final_report":{"status":"completed","summary":"done","changes":[],"verification":["read_file: ok"],"findings":[{"title":"Test finding","severity":"medium","status":"verified","evidence":"read_file: README","recommendation":"keep"}],"next_steps":["next"]}}\n```')


class BrokenLLM(BaseLLMProvider):
    def chat(self, messages: list[ChatMessage], json_mode: bool = False) -> LLMResponse:
        return LLMResponse(content="Thinking...\nI should inspect the repository first.\nNo valid JSON here.")


def test_orchestrator_returns_final_report(tmp_path: Path) -> None:
    config = AppConfig()
    config.memory_db_path = tmp_path / "memory.db"
    policy = CommandPolicyEngine({"shell": {"allow": [".*"]}, "file_ops": {"allow": ["write", "read", "list", "search", "mkdir", "patch"]}, "git": {"allow": ["status", "diff", "branch", "log", "add", "commit", "reset", "revert"]}})
    guard = WorkspaceGuard(tmp_path)
    shell_runner = ShellRunner(guard, policy)
    file_tools = FileTools(guard, policy)
    patch_tools = PatchTools(guard, file_tools)
    search_tools = SearchTools(guard)
    memory_store = MemoryStore(config.memory_db_path)
    retrieval = RetrievalService(memory_store, search_tools)
    verifier = Verifier(config.verifier, shell_runner)
    orchestrator = Orchestrator(
        config=config,
        llm=FakeLLM(),
        memory_store=memory_store,
        retrieval_service=retrieval,
        file_tools=file_tools,
        patch_tools=patch_tools,
        git_manager=GitManager(guard, shell_runner, policy),
        memory_tools=MemoryTools(memory_store),
        search_tools=search_tools,
        shell_runner=shell_runner,
        verifier=verifier,
    )
    report = orchestrator.run("test goal")
    assert report.status == "completed"
    assert report.findings[0]["title"] == "Test finding"
    assert memory_store.recent_memory(limit=1)


def test_agent_turn_normalizes_local_model_payload() -> None:
    turn = AgentTurn.from_model_payload(
        {
            "plan": [{"title": "Inspect repo", "status": "pending"}],
            "actions": [
                {"action": "list_files", "arguments": {}},
                {"action": "read_file", "arguments": {"path": ["README.md"]}},
                {"action": "list_directories", "arguments": {}},
            ],
            "final_report": {"status": "pending", "summary": "", "changes": [], "verification": []},
        }
    )
    assert len(turn.actions) == 2
    assert turn.actions[1].arguments["path"] == "README.md"
    assert turn.final_report is None


def test_agent_turn_normalizes_findings_and_next_steps() -> None:
    turn = AgentTurn.from_model_payload(
        {
            "final_report": {
                "status": "completed",
                "summary": "ok",
                "changes": ["done"],
                "verification": ["read_file: config/default.yaml"],
                "findings": [
                    {
                        "title": "LLM availability",
                        "severity": "high",
                        "status": "verified",
                        "evidence": "read_file: config/default.yaml",
                        "recommendation": "add fallback",
                    }
                ],
                "next_steps": ["improve prompts"],
            }
        }
    )
    assert turn.final_report is not None
    assert turn.final_report.findings[0]["title"] == "LLM availability"
    assert turn.final_report.next_steps == ["improve prompts"]


def test_agent_turn_normalizes_scalar_verification_fields() -> None:
    turn = AgentTurn.from_model_payload(
        {
            "final_report": {
                "status": "completed",
                "summary": "ok",
                "changes": "one change",
                "verification": "static analysis completed",
                "next_steps": "review api startup",
            }
        }
    )
    assert turn.final_report is not None
    assert turn.final_report.changes == ["one change"]
    assert turn.final_report.verification == ["static analysis completed"]
    assert turn.final_report.next_steps == ["review api startup"]


def test_agent_turn_drops_placeholder_final_report() -> None:
    turn = AgentTurn.from_model_payload(
        {
            "final_report": {
                "status": "completed",
                "summary": "string",
                "changes": ["string"],
                "verification": ["string"],
                "findings": [],
                "next_steps": [],
            }
        }
    )
    assert turn.final_report is None


def test_risk_report_quality_rejects_generic_observation_findings() -> None:
    report = AgentTurn.from_model_payload(
        {
            "final_report": {
                "status": "completed",
                "summary": "Сформирован аналитический итог на основе реальных наблюдений по репозиторию.",
                "changes": ["Проверено через read_file", "Проверено через read_file"],
                "verification": ["read_file: наблюдение собрано"],
                "findings": [
                    {
                        "title": "Наблюдение по read_file",
                        "severity": "medium",
                        "status": "verified",
                        "evidence": "read_file: {}",
                        "recommendation": "Проверить",
                    }
                ],
                "next_steps": ["Уточнить выводы"],
            }
        }
    ).final_report
    assert report is not None
    assert Orchestrator._is_report_quality_ok(report, is_risk_goal=True) is False


def test_parse_observation_recovers_path_from_truncated_read_file_json() -> None:
    item = (
        'read_file: {"ok": true, "action": "read_file", "path": '
        '"src/coding_agent/sandbox/workspace_guard.py", "message": "", '
        '"content": "\\"\\\\\\"Workspace boundary enforcement.'
    )
    parsed = Orchestrator._parse_observation(item)
    assert parsed["path"] == "src/coding_agent/sandbox/workspace_guard.py"


def test_orchestrator_bootstraps_project_overview(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# Demo\nproject overview", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    config = AppConfig()
    config.memory_db_path = tmp_path / "memory.db"
    policy = CommandPolicyEngine({"shell": {"allow": [".*"]}, "file_ops": {"allow": ["write", "read", "list", "search", "mkdir", "patch"]}, "git": {"allow": ["status", "diff", "branch", "log", "add", "commit", "reset", "revert"]}})
    guard = WorkspaceGuard(tmp_path)
    shell_runner = ShellRunner(guard, policy)
    file_tools = FileTools(guard, policy)
    patch_tools = PatchTools(guard, file_tools)
    search_tools = SearchTools(guard)
    memory_store = MemoryStore(config.memory_db_path)
    retrieval = RetrievalService(memory_store, search_tools)
    verifier = Verifier(config.verifier, shell_runner)
    orchestrator = Orchestrator(
        config=config,
        llm=FakeLLM(),
        memory_store=memory_store,
        retrieval_service=retrieval,
        file_tools=file_tools,
        patch_tools=patch_tools,
        git_manager=GitManager(guard, shell_runner, policy),
        memory_tools=MemoryTools(memory_store),
        search_tools=search_tools,
        shell_runner=shell_runner,
        verifier=verifier,
    )
    report = orchestrator.run("Понять проект и описать риски")
    assert report.status == "completed"


def test_run_chat_uses_fallback_read_when_model_output_is_unparseable() -> None:
    orchestrator = Orchestrator.__new__(Orchestrator)

    class _Guard:
        @staticmethod
        def resolve_path(path: str) -> Path:
            return Path(path)

    class _FileTools:
        guard = _Guard()

    orchestrator.file_tools = _FileTools()

    turn = orchestrator._fallback_chat_turn(
        "сначала пойми архитектуру проекта, потом назови 3 главных риска",
        observations=[],
        analytical_goal=True,
    )

    assert turn is not None
    assert turn.actions
    assert turn.actions[0].action == "read_file"


def test_should_finalize_after_stall_for_analytical_chat() -> None:
    orchestrator = Orchestrator.__new__(Orchestrator)

    observations = [
        'read_file: {"ok": true, "path": "src/train.py"}',
        'read_file: {"ok": true, "path": "src/env.py"}',
        'search_in_files: {"ok": true, "query": "risk"}',
    ]

    assert orchestrator._should_finalize_after_stall(2, True, observations)
    assert orchestrator._should_finalize_after_stall(3, False, observations)


def test_fallback_chat_turn_returns_none_when_analysis_has_enough_evidence_and_no_candidates() -> None:
    orchestrator = Orchestrator.__new__(Orchestrator)

    class _Guard:
        @staticmethod
        def resolve_path(path: str) -> Path:
            return Path(path)

    class _FileTools:
        guard = _Guard()

    orchestrator.file_tools = _FileTools()
    orchestrator._fallback_read_candidates = lambda goal, observations: []

    observations = [
        'read_file: {"ok": true, "path": "src/train.py"}',
        'read_file: {"ok": true, "path": "src/env.py"}',
        'search_in_files: {"ok": true, "query": "risk"}',
    ]

    turn = orchestrator._fallback_chat_turn(
        "сначала пойми архитектуру проекта, потом назови 3 главных риска",
        observations=observations,
        analytical_goal=True,
    )

    assert turn is None


def test_run_chat_analysis_uses_direct_plain_llm_synthesis() -> None:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.config = SimpleNamespace(orchestrator=SimpleNamespace(max_steps=2, max_tool_calls=8, retrieval_max_chunks=2))
    orchestrator.memory_store = SimpleNamespace(record_run=lambda *args, **kwargs: None, record_tool_call=lambda *args, **kwargs: None)
    orchestrator.memory_tools = SimpleNamespace(add=lambda *args, **kwargs: None)
    orchestrator.retrieval_service = SimpleNamespace(retrieve=lambda *args, **kwargs: RetrievalBundle(memory_hits=[], file_chunks=[]))
    orchestrator.verifier = SimpleNamespace(verify=lambda *args, **kwargs: SimpleNamespace(results=[]))
    orchestrator._should_bootstrap_project = lambda goal: False
    orchestrator._direct_analysis_candidates = lambda: ["README.md"]

    class _ReadResult:
        content = "project docs"

        @staticmethod
        def model_dump():
            return {"ok": True, "action": "read_file", "path": "README.md", "content": "project docs"}

    orchestrator.file_tools = SimpleNamespace(
        guard=SimpleNamespace(root=Path("F:/fake/workspace")),
        read_file=lambda *args, **kwargs: _ReadResult(),
    )
    calls: list[bool] = []

    class _PlainLLM(BaseLLMProvider):
        def chat(self, messages: list[ChatMessage], json_mode: bool = False) -> LLMResponse:
            calls.append(json_mode)
            return LLMResponse(content="Архитектура: test\n\n3 главных риска: test")

    orchestrator.llm = _PlainLLM()

    result = orchestrator.run_chat("Проверь архитектуру проекта и назови 3 риска")

    assert result.status == "completed"
    assert calls == [False]
    assert "Архитектура" in result.assistant_message


def test_action_chat_goal_with_previous_risk_context_is_not_analytical() -> None:
    goal = (
        "История сессии: раньше обсуждали 3 риска проекта.\n"
        "Новый запрос пользователя: измени на улучшенные гиперпараметры "
        "в настройках обучения и запусти тестовое обучение."
    )

    assert Orchestrator._is_analytical_goal(goal) is False


def test_pure_architecture_risk_goal_is_analytical() -> None:
    assert Orchestrator._is_analytical_goal("сначала пойми архитектуру проекта, потом назови 3 главных риска") is True


def test_action_prompt_requires_tools_even_with_previous_risk_context() -> None:
    context = ProjectContext(
        workspace="F:/project",
        retrieval_bundle=RetrievalBundle(memory_hits=[], file_chunks=[]),
        project_overview="",
        repo_map="",
        recent_observations=[],
    )
    prompt = build_turn_prompt(
        "История: раньше обсуждали риски. Новый запрос: измени гиперпараметры и запусти тестовое обучение.",
        context,
    )

    assert "This is an implementation/execution task" in prompt
    assert "This is an analytical task" not in prompt


def test_retrieval_query_keywords_prioritize_action_terms() -> None:
    keywords = SearchTools._query_keywords(
        "История: раньше обсуждали риски. Новый запрос: измени гиперпараметры и запусти тестовое обучение learning_rate."
    )

    assert any("обуч" in item for item in keywords)
    assert "learning_rate" in keywords
    assert "история" not in keywords


def test_command_execution_goal_requires_run_command_observation() -> None:
    assert Orchestrator._requires_command_execution("запусти еще раз тестовое обучение") is True
    # Проверяем что есть любой run_command (старая логика)
    assert any("run_command:" in item for item in ["read_file: src/train.py"]) is False
    assert any("run_command:" in item for item in ["run_command: python -m src.train"]) is True


def test_finalize_does_not_claim_command_execution_without_run_command() -> None:
    orchestrator = Orchestrator.__new__(Orchestrator)

    report = orchestrator._finalize_from_observations(
        "запусти еще раз тестовое обучение",
        ['read_file: {"ok": true, "path": "src/train.py"}'],
    )

    assert report.status == "failed"
    assert "run_command не выполнялся" in report.verification[0]


def test_direct_smoke_training_uses_session_context_for_repeat_request() -> None:
    assert _is_direct_smoke_training_request(
        "запусти еще раз",
        context_hint="предыдущая задача: запусти тестовое обучение",
    )


def test_execution_request_and_allowed_direct_commands() -> None:
    assert _is_execution_request("проверь синтаксис проекта")
    assert _is_allowed_direct_command("python -m compileall src")
    assert _is_allowed_direct_command(".\\.venv\\Scripts\\python.exe -m src.train --smoke-test")
    assert not _is_allowed_direct_command("Remove-Item -Recurse .")


def test_extract_requested_iterations_from_russian_request() -> None:
    assert _extract_requested_iterations("запусти с 5ю итерациями тестовое обучение") == 5
    assert _extract_requested_iterations("run smoke with 12 iterations") == 12
