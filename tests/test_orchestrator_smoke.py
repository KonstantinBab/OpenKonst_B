from pathlib import Path

from coding_agent.config.settings import AppConfig
from coding_agent.core.orchestrator import Orchestrator
from coding_agent.core.planner import AgentTurn
from coding_agent.core.verifier import Verifier
from coding_agent.llm.base import BaseLLMProvider, ChatMessage, LLMResponse
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


class FakeLLM(BaseLLMProvider):
    def chat(self, messages: list[ChatMessage], json_mode: bool = False) -> LLMResponse:
        return LLMResponse(content='```json\n{"final_report":{"status":"completed","summary":"done","changes":[],"verification":[]}}\n```')


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
