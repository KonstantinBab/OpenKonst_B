from pathlib import Path

import pytest

from coding_agent.sandbox.command_policy import CommandPolicyEngine
from coding_agent.sandbox.workspace_guard import WorkspaceGuard
from coding_agent.tools.file_tools import FileTools
from coding_agent.util.errors import ToolExecutionError


def test_file_tools_write_and_read(tmp_path: Path) -> None:
    policy = CommandPolicyEngine({"shell": {}, "file_ops": {"allow": ["write", "read", "list", "search", "mkdir", "patch"]}, "git": {}})
    tools = FileTools(WorkspaceGuard(tmp_path), policy)
    tools.write_file("a.txt", "hello")
    result = tools.read_file("a.txt")
    assert result.content == "hello"


def test_file_tools_search(tmp_path: Path) -> None:
    policy = CommandPolicyEngine({"shell": {}, "file_ops": {"allow": ["write", "read", "list", "search", "mkdir", "patch"]}, "git": {}})
    tools = FileTools(WorkspaceGuard(tmp_path), policy)
    tools.write_file("a.txt", "needle")
    result = tools.search_in_files("needle")
    assert result.paths


def test_replace_in_file_requires_existing_target_text(tmp_path: Path) -> None:
    policy = CommandPolicyEngine({"shell": {}, "file_ops": {"allow": ["write", "read", "list", "search", "mkdir", "patch"]}, "git": {}})
    tools = FileTools(WorkspaceGuard(tmp_path), policy)
    tools.write_file("a.txt", "hello")
    with pytest.raises(ToolExecutionError):
        tools.replace_in_file("a.txt", "missing", "updated")
