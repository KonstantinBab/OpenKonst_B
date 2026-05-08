from pathlib import Path

from coding_agent.sandbox.shell_runner import ShellCommand
from coding_agent.sandbox.command_policy import CommandPolicyEngine
from coding_agent.sandbox.shell_runner import ShellRunner
from coding_agent.sandbox.workspace_guard import WorkspaceGuard
from coding_agent.tools.git_tools import GitManager


def test_git_tools_status_smoke(tmp_path: Path) -> None:
    policy = CommandPolicyEngine({"shell": {"allow": [".*"]}, "file_ops": {}, "git": {"allow": ["status", "diff", "branch", "log", "add", "commit", "reset", "revert"]}})
    guard = WorkspaceGuard(tmp_path)
    runner = ShellRunner(guard, policy)
    init_result = runner.run(ShellCommand(command="git init"))
    assert init_result.exit_code == 0
    manager = GitManager(guard, runner, policy)
    status = manager.status()
    assert status.exit_code == 0
    assert manager.detect_repo() is True
