from pathlib import Path

from coding_agent.sandbox.command_policy import CommandPolicyEngine
from coding_agent.sandbox.shell_runner import ShellCommand, ShellRunner
from coding_agent.sandbox.workspace_guard import WorkspaceGuard


def test_shell_runner_dry_run(tmp_path: Path) -> None:
    policy = CommandPolicyEngine({"shell": {"allow": [".*"]}, "file_ops": {}, "git": {}})
    runner = ShellRunner(WorkspaceGuard(tmp_path), policy)
    result = runner.run(ShellCommand(command="Write-Output 'hello'", dry_run=True))
    assert result.dry_run is True
    assert result.exit_code is None


def test_shell_runner_executes_command(tmp_path: Path) -> None:
    policy = CommandPolicyEngine({"shell": {"allow": [".*"]}, "file_ops": {}, "git": {}})
    runner = ShellRunner(WorkspaceGuard(tmp_path), policy)
    result = runner.run(ShellCommand(command="Write-Output 'hello'"))
    assert result.exit_code == 0
    assert "hello" in result.stdout.lower()


def test_shell_runner_timeout_returns_structured_result(tmp_path: Path) -> None:
    policy = CommandPolicyEngine({"shell": {"allow": [".*"]}, "file_ops": {}, "git": {}})
    runner = ShellRunner(WorkspaceGuard(tmp_path), policy)
    result = runner.run(ShellCommand(command="Start-Sleep -Seconds 2", timeout_seconds=1))
    assert result.exit_code == -1
    assert result.timed_out is True
