"""Safe git operations routed through the shell runner."""

from __future__ import annotations

from pydantic import BaseModel

from coding_agent.sandbox.command_policy import CommandPolicyEngine
from coding_agent.sandbox.shell_runner import ShellCommand, ShellRunner
from coding_agent.sandbox.workspace_guard import WorkspaceGuard


class GitResult(BaseModel):
    ok: bool
    action: str
    stdout: str
    stderr: str
    exit_code: int | None


class GitManager:
    """Git operations with policy integration and Windows-safe command routing."""

    def __init__(self, guard: WorkspaceGuard, shell_runner: ShellRunner, policy: CommandPolicyEngine):
        self.guard = guard
        self.shell_runner = shell_runner
        self.policy = policy

    def detect_repo(self) -> bool:
        result = self.shell_runner.run(ShellCommand(command="git rev-parse --is-inside-work-tree", cwd="."))
        return result.exit_code == 0 and result.stdout.strip().lower() == "true"

    def status(self) -> GitResult:
        return self._run("status", "git status --short --branch")

    def current_branch(self) -> GitResult:
        return self._run("branch", "git branch --show-current")

    def create_branch(self, name: str, approval: bool = False) -> GitResult:
        return self._run("branch", f"git checkout -b {self._quote(name)}", approval=approval)

    def diff(self, ref: str | None = None) -> GitResult:
        command = "git diff" if not ref else f"git diff {ref}"
        return self._run("diff", command)

    def add_paths(self, paths: list[str], approval: bool = False) -> GitResult:
        joined = " ".join(self._quote(path) for path in paths)
        return self._run("add", f"git add {joined}", approval=approval)

    def commit(self, message: str, approval: bool = False) -> GitResult:
        return self._run("commit", f"git commit -m {self._quote(message)}", approval=approval)

    def reset_working_tree(self, mode: str = "--hard", approval: bool = False) -> GitResult:
        return self._run("reset", f"git reset {mode}", approval=approval)

    def revert(self, revision: str, approval: bool = False) -> GitResult:
        return self._run("revert", f"git revert {revision} --no-edit", approval=approval)

    def short_log_summary(self, limit: int = 5) -> GitResult:
        return self._run("log", f'git log --oneline -n {limit}')

    def _run(self, action: str, command: str, approval: bool = False) -> GitResult:
        decision = self.policy.evaluate_git(action)
        self.policy.enforce(decision, approval=approval)
        if action not in {"status", "branch"} and not self.detect_repo():
            return GitResult(ok=False, action=action, stdout="", stderr="Workspace is not a git repository.", exit_code=1)
        result = self.shell_runner.run(ShellCommand(command=command, cwd=".", approval=approval))
        return GitResult(
            ok=(result.exit_code == 0 or result.exit_code is None),
            action=action,
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
        )

    @staticmethod
    def _quote(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"
