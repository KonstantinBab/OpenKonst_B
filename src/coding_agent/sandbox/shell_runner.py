"""Single guarded shell execution backend."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

from pydantic import BaseModel, Field

from coding_agent.sandbox.command_policy import CommandPolicyEngine, PolicyDecision
from coding_agent.sandbox.workspace_guard import WorkspaceGuard
from coding_agent.util.errors import ToolExecutionError


class ShellCommand(BaseModel):
    command: str
    cwd: str | None = None
    timeout_seconds: int = 60
    env: dict[str, str] = Field(default_factory=dict)
    dry_run: bool = False
    approval: bool = False


class ShellResult(BaseModel):
    command: str
    exit_code: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    cwd: str
    decision: PolicyDecision
    backend: str
    timed_out: bool = False
    dry_run: bool = False


class ShellRunner:
    """PowerShell-first shell runner with policy checks and workspace confinement."""

    def __init__(self, guard: WorkspaceGuard, policy: CommandPolicyEngine):
        self.guard = guard
        self.policy = policy
        self.is_windows = platform.system().lower().startswith("win")

    def run(self, request: ShellCommand) -> ShellResult:
        decision = self.policy.evaluate_shell(request.command)
        self.policy.enforce(decision, approval=request.approval)
        cwd = self.guard.ensure_cwd(request.cwd)
        backend = self._backend_name()
        if request.dry_run:
            return ShellResult(
                command=request.command,
                exit_code=None,
                stdout="",
                stderr="",
                duration_seconds=0.0,
                cwd=str(cwd),
                decision=decision,
                backend=backend,
                dry_run=True,
            )

        shell_args = self._shell_args(request.command)
        env = os.environ.copy()
        env.update(request.env)
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                shell_args,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=request.timeout_seconds,
                env=env,
            )
            timed_out = False
            exit_code = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            exit_code = -1
            stdout = exc.stdout or ""
            stderr = (exc.stderr or "") + f"\nCommand timed out after {request.timeout_seconds} seconds."
        except OSError as exc:
            raise ToolExecutionError(f"Failed to execute shell backend '{backend}': {exc}") from exc
        duration = time.perf_counter() - started
        redacted_stdout = self._redact(stdout, request.env)
        redacted_stderr = self._redact(stderr, request.env)
        return ShellResult(
            command=request.command,
            exit_code=exit_code,
            stdout=redacted_stdout,
            stderr=redacted_stderr,
            duration_seconds=duration,
            cwd=str(cwd),
            decision=decision,
            backend=backend,
            timed_out=timed_out,
        )

    def _shell_args(self, command: str) -> list[str]:
        if self.is_windows:
            executable = shutil.which("powershell") or shutil.which("pwsh") or "powershell"
            return [executable, "-NoLogo", "-NoProfile", "-Command", command]
        return ["sh", "-lc", command]

    def _backend_name(self) -> str:
        if self.is_windows:
            return shutil.which("powershell") or shutil.which("pwsh") or "powershell"
        return "sh"

    @staticmethod
    def _redact(content: str, env_overrides: dict[str, str]) -> str:
        redacted = content
        for value in env_overrides.values():
            if value and len(value) >= 4:
                redacted = redacted.replace(value, "***REDACTED***")
        return redacted
