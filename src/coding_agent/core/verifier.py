"""Verification runner."""

from __future__ import annotations

from pydantic import BaseModel, Field

from coding_agent.config.settings import VerifierSettings
from coding_agent.sandbox.shell_runner import ShellCommand, ShellRunner


class VerificationCommandResult(BaseModel):
    command: str
    exit_code: int | None
    stdout: str
    stderr: str


class VerificationSummary(BaseModel):
    profile: str
    passed: bool
    results: list[VerificationCommandResult] = Field(default_factory=list)


class Verifier:
    """Profile-based verify pipeline."""

    def __init__(self, settings: VerifierSettings, shell_runner: ShellRunner):
        self.settings = settings
        self.shell_runner = shell_runner

    def verify(self, profile: str | None = None) -> VerificationSummary:
        selected = profile or self.settings.default_profile
        if not hasattr(self.settings, selected):
            available = ", ".join(("python", "node", "generic"))
            raise ValueError(f"Unknown verification profile '{selected}'. Available profiles: {available}")
        profile_config = getattr(self.settings, selected)
        commands = profile_config.tests + profile_config.lint + profile_config.type_check + profile_config.build
        results: list[VerificationCommandResult] = []
        passed = True
        for command in commands:
            result = self.shell_runner.run(ShellCommand(command=command, cwd="."))
            results.append(
                VerificationCommandResult(
                    command=command,
                    exit_code=result.exit_code,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
            )
            if result.exit_code not in (0, None):
                passed = False
        return VerificationSummary(profile=selected, passed=passed, results=results)
