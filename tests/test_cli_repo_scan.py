from pathlib import Path

from typer.testing import CliRunner

import coding_agent.cli.main as cli_main
from coding_agent.core.repo_scan import RepoScanResult


runner = CliRunner()


class _FakeRepoScanner:
    def load_artifact_with_path(self, target: Path | None = None):
        return (
            Path(__file__),
            RepoScanResult(
                confirmed_frameworks=["FastAPI", "Typer"],
                possible_frameworks=[],
                api_surfaces=["GET /health (src/coding_agent/api/app.py)"],
                security_boundaries=["WorkspaceGuard"],
            ),
        )

    def save_artifact(self, target: Path | None = None) -> Path:
        return Path(__file__)


class _FakeOrchestrator:
    def __init__(self) -> None:
        self.repo_scanner = _FakeRepoScanner()


class _FakeRuntime:
    def __init__(self) -> None:
        self.orchestrator = _FakeOrchestrator()
        self.llm = type("FakeLLM", (), {"warmup": staticmethod(lambda: None)})()
        self.guard = type("FakeGuard", (), {"root": Path("F:/fake/workspace")})()


def test_doctor_repo_scan_reports_artifact_summary(monkeypatch) -> None:
    monkeypatch.setattr(cli_main, "_runtime", lambda workspace, model=None: _FakeRuntime())

    result = runner.invoke(cli_main.app, ["doctor", "repo-scan", "--workspace", "F:/fake/workspace"])

    assert result.exit_code == 0
    assert '"ok": true' in result.stdout.lower()
    assert "FastAPI" in result.stdout
    assert "WorkspaceGuard" in result.stdout


def test_resolve_report_path_uses_workspace_root_for_relative_paths() -> None:
    target = cli_main._resolve_report_path("artifacts/final_report.json", base_dir=Path("F:/fake/workspace"))
    assert target == Path("F:/fake/workspace/artifacts/final_report.json").resolve()


def test_chat_command_defaults_workspace(monkeypatch) -> None:
    monkeypatch.setattr(cli_main, "_runtime", lambda workspace, model=None: _FakeRuntime())
    monkeypatch.setattr(cli_main, "_load_chat_state", lambda path: cli_main.ChatSessionState())
    monkeypatch.setattr(cli_main, "_resolve_report_path", lambda path, base_dir=None: Path("F:/fake/last_chat_report.md"))
    monkeypatch.setattr(cli_main, "_session_report_path", lambda path, base_dir=None: Path("F:/fake/last_chat_report_session.md"))
    monkeypatch.setattr(cli_main, "_write_chat_report_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_main, "_append_session_report", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_main, "_write_chat_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_main.typer, "prompt", lambda *_args, **_kwargs: "exit")

    result = runner.invoke(cli_main.app, ["chat"])

    assert result.exit_code == 0
    assert "Режим чата запущен" in result.stdout
