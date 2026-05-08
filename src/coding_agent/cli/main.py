"""Typer CLI entrypoint."""

from __future__ import annotations

from pathlib import Path
import subprocess

import typer
import httpx

from coding_agent.config.loader import load_app_config, load_policy_config
from coding_agent.core.bootstrap import Runtime
from coding_agent.core.bootstrap import build_runtime
from coding_agent.util.errors import CodingAgentError
from coding_agent.util.json_utils import to_pretty_json
from coding_agent.util.logging import configure_logging

app = typer.Typer(help="Windows-first self-hosted coding agent")
memory_app = typer.Typer(help="Memory operations")
git_app = typer.Typer(help="Git operations")
config_app = typer.Typer(help="Configuration utilities")
doctor_app = typer.Typer(help="Runtime diagnostics")
app.add_typer(memory_app, name="memory")
app.add_typer(git_app, name="git")
app.add_typer(config_app, name="config")
app.add_typer(doctor_app, name="doctor")


def _runtime(workspace: str, model: str | None = None) -> Runtime:
    workspace_path = Path(workspace).expanduser().resolve()
    runtime = build_runtime(
        workspace_path,
        config_path=workspace_path / "config" / "default.yaml",
        policy_path=workspace_path / "config" / "policy.yaml",
        model_override=model,
    )
    configure_logging(runtime.config.log_level)
    return runtime


@app.command("run")
def run_agent(
    workspace: str = typer.Option(..., "--workspace", help="Absolute or relative workspace path"),
    goal: str = typer.Option(..., "--goal", help="Agent goal"),
    model: str | None = typer.Option(None, "--model", help="Override Ollama/OpenAI-compatible model for this run"),
) -> None:
    try:
        runtime = _runtime(workspace, model=model)
        runtime.llm.warmup()
        result = runtime.orchestrator.run(goal)
        typer.echo(to_pretty_json(result.model_dump(mode="json")))
    except CodingAgentError as exc:
        typer.secho(f"Agent error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command("inspect")
def inspect_workspace(workspace: str = typer.Option(..., "--workspace", help="Absolute or relative workspace path")) -> None:
    runtime = _runtime(workspace)
    info = {
        "workspace": str(runtime.guard.root),
        "is_git_repo": runtime.git_manager.detect_repo(),
        "recent_memories": [item.model_dump(mode="json") for item in runtime.memory_store.recent_memory(limit=5)],
    }
    typer.echo(to_pretty_json(info))


@app.command("verify")
def verify_workspace(
    workspace: str = typer.Option(..., "--workspace", help="Absolute or relative workspace path"),
    profile: str = typer.Option("generic", "--profile", help="Verification profile"),
) -> None:
    runtime = _runtime(workspace)
    result = runtime.verifier.verify(profile)
    typer.echo(to_pretty_json(result.model_dump(mode="json")))


@memory_app.command("search")
def memory_search(
    workspace: str = typer.Option(..., "--workspace", help="Absolute or relative workspace path"),
    query: str = typer.Option(..., "--query", help="Memory query"),
    limit: int = typer.Option(10, "--limit", help="Maximum number of results"),
) -> None:
    runtime = _runtime(workspace)
    typer.echo(to_pretty_json([item.model_dump(mode="json") for item in runtime.memory_tools.search(query, limit)]))


@git_app.command("status")
def git_status(workspace: str = typer.Option(..., "--workspace", help="Absolute or relative workspace path")) -> None:
    runtime = _runtime(workspace)
    typer.echo(to_pretty_json(runtime.git_manager.status().model_dump(mode="json")))


@config_app.command("validate")
def config_validate(config_path: str = "./config/default.yaml", policy_path: str = "./config/policy.yaml") -> None:
    config = load_app_config(Path(config_path))
    policy = load_policy_config(Path(policy_path))
    typer.echo(to_pretty_json({"config": config.model_dump(mode="json"), "policy_loaded": bool(policy)}))


@doctor_app.command("ollama")
def doctor_ollama(
    workspace: str = typer.Option(..., "--workspace", help="Absolute or relative workspace path"),
    model: str | None = typer.Option(None, "--model", help="Override model for diagnostics"),
) -> None:
    runtime = _runtime(workspace, model=model)
    payload: dict[str, object] = {
        "base_url": runtime.config.llm.base_url,
        "model": runtime.config.llm.model,
        "tags_ok": False,
        "warmup_ok": False,
        "ollama_version": _run_diagnostic_command(["ollama", "--version"]),
        "ollama_list": _run_diagnostic_command(["ollama", "list"]),
        "ollama_ps": _run_diagnostic_command(["ollama", "ps"]),
        "ollama_processes": _list_ollama_processes(),
    }
    try:
        tags_response = httpx.get(f"{runtime.config.llm.base_url}/api/tags", timeout=15)
        payload["tags_ok"] = tags_response.status_code == 200
        payload["tags_status_code"] = tags_response.status_code
        payload["tags_body_preview"] = tags_response.text[:1000]
    except httpx.HTTPError as exc:
        payload["tags_error"] = str(exc)
    try:
        runtime.llm.warmup()
        payload["warmup_ok"] = True
    except CodingAgentError as exc:
        payload["warmup_error"] = str(exc)
    payload["diagnosis"] = _diagnose_ollama(payload)
    typer.echo(to_pretty_json(payload))


def _diagnose_ollama(payload: dict[str, object]) -> list[str]:
    diagnosis: list[str] = []
    processes = payload.get("ollama_processes")
    if isinstance(processes, list) and len(processes) > 2:
        diagnosis.append(
            "Multiple Ollama processes are running. If /api/tags returns 503, restart Ollama from the tray app "
            "or stop duplicate ollama.exe processes, then start Ollama again."
        )
    if payload.get("tags_status_code") == 503:
        diagnosis.append(
            "Ollama CLI is reachable, but the HTTP API returns 503. This points to a stuck or overloaded Ollama "
            "server rather than a missing model."
        )
    warmup_error = payload.get("warmup_error")
    if isinstance(warmup_error, str) and "/api/tags HTTP 503" in warmup_error:
        diagnosis.append(
            "Recommended recovery: close Ollama from the Windows tray, run `taskkill /IM ollama.exe /F`, reopen "
            "Ollama, verify `ollama list`, then rerun `agent doctor ollama --workspace F:\\chrommm\\OpenKonst_B`."
        )
    if not diagnosis:
        diagnosis.append("No obvious Ollama process or HTTP readiness issue detected.")
    return diagnosis


def _run_diagnostic_command(command: list[str]) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    except Exception as exc:  # noqa: BLE001
        return str(exc)
    output = (result.stdout or result.stderr).strip()
    return output or f"exit_code={result.returncode}"


def _list_ollama_processes() -> list[dict[str, object]]:
    powershell = [
        "powershell",
        "-NoLogo",
        "-NoProfile",
        "-Command",
        "Get-Process | Where-Object { $_.ProcessName -like 'ollama*' } | "
        "Select-Object ProcessName,Id,Path | ConvertTo-Json -Compress",
    ]
    try:
        result = subprocess.run(powershell, capture_output=True, text=True, timeout=10)
    except Exception:  # noqa: BLE001
        return []
    if result.returncode != 0 or not result.stdout.strip():
        return []
    try:
        import json

        parsed = json.loads(result.stdout)
    except Exception:  # noqa: BLE001
        return [{"raw": result.stdout.strip()}]
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return parsed
    return [{"raw": parsed}]


if __name__ == "__main__":
    app()
