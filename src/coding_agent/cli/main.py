"""Typer CLI entrypoint."""

from __future__ import annotations

from pathlib import Path
import subprocess
import re
import threading
import time
import webbrowser

import typer
import httpx

from coding_agent.config.loader import load_app_config, load_policy_config
from coding_agent.core.bootstrap import Runtime
from coding_agent.core.bootstrap import build_runtime
from coding_agent.core.session import ChatHistoryEntry, ChatSessionState
from coding_agent.llm.base import ChatMessage
from coding_agent.util.errors import CodingAgentError
from coding_agent.util.json_utils import to_pretty_json
from coding_agent.util.logging import configure_logging

app = typer.Typer(help="Windows-first self-hosted coding agent")
memory_app = typer.Typer(help="Memory operations")
git_app = typer.Typer(help="Git operations")
config_app = typer.Typer(help="Configuration utilities")
doctor_app = typer.Typer(help="Runtime diagnostics")
repo_app = typer.Typer(help="Repository scan utilities")
app.add_typer(memory_app, name="memory")
app.add_typer(git_app, name="git")
app.add_typer(config_app, name="config")
app.add_typer(doctor_app, name="doctor")
app.add_typer(repo_app, name="repo")

PREFERRED_CHAT_MODELS = (
    "qwen2.5-coder:14b",
    "qwen3.5:9b",
    "deepseek-coder:6.7b",
)


def _agent_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _runtime_paths(workspace: str) -> tuple[Path, Path, Path]:
    workspace_path = Path(workspace).expanduser().resolve()
    agent_root = _agent_root()
    config_path = workspace_path / "config" / "default.yaml"
    policy_path = workspace_path / "config" / "policy.yaml"
    if not config_path.exists():
        config_path = agent_root / "config" / "default.yaml"
    if not policy_path.exists():
        policy_path = agent_root / "config" / "policy.yaml"
    return workspace_path, config_path, policy_path


def _runtime(workspace: str, model: str | None = None) -> Runtime:
    workspace_path, config_path, policy_path = _runtime_paths(workspace)
    runtime = build_runtime(
        workspace_path,
        config_path=config_path,
        policy_path=policy_path,
        model_override=model,
    )
    configure_logging(runtime.config.log_level)
    return runtime


@app.command("run")
def run_agent(
    workspace: str = typer.Option(..., "--workspace", help="Absolute or relative workspace path"),
    goal: str = typer.Option(..., "--goal", help="Agent goal"),
    model: str | None = typer.Option(None, "--model", help="Override Ollama/OpenAI-compatible model for this run"),
    output_report: str | None = typer.Option(None, "--output-report", help="Path to save final_report JSON"),
) -> None:
    try:
        runtime = _runtime(workspace, model=model)
        runtime.llm.warmup()
        result = runtime.orchestrator.run(goal)
        payload = result.model_dump(mode="json")
        typer.echo(to_pretty_json(payload))
        if output_report:
            _write_report_file(output_report, payload, base_dir=runtime.guard.root)
    except CodingAgentError as exc:
        typer.secho(f"Agent error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command("assist")
def assist_agent(
    workspace: str = typer.Option(..., "--workspace", help="Absolute or relative workspace path"),
    goal: str = typer.Option("Улучшить этот репозиторий", "--goal", help="High-level objective"),
    strict_json: bool = typer.Option(True, "--strict-json/--no-strict-json", help="Print only final_report JSON"),
    output_report: str | None = typer.Option(None, "--output-report", help="Path to save last final_report JSON"),
) -> None:
    """Interactive assistant loop with proposal/approval flow."""
    runtime = _runtime(workspace)
    runtime.llm.warmup()
    current_goal = goal
    execute_directly = False
    while True:
        if execute_directly:
            selected_goal = current_goal
        else:
            proposals = _propose_next_actions(runtime, current_goal)
            _print_proposals(proposals)
            choice = typer.prompt(
                "Выберите действие: номер, 'all', 'custom' или 'exit'",
                default="1",
            ).strip()
            if choice.lower() == "exit":
                typer.echo("Остановлено пользователем.")
                return
            if choice.lower() == "custom":
                selected_goal = typer.prompt("Введите вашу цель").strip()
                execute_directly = True
            elif choice.lower() == "all":
                selected_goal = " ; ".join(item["action_goal"] for item in proposals if item.get("action_goal"))
            else:
                selected_goal = _goal_from_choice(choice, proposals)
                if not selected_goal:
                    typer.echo("Некорректный выбор. Попробуйте снова.")
                    continue

        if _requires_approval(selected_goal):
            typer.echo(f"Планируемое изменение: {selected_goal}")
            approved = typer.confirm("Применить это изменение сейчас?", default=True)
            if not approved:
                continue

        result = runtime.orchestrator.run(selected_goal)
        payload = result.model_dump(mode="json")
        typer.echo(to_pretty_json(payload))
        if output_report:
            _write_report_file(output_report, payload, base_dir=runtime.guard.root)

        next_step = typer.prompt(
            "Дальше: 'continue' (новые предложения), 'custom' (ваша цель) или 'exit'",
            default="continue",
        ).strip().lower()
        if next_step == "exit":
            return
        if next_step == "custom":
            current_goal = typer.prompt("Введите следующую цель").strip()
            execute_directly = True
        else:
            current_goal = f"Продолжить улучшение репозитория после: {result.summary}"
            execute_directly = False


@app.command("chat")
def chat_agent(
    workspace: str = typer.Option(".", "--workspace", help="Target project workspace; defaults to current directory"),
    model: str | None = typer.Option(None, "--model", help="Override chat model; defaults to a fast installed coding model when available"),
    output_report: str = typer.Option("artifacts/last_chat_report.md", "--output-report", help="Path to save last chat transcript"),
    russian_only: bool = typer.Option(True, "--russian-only/--no-russian-only", help="Force Russian in summaries"),
) -> None:
    """Interactive conversational loop with persistent session state."""
    selected_model = model or _select_default_chat_model(workspace)
    runtime = _runtime(workspace, model=selected_model)
    runtime.llm.warmup()
    active_model = getattr(getattr(getattr(runtime, "config", None), "llm", None), "model", selected_model or "unknown")
    typer.echo(f"Chat model: {active_model}")
    report_path = _resolve_report_path(output_report, base_dir=runtime.guard.root)
    session_path = _session_report_path(output_report, base_dir=runtime.guard.root)
    state_path = _chat_state_path(report_path)
    chat_state = _load_chat_state(state_path)
    typer.echo("Режим чата запущен. Пишите обычными фразами, агент сам исследует проект и меняет код. Для выхода введите 'exit'.")
    while True:
        user_goal = typer.prompt("Вы").strip()
        if not user_goal:
            continue
        if user_goal.lower() in {"exit", "quit", "q"}:
            typer.echo("Выход из чата.")
            return
        switched_runtime = _maybe_switch_chat_workspace(user_goal, runtime)
        if switched_runtime is not None:
            runtime = switched_runtime
            report_path = _resolve_report_path(output_report, base_dir=runtime.guard.root)
            session_path = _session_report_path(output_report, base_dir=runtime.guard.root)
            state_path = _chat_state_path(report_path)
            chat_state = _load_chat_state(state_path)
            payload = {
                "status": "completed",
                "summary": f"Workspace переключён на {runtime.guard.root}. Теперь напишите, что именно нужно изучить, проверить или исправить в этом проекте.",
                "changes": [],
                "verification": [],
                "findings": [],
                "next_steps": ["Например: 'сначала пойми архитектуру проекта, потом назови 3 главных риска'."],
            }
            payload = _normalize_chat_payload(payload, russian_only=russian_only)
            chat_state = _update_chat_state(chat_state, user_goal, payload)
            typer.echo(_format_chat_response(payload, chat_state))
            _write_chat_report_file(output_report, payload, user_goal, base_dir=runtime.guard.root)
            _append_session_report(session_path, payload, user_goal)
            _write_chat_state(state_path, chat_state)
            continue
        effective_goal, immediate_override = _resolve_chat_request(user_goal, chat_state)
        if immediate_override is not None:
            payload = _normalize_chat_payload(immediate_override, russian_only=russian_only)
            chat_state = _update_chat_state(chat_state, user_goal, payload, effective_goal=effective_goal)
            typer.echo(_format_chat_response(payload, chat_state))
            _write_chat_report_file(output_report, payload, user_goal, base_dir=runtime.guard.root)
            _append_session_report(session_path, payload, user_goal)
            _write_chat_state(state_path, chat_state)
            continue
        immediate_payload = _maybe_handle_chat_message(user_goal, runtime.guard.root)
        if immediate_payload is not None:
            payload = _normalize_chat_payload(immediate_payload, russian_only=russian_only)
            chat_state = _update_chat_state(chat_state, user_goal, payload, effective_goal=effective_goal)
            typer.echo(_format_chat_response(payload, chat_state))
            _write_chat_report_file(output_report, payload, user_goal, base_dir=runtime.guard.root)
            _append_session_report(session_path, payload, user_goal)
            _write_chat_state(state_path, chat_state)
            continue
        try:
            goal = _build_chat_goal(effective_goal, chat_state, russian_only=russian_only)
            result = _run_chat_with_heartbeat(
                runtime,
                goal,
                progress_callback=lambda message: typer.echo(f"[agent] {message}"),
            )
            payload = result.model_dump(mode="json")
            auto_passes = 0
            while auto_passes < 2 and _should_auto_continue_chat_task(payload, effective_goal):
                auto_passes += 1
                typer.echo("[agent] Довожу активную задачу дальше без нового запроса...")
                follow_up_goal = (
                    f"Продолжай активную задачу: {effective_goal}. "
                    f"Закрой по возможности следующие хвосты: {'; '.join(_extract_pending_verifications(payload)[:6])}."
                )
                follow_up_state = _update_chat_state(chat_state, user_goal, payload, effective_goal=effective_goal)
                goal = _build_chat_goal(follow_up_goal, follow_up_state, russian_only=russian_only)
                result = _run_chat_with_heartbeat(
                    runtime,
                    goal,
                    progress_callback=lambda message: typer.echo(f"[agent] {message}"),
                )
                payload = result.model_dump(mode="json")
        except KeyboardInterrupt:
            raise
        except BaseException as exc:  # noqa: BLE001
            payload = {
                "status": "failed",
                "summary": f"Ошибка выполнения запроса: {exc}",
                "changes": [],
                "verification": [],
            }
        payload = _normalize_chat_payload(payload, russian_only=russian_only)
        chat_state = _update_chat_state(chat_state, user_goal, payload, effective_goal=effective_goal)
        assistant_message = str(payload.get("assistant_message") or "").strip()
        typer.echo(assistant_message or _format_chat_response(payload, chat_state))
        _write_chat_report_file(output_report, payload, user_goal, base_dir=runtime.guard.root)
        _append_session_report(session_path, payload, user_goal)
        _write_chat_state(state_path, chat_state)


@app.command("ui")
def chat_ui(
    host: str = typer.Option("127.0.0.1", "--host", help="Host for the local chat UI"),
    port: int = typer.Option(8765, "--port", help="Port for the local chat UI"),
    open_browser: bool = typer.Option(True, "--open-browser/--no-open-browser", help="Open the browser automatically"),
) -> None:
    """Start a local browser chat UI."""
    import os
    import signal
    import uvicorn

    url = f"http://{host}:{port}/ui"
    typer.echo(f"Локальный чат агента: {url}")
    typer.echo("PowerShell можно оставить открытым как сервер. Общаться с агентом нужно в браузере.")

    def stop_immediately(signum, frame) -> None:  # noqa: ANN001
        typer.echo("\nЛокальный чат остановлен.")
        os._exit(0)

    signal.signal(signal.SIGINT, stop_immediately)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop_immediately)

    if open_browser:
        webbrowser.open(url)
    config = uvicorn.Config(
        "coding_agent.api.app:app",
        host=host,
        port=port,
        reload=False,
        timeout_graceful_shutdown=1,
    )
    server = uvicorn.Server(config)
    try:
        server.run()
    except KeyboardInterrupt:
        typer.echo("Локальный чат остановлен.")


@app.command("inspect")
def inspect_workspace(workspace: str = typer.Option(..., "--workspace", help="Absolute or relative workspace path")) -> None:
    runtime = _runtime(workspace)
    repo_scan_loaded = runtime.orchestrator.repo_scanner.load_artifact_with_path()
    repo_scan_path = str(repo_scan_loaded[0]) if repo_scan_loaded else str(runtime.orchestrator.repo_scanner.default_artifact_path())
    repo_scan = repo_scan_loaded[1] if repo_scan_loaded else None
    info = {
        "workspace": str(runtime.guard.root),
        "is_git_repo": runtime.git_manager.detect_repo(),
        "recent_memories": [item.model_dump(mode="json") for item in runtime.memory_store.recent_memory(limit=5)],
        "repo_scan_artifact": repo_scan_path,
        "repo_scan": repo_scan.model_dump(mode="json") if repo_scan else None,
    }
    typer.echo(to_pretty_json(info))


@repo_app.command("scan")
def repo_scan_command(
    workspace: str = typer.Option(..., "--workspace", help="Absolute or relative workspace path"),
    output: str | None = typer.Option(None, "--output", help="Optional path for repo_scan JSON"),
) -> None:
    runtime = _runtime(workspace)
    target = Path(output).expanduser().resolve() if output else runtime.orchestrator.repo_scanner.default_artifact_path()
    path = runtime.orchestrator.repo_scanner.save_artifact(target)
    payload = runtime.orchestrator.repo_scanner.load_artifact(path)
    typer.echo(to_pretty_json({"saved_to": str(path), "repo_scan": payload.model_dump(mode="json") if payload else None}))


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


@doctor_app.command("repo-scan")
def doctor_repo_scan(
    workspace: str = typer.Option(..., "--workspace", help="Absolute or relative workspace path"),
) -> None:
    runtime = _runtime(workspace)
    loaded = runtime.orchestrator.repo_scanner.load_artifact_with_path()
    if loaded is None:
        saved_path = runtime.orchestrator.repo_scanner.save_artifact()
        loaded = runtime.orchestrator.repo_scanner.load_artifact_with_path(saved_path)
    if loaded is None:
        typer.echo(to_pretty_json({"ok": False, "error": "repo_scan artifact is unavailable"}))
        raise typer.Exit(code=1)
    artifact_path, repo_scan = loaded
    stat = artifact_path.stat()
    payload = {
        "ok": True,
        "artifact_path": str(artifact_path),
        "last_modified": stat.st_mtime,
        "summary": {
            "confirmed_frameworks": repo_scan.confirmed_frameworks,
            "api_surfaces": repo_scan.api_surfaces,
            "security_boundaries": repo_scan.security_boundaries,
        },
    }
    typer.echo(to_pretty_json(payload))


@doctor_app.command("final-report")
def doctor_final_report(
    workspace: str = typer.Option(..., "--workspace", help="Absolute or relative workspace path"),
) -> None:
    runtime = _runtime(workspace)
    report_path = runtime.guard.root / "artifacts" / "final_report.json"
    if not report_path.exists():
        typer.echo(to_pretty_json({"ok": False, "error": "final_report artifact is unavailable", "artifact_path": str(report_path)}))
        raise typer.Exit(code=1)
    import json

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    stat = report_path.stat()
    summary = {
        "status": payload.get("status"),
        "summary": payload.get("summary"),
        "changes_count": len(payload.get("changes") or []),
        "verification_count": len(payload.get("verification") or []),
        "findings_count": len(payload.get("findings") or []),
    }
    typer.echo(
        to_pretty_json(
            {
                "ok": True,
                "artifact_path": str(report_path),
                "last_modified": stat.st_mtime,
                "summary": summary,
            }
        )
    )


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


def _resolve_report_path(path: str, base_dir: Path | None = None) -> Path:
    target = Path(path).expanduser()
    if not target.is_absolute():
        base = base_dir.resolve() if base_dir else Path.cwd()
        target = (base / target).resolve()
    return target


def _write_report_file(path: str, payload: dict[str, object], base_dir: Path | None = None) -> None:
    target = _resolve_report_path(path, base_dir=base_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(to_pretty_json(payload), encoding="utf-8")
    typer.echo(f"Report saved to: {target}")


def _write_chat_report_file(path: str, payload: dict[str, object], user_goal: str, base_dir: Path | None = None) -> None:
    target = _resolve_report_path(path, base_dir=base_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_render_chat_report(payload, user_goal), encoding="utf-8")


def _session_report_path(path: str, base_dir: Path | None = None) -> Path:
    from datetime import datetime

    target = _resolve_report_path(path, base_dir=base_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return target.with_name(f"{target.stem}_session_{timestamp}{target.suffix}")


def _append_session_report(session_path: Path, payload: dict[str, object], user_goal: str) -> None:
    from datetime import datetime
    import json

    if session_path.exists():
        try:
            data = json.loads(session_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            data = {"messages": []}
    else:
        data = {"messages": []}
    data.setdefault("messages", []).append(
        {
            "timestamp": datetime.now().isoformat(),
            "goal": user_goal,
            "final_report": payload,
        }
    )
    session_path.write_text(to_pretty_json(data), encoding="utf-8")


def _render_chat_report(payload: dict[str, object], user_goal: str) -> str:
    lines = [
        f"# Последний запрос",
        "",
        user_goal,
        "",
        "## Ответ агента",
        "",
        _format_chat_response(payload, ChatSessionState(last_changed_files=_extract_changed_files(payload))),
        "",
    ]
    return "\n".join(lines).strip() + "\n"


def _maybe_handle_chat_message(user_goal: str, workspace_root: Path | None = None) -> dict[str, object] | None:
    lowered = user_goal.lower()
    wait_markers = (
        "сейчас пришлю",
        "сейчас тебе пришлю",
        "путь пришлю",
        "путь сейчас",
        "потом пришлю",
        "подожди",
        "пока не делай",
        "сначала пришлю",
    )
    if any(marker in lowered for marker in wait_markers):
        return {
            "status": "completed",
            "summary": "Жду следующий ввод. Как только пришлёте путь или конкретную задачу, начну разбор проекта и работу по файлам.",
            "changes": [],
            "verification": [],
            "findings": [],
            "next_steps": ["Прислать путь к проекту или уточнить, что именно нужно изучить/исправить."],
        }
    detected_path = _extract_requested_workspace(user_goal)
    if detected_path:
        current_root = workspace_root.resolve() if workspace_root else None
        if current_root and detected_path.resolve() != current_root:
            return {
                "status": "completed",
                "summary": (
                    f"Путь получил: {detected_path}. Но текущий chat запущен на workspace {current_root}. "
                    "Если этот путь существует, я могу переключиться на него прямо внутри чата."
                ),
                "changes": [],
                "verification": [],
                "findings": [],
                "next_steps": [
                    f"Повторно пришлите путь {detected_path} или команду с --workspace, и я переключу текущую chat-сессию.",
                    "Либо пришлите задачу для текущего workspace, если работать нужно всё-таки с ним.",
                ],
            }
        return {
            "status": "completed",
            "summary": f"Путь подтвердил: {detected_path}. Теперь напишите, что именно нужно изучить, проверить или исправить в этом проекте.",
            "changes": [],
            "verification": [],
            "findings": [],
            "next_steps": ["Например: 'сначала пойми архитектуру, потом назови 3 главных риска'."],
        }
    return None


def _extract_requested_workspace(user_goal: str) -> Path | None:
    workspace_match = re.search(r"--workspace\s+([A-Za-z]:\\[^\r\n\t]+)", user_goal.strip())
    raw_path: str | None = None
    if workspace_match:
        raw_path = workspace_match.group(1).strip().strip('"')
    else:
        match = re.search(r"^([A-Za-z]:\\[^\r\n\t]+)\s*$", user_goal.strip())
        if match:
            raw_path = match.group(1).strip().strip('"')
    if not raw_path:
        return None
    raw_path = raw_path.rstrip("\\")
    try:
        return Path(raw_path).expanduser()
    except Exception:  # noqa: BLE001
        return None


def _maybe_switch_chat_workspace(user_goal: str, runtime: Runtime) -> Runtime | None:
    requested = _extract_requested_workspace(user_goal)
    if requested is None:
        return None
    requested = requested.resolve()
    if requested == runtime.guard.root:
        return None
    if not requested.exists() or not requested.is_dir():
        return None
    try:
        new_runtime = _runtime(str(requested), model=_select_default_chat_model(str(requested)))
        new_runtime.llm.warmup()
    except Exception:
        return None
    return new_runtime


def _run_chat_with_heartbeat(
    runtime: Runtime,
    goal: str,
    progress_callback,
    heartbeat_seconds: float = 12.0,
):
    result_box: dict[str, object] = {}
    error_box: dict[str, BaseException] = {}
    last_progress = {"message": "Запускаю рабочий цикл агента..."}

    def wrapped_progress(message: str) -> None:
        last_progress["message"] = message
        progress_callback(message)

    def target() -> None:
        try:
            result_box["result"] = runtime.orchestrator.run_chat(goal, progress_callback=wrapped_progress)
        except BaseException as exc:  # noqa: BLE001
            error_box["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    last_notice = time.monotonic()
    while worker.is_alive():
        worker.join(timeout=1.0)
        if worker.is_alive() and time.monotonic() - last_notice >= heartbeat_seconds:
            typer.echo(f"[agent] Всё ещё работаю... Последний этап: {last_progress['message']}")
            last_notice = time.monotonic()
    if "error" in error_box:
        raise error_box["error"]
    return result_box["result"]


def _select_default_chat_model(workspace: str) -> str | None:
    _, config_path, _ = _runtime_paths(workspace)
    try:
        config = load_app_config(config_path)
    except Exception:
        return None
    if getattr(config.llm, "provider", "") != "ollama":
        return config.llm.model
    installed = _installed_ollama_models()
    for candidate in PREFERRED_CHAT_MODELS:
        if candidate in installed:
            return candidate
    if config.llm.model in installed:
        return config.llm.model
    return config.llm.model


def _installed_ollama_models() -> set[str]:
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=15,
            encoding="utf-8",
            errors="replace",
        )
    except Exception:
        return set()
    if result.returncode != 0:
        return set()
    models: set[str] = set()
    for line in (result.stdout or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("NAME"):
            continue
        parts = stripped.split()
        if parts:
            models.add(parts[0])
    return models


def _chat_state_path(report_path: Path) -> Path:
    return report_path.with_name(f"{report_path.stem}_state.json")


def _load_chat_state(path: Path) -> ChatSessionState:
    import json

    if not path.exists():
        return ChatSessionState()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return ChatSessionState()
    try:
        return ChatSessionState.model_validate(payload)
    except Exception:  # noqa: BLE001
        return ChatSessionState()


def _write_chat_state(path: Path, state: ChatSessionState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_pretty_json(state.model_dump(mode="json")), encoding="utf-8")


def _build_chat_goal(user_goal: str, state: ChatSessionState, russian_only: bool) -> str:
    strategy = _chat_task_strategy(user_goal)
    sections = [
        "Ты работаешь в режиме живого чата с пользователем.",
        "Пользователь пишет обычным текстом. Сам исследуй код, вноси изменения в файлы и запускай проверки при необходимости.",
        "Не пиши промежуточный final_report пользователю. Выполни задачу как инженер и верни нормальный итог по изменениям.",
        "Если меняешь код, опирайся на текущий workspace и предыдущий контекст сессии.",
        "Не останавливайся после первой частичной находки. Если задача большая, сделай несколько внутренних шагов подряд до осмысленного результата или до конкретного блокера.",
        f"Режим работы для этого запроса: {strategy['mode']}.",
        f"Стратегия: {strategy['instruction']}",
    ]
    if state.active_task:
        sections.append(f"Активная задача сессии: {state.active_task}")
        sections.append(f"Статус активной задачи: {state.task_status}")
    if state.current_goal:
        sections.append(f"Текущая долгоживущая цель сессии: {state.current_goal}")
    if state.last_summary:
        sections.append(f"Итог прошлого хода: {state.last_summary}")
    if state.last_changed_files:
        sections.append("Последние изменённые файлы: " + ", ".join(state.last_changed_files[:8]))
    if state.pending_verifications:
        sections.append("Незавершённые проверки: " + "; ".join(state.pending_verifications[:6]))
    recent_messages = state.messages[-6:]
    if recent_messages:
        history_text = "\n".join(f"{item.role}: {item.content}" for item in recent_messages)
        sections.append(f"Недавняя история:\n{history_text}")
    sections.append(f"Новый запрос пользователя:\n{user_goal}")
    if russian_only:
        sections.append("Верни итог только на русском языке.")
    return "\n\n".join(sections)


def _update_chat_state(
    state: ChatSessionState,
    user_goal: str,
    payload: dict[str, object],
    effective_goal: str | None = None,
) -> ChatSessionState:
    assistant_summary = str(payload.get("summary") or "").strip()
    assistant_message = str(payload.get("assistant_message") or assistant_summary).strip()
    updated_messages = list(state.messages)
    updated_messages.append(ChatHistoryEntry(role="user", content=user_goal))
    if assistant_message:
        updated_messages.append(ChatHistoryEntry(role="assistant", content=assistant_message))
    changed_files = list(payload.get("changed_files") or []) or _extract_changed_files(payload)
    pending = _extract_pending_verifications(payload)
    goal_for_state = (effective_goal or user_goal).strip()
    active_task = state.active_task
    task_status = state.task_status
    if _is_continue_request(user_goal) and active_task:
        goal_for_state = active_task
    elif goal_for_state:
        active_task = goal_for_state
    status = str(payload.get("status") or "").strip().lower()
    if status == "failed":
        task_status = "blocked"
    elif pending:
        task_status = "in_progress"
    elif status == "completed":
        task_status = "completed"
    elif status:
        task_status = status
    return ChatSessionState(
        current_goal=goal_for_state,
        active_task=active_task,
        task_status=task_status,
        last_summary=assistant_summary,
        last_changed_files=changed_files or state.last_changed_files,
        pending_verifications=pending,
        messages=updated_messages[-12:],
        turn_count=state.turn_count + 1,
    )


def _extract_changed_files(payload: dict[str, object]) -> list[str]:
    text_fragments: list[str] = []
    for key in ("changes", "verification", "next_steps"):
        raw = payload.get(key) or []
        if isinstance(raw, list):
            text_fragments.extend(str(item) for item in raw)
    findings = payload.get("findings") or []
    if isinstance(findings, list):
        for item in findings:
            if isinstance(item, dict):
                text_fragments.append(str(item.get("evidence") or ""))
                text_fragments.append(str(item.get("recommendation") or ""))
    path_pattern = re.compile(r'([A-Za-z0-9_./\\-]+\.(?:py|yaml|yml|json|toml|md|txt))')
    seen: set[str] = set()
    results: list[str] = []
    for fragment in text_fragments:
        for match in path_pattern.findall(fragment):
            normalized = match.replace("\\", "/")
            if normalized in seen:
                continue
            seen.add(normalized)
            results.append(normalized)
    return results[:12]


def _extract_pending_verifications(payload: dict[str, object]) -> list[str]:
    pending: list[str] = []
    for item in payload.get("next_steps") or []:
        text = str(item).strip()
        if text:
            pending.append(text)
    for item in payload.get("verification") or []:
        text = str(item).strip()
        if not text:
            continue
        lowered = text.lower()
        if "exit_code" in lowered or ": 1" in lowered or "failed" in lowered or "требует" in lowered:
            pending.append(text)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in pending:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped[:8]


def _format_chat_response(payload: dict[str, object], state: ChatSessionState) -> str:
    lines: list[str] = []
    summary = str(payload.get("summary") or "").strip() or "Ход выполнен."
    lines.append(summary)
    changes = [str(item).strip() for item in (payload.get("changes") or []) if str(item).strip()]
    verification = [str(item).strip() for item in (payload.get("verification") or []) if str(item).strip()]
    findings = payload.get("findings") or []
    next_steps = [str(item).strip() for item in (payload.get("next_steps") or []) if str(item).strip()]
    if changes:
        lines.append("")
        lines.append("Что сделал:")
        lines.extend(f"- {item}" for item in changes[:5])
    concrete_findings: list[str] = []
    if isinstance(findings, list):
        for item in findings[:3]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            recommendation = str(item.get("recommendation") or "").strip()
            if title:
                concrete_findings.append(f"{title}" + (f" — {recommendation}" if recommendation else ""))
    if concrete_findings:
        lines.append("")
        lines.append("На что обратил внимание:")
        lines.extend(f"- {item}" for item in concrete_findings)
    if verification:
        lines.append("")
        lines.append("Чем проверил:")
        lines.extend(f"- {item}" for item in verification[:5])
    if next_steps:
        lines.append("")
        lines.append("Что ещё осталось:")
        lines.extend(f"- {item}" for item in next_steps[:4])
    if state.last_changed_files:
        lines.append("")
        lines.append("Последние затронутые файлы: " + ", ".join(state.last_changed_files[:6]))
    if state.active_task and state.task_status == "in_progress":
        lines.append("")
        lines.append(f"Активная задача остаётся в работе: {state.active_task}")
    return "\n".join(lines)


def _normalize_chat_payload(payload: dict[str, object], russian_only: bool) -> dict[str, object]:
    import re

    normalized = {
        "status": payload.get("status") if payload.get("status") in {"completed", "stopped", "failed"} else "failed",
        "summary": str(payload.get("summary") or ""),
        "changes": list(payload.get("changes") or []),
        "verification": list(payload.get("verification") or []),
        "findings": list(payload.get("findings") or []),
        "next_steps": list(payload.get("next_steps") or []),
        "changed_files": list(payload.get("changed_files") or []),
        "assistant_message": str(payload.get("assistant_message") or ""),
    }
    ansi = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
    normalized["summary"] = ansi.sub("", str(normalized["summary"])).replace("\u001b", "").strip()
    normalized["changes"] = [ansi.sub("", str(x)).replace("\u001b", "").strip() for x in normalized["changes"]]
    normalized["verification"] = [ansi.sub("", str(x)).replace("\u001b", "").strip() for x in normalized["verification"]]
    cleaned_findings: list[dict[str, str]] = []
    for item in normalized["findings"]:
        if not isinstance(item, dict):
            continue
        cleaned_findings.append(
            {
                "title": ansi.sub("", str(item.get("title", ""))).replace("\u001b", "").strip(),
                "severity": ansi.sub("", str(item.get("severity", ""))).replace("\u001b", "").strip(),
                "status": ansi.sub("", str(item.get("status", ""))).replace("\u001b", "").strip(),
                "evidence": ansi.sub("", str(item.get("evidence", ""))).replace("\u001b", "").strip(),
                "recommendation": ansi.sub("", str(item.get("recommendation", ""))).replace("\u001b", "").strip(),
            }
        )
    normalized["findings"] = cleaned_findings
    normalized["next_steps"] = [ansi.sub("", str(x)).replace("\u001b", "").strip() for x in normalized["next_steps"]]
    normalized["changed_files"] = [ansi.sub("", str(x)).replace("\u001b", "").strip() for x in normalized["changed_files"]]
    normalized["assistant_message"] = ansi.sub("", str(normalized["assistant_message"])).replace("\u001b", "").strip()
    if (
        russian_only
        and normalized["status"] != "failed"
        and normalized["summary"]
        and re.search(r"[A-Za-z]{5,}", normalized["summary"])
        and (normalized["changes"] or normalized["verification"] or normalized["findings"])
    ):
        normalized["summary"] = "Ответ сформирован. Итог приведён на русском языке."
    return normalized


def _chat_task_strategy(user_goal: str) -> dict[str, str]:
    lowered = user_goal.lower()
    if any(token in lowered for token in ("риск", "архитект", "анализ", "audit", "analy", "проверь проект")):
        return {
            "mode": "analysis",
            "instruction": (
                "Сначала построй обзор проекта, найди ключевые модули и собери evidence через read_file/search_in_files/"
                "run_command. Не предлагай правки, пока не сформулируешь подтверждённые выводы."
            ),
        }
    if any(token in lowered for token in ("исправ", "fix", "bug", "ошиб", "почини", "измени", "изменить", "обнови")):
        return {
            "mode": "fix",
            "instruction": (
                "Сначала локализуй источник дефекта, затем внеси минимально достаточное изменение, после чего обязательно "
                "запусти релевантную проверку и коротко отчитайся, что именно исправлено."
            ),
        }
    if any(token in lowered for token in ("рефактор", "refactor", "cleanup", "упрости", "перестрой")):
        return {
            "mode": "refactor",
            "instruction": (
                "Сначала пойми текущую структуру и зависимости, затем меняй код небольшими безопасными шагами, сохраняя "
                "поведение. После изменений проверь, что ключевые сценарии не сломаны."
            ),
        }
    if any(token in lowered for token in ("добав", "implement", "feature", "сделай команду", "новую команду", "новый endpoint", "запусти", "запустить", "обучи", "обучение")):
        return {
            "mode": "feature",
            "instruction": (
                "Сначала найди подходящую точку расширения в проекте, затем реализуй фичу end-to-end, добавь или обнови "
                "проверки и объясни, где именно теперь живёт новая возможность."
            ),
        }
    return {
        "mode": "general",
        "instruction": (
            "Сначала уточни структуру затронутой части проекта по коду, затем выбери следующий практический шаг и доведи "
            "его до осмысленного промежуточного результата."
        ),
    }


def _is_continue_request(user_goal: str) -> bool:
    lowered = user_goal.strip().lower()
    return lowered in {
        "продолжай",
        "продолжить",
        "продолжи",
        "continue",
        "go on",
        "дальше",
        "еще",
        "ещё",
    }


def _resolve_chat_request(user_goal: str, state: ChatSessionState) -> tuple[str, dict[str, object] | None]:
    if _is_memory_request(user_goal):
        summary_parts: list[str] = []
        if state.active_task:
            summary_parts.append(f"Активная задача: {state.active_task} ({state.task_status}).")
        if state.last_summary:
            summary_parts.append(f"Последний итог: {state.last_summary}")
        if state.last_changed_files:
            summary_parts.append("Последние файлы в контексте: " + ", ".join(state.last_changed_files[:8]))
        if state.pending_verifications:
            summary_parts.append("Незавершённые проверки: " + "; ".join(state.pending_verifications[:6]))
        recent_user_messages = [item.content for item in state.messages if item.role == "user"][-3:]
        if recent_user_messages:
            summary_parts.append("Последние запросы: " + " | ".join(recent_user_messages))
        payload = {
            "status": "completed",
            "summary": "\n".join(summary_parts) if summary_parts else "Пока в этой chat-сессии нет сохранённого контекста.",
            "changes": [],
            "verification": [],
            "findings": [],
            "next_steps": ["Напишите задачу действием: 'исправь...', 'добавь...', 'измени...', 'рефакторинг...'."],
        }
        return user_goal, payload
    if _looks_like_pasted_assistant_fragment(user_goal):
        payload = {
            "status": "completed",
            "summary": "Похоже, это обрывок прошлого ответа, а не новая задача. Я не буду записывать его как активную цель.",
            "changes": [],
            "verification": [],
            "findings": [],
            "next_steps": ["Сформулируйте действие: например, 'исправь проверку адаптивных расписаний' или 'добавь тест на env.py'."],
        }
        return state.active_task or user_goal, payload
    if _is_continue_request(user_goal):
        if state.active_task:
            return (
                f"Продолжай активную задачу: {state.active_task}. "
                + (
                    "Сначала закрой незавершённые проверки: " + "; ".join(state.pending_verifications[:6]) + "."
                    if state.pending_verifications
                    else "Продолжи работу до более завершённого результата."
                ),
                None,
            )
        payload = {
            "status": "completed",
            "summary": "Пока нет активной задачи, которую можно продолжить.",
            "changes": [],
            "verification": [],
            "findings": [],
            "next_steps": ["Сформулируйте задачу обычной фразой, и я начну работу."],
        }
        return user_goal, payload
    return user_goal, None


def _is_memory_request(user_goal: str) -> bool:
    lowered = user_goal.strip().lower()
    return lowered in {
        "что ты помнишь?",
        "что ты помнишь",
        "что помнишь?",
        "что помнишь",
        "память",
        "покажи память",
        "контекст",
        "покажи контекст",
    }


def _looks_like_pasted_assistant_fragment(user_goal: str) -> bool:
    text = user_goal.strip()
    lowered = text.lower()
    if len(text) < 20:
        return False
    fragment_markers = (
        "попробуйте различные комбинации",
        "проверьте, как изменяются",
        "эти шаги помогут",
        "контекст анализа:",
        "главных риска:",
        "что проверить дальше",
    )
    action_markers = (
        "исправ",
        "добав",
        "измени",
        "сделай",
        "рефактор",
        "проверь",
        "найди",
        "обнови",
        "удали",
        "создай",
    )
    return any(marker in lowered for marker in fragment_markers) and not any(marker in lowered for marker in action_markers)


def _should_auto_continue_chat_task(payload: dict[str, object], effective_goal: str) -> bool:
    status = str(payload.get("status") or "").strip().lower()
    if status != "completed":
        return False
    if not _extract_pending_verifications(payload):
        return False
    text = " ".join(
        [
            str(payload.get("summary") or ""),
            *[str(item) for item in (payload.get("next_steps") or [])],
        ]
    ).lower()
    blocker_markers = (
        "нужен",
        "пришлите",
        "уточните",
        "невозможно",
        "ошибка",
        "blocked",
        "требуется",
    )
    if any(marker in text for marker in blocker_markers):
        return False
    goal_lower = effective_goal.lower()
    return any(token in goal_lower for token in ("исправ", "добав", "рефактор", "архитект", "риск", "проект", "проверь"))


def _propose_next_actions(runtime: Runtime, goal: str) -> list[dict[str, str]]:
    retrieval = runtime.retrieval_service.retrieve(goal, max_chunks=6)
    context = "\n".join(f"- {chunk.path}" for chunk in retrieval.file_chunks[:8]) or "- No files matched lexical retrieval"
    prompt = (
        "Return valid JSON only.\n"
        "You are an engineering assistant. Suggest 3 practical next improvements for this repository.\n"
        "Use Russian language for all proposal text fields.\n"
        "Each proposal must include id, title, impact, risk, action_goal.\n"
        f"Current objective: {goal}\n"
        f"Relevant files:\n{context}\n"
        'Schema: {"proposals":[{"id":"1","title":"...","impact":"low|medium|high","risk":"low|medium|high","action_goal":"..."}]}'
    )
    response = runtime.llm.chat([ChatMessage(role="user", content=prompt)], json_mode=True)
    from coding_agent.util.json_utils import extract_json_object

    try:
        payload = extract_json_object(response.content)
        proposals = payload.get("proposals", [])
    except Exception:
        proposals = []
    cleaned: list[dict[str, str]] = []
    for idx, item in enumerate(proposals, start=1):
        if not isinstance(item, dict):
            continue
        cleaned.append(
            {
                "id": str(item.get("id") or idx),
                "title": str(item.get("title") or f"Suggestion {idx}"),
                "impact": str(item.get("impact") or "medium"),
                "risk": str(item.get("risk") or "medium"),
                "action_goal": str(item.get("action_goal") or item.get("title") or ""),
            }
        )
    if cleaned:
        return cleaned[:5]
    return [
        {
            "id": "1",
            "title": "Проверить текущую архитектуру и риски",
            "impact": "high",
            "risk": "low",
            "action_goal": "Проверить структуру репозитория и кратко описать архитектуру с ключевыми рисками и возможностями улучшения.",
        },
        {
            "id": "2",
            "title": "Усилить восстановление JSON от локальной модели",
            "impact": "high",
            "risk": "medium",
            "action_goal": "Улучшить устойчивость парсинга ответов локальной модели и добавить тесты для поврежденных ответов.",
        },
        {
            "id": "3",
            "title": "Стабилизировать диагностику Ollama",
            "impact": "medium",
            "risk": "low",
            "action_goal": "Улучшить диагностику doctor и поведение fallback при периодических ответах Ollama 503.",
        },
    ]


def _print_proposals(proposals: list[dict[str, str]]) -> None:
    typer.echo("\nПредложенные улучшения:")
    for item in proposals:
        typer.echo(f"{item['id']}. {item['title']} (impact={item['impact']}, risk={item['risk']})")


def _goal_from_choice(choice: str, proposals: list[dict[str, str]]) -> str | None:
    for item in proposals:
        if item["id"] == choice:
            return item["action_goal"]
    return None


def _requires_approval(goal: str) -> bool:
    return bool(re.search(r"\b(delete|remove|reset|revert|rename|move|commit)\b", goal, flags=re.IGNORECASE))


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
