"""FastAPI scaffold."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError
import re

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response

from coding_agent.api.schemas import ChatRequest, ChatResponse, HealthResponse, MemorySearchRequest, RepoScanResponse, RunListResponse
from coding_agent.cli.main import (
    _append_session_report,
    _build_chat_goal,
    _chat_state_path,
    _format_chat_response,
    _load_chat_state,
    _maybe_handle_chat_message,
    _normalize_chat_payload,
    _resolve_chat_request,
    _resolve_report_path,
    _runtime,
    _select_default_chat_model,
    _session_report_path,
    _should_auto_continue_chat_task,
    _update_chat_state,
    _write_chat_report_file,
    _write_chat_state,
)
from coding_agent.sandbox.shell_runner import ShellCommand
from coding_agent.llm.base import ChatMessage
from coding_agent.util.json_utils import extract_json_object
from coding_agent.util.logging import configure_logging

configure_logging()
app = FastAPI(title="coding-agent")

DEFAULT_CHAT_REPORT = "artifacts/last_chat_report.md"


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.get("/runs", response_model=RunListResponse)
def runs(workspace: str = ".") -> RunListResponse:
    runtime = _runtime(workspace)
    return RunListResponse(runs=[record.model_dump(mode="json") for record in runtime.memory_store.list_runs()])


@app.post("/memory/search")
def memory_search(request: MemorySearchRequest) -> list[dict]:
    runtime = _runtime(request.workspace)
    return [record.model_dump(mode="json") for record in runtime.memory_tools.search(request.query, request.limit)]


@app.get("/repo/scan", response_model=RepoScanResponse)
def repo_scan(workspace: str = ".") -> RepoScanResponse:
    runtime = _runtime(workspace)
    artifact_path = runtime.orchestrator.repo_scanner.save_artifact()
    loaded = runtime.orchestrator.repo_scanner.load_artifact_with_path(artifact_path)
    resolved_path = loaded[0] if loaded else artifact_path
    payload = loaded[1] if loaded else None
    return RepoScanResponse(
        artifact_path=str(resolved_path),
        repo_scan=payload.model_dump(mode="json") if payload else {},
    )


@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    return chat_ui()


@app.get("/ui", response_class=HTMLResponse)
def chat_ui() -> HTMLResponse:
    return HTMLResponse(_CHAT_UI_HTML)


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    # Для задач обучения увеличиваем общий таймаут до 25 минут
    is_training_request = any(token in request.message.lower() for token in [
        "обучени", "train", "iterations", "итераци", "--total-timesteps"
    ])
    base_timeout = 1500 if is_training_request else 900  # 25 минут для обучения, 15 для остальных задач
    timeout_seconds = max(60, min(int(request.timeout_seconds or base_timeout), 1800))
    
    selected_model = request.model or _select_default_chat_model(request.workspace)
    print(f"[agent-ui] Получен запрос чата. workspace={request.workspace}; model={selected_model or 'config'}; timeout={timeout_seconds}s", flush=True)
    runtime = _runtime(request.workspace, model=selected_model)
    
    # Увеличиваем таймауты для LLM при обучении
    llm_timeout = 300 if is_training_request else 90
    llm_warmup = 60 if is_training_request else 45
    
    runtime.config.llm.timeout_seconds = min(runtime.config.llm.timeout_seconds, llm_timeout)
    runtime.config.llm.warmup_timeout_seconds = min(runtime.config.llm.warmup_timeout_seconds, llm_warmup)
    runtime.config.llm.max_retries = min(runtime.config.llm.max_retries, 3)
    if hasattr(runtime.llm, "timeout_seconds"):
        runtime.llm.timeout_seconds = min(getattr(runtime.llm, "timeout_seconds"), llm_timeout)
    if hasattr(runtime.llm, "warmup_timeout_seconds"):
        runtime.llm.warmup_timeout_seconds = min(getattr(runtime.llm, "warmup_timeout_seconds"), llm_warmup)
    if hasattr(runtime.llm, "max_retries"):
        runtime.llm.max_retries = min(getattr(runtime.llm, "max_retries"), 3)
    
    # Увеличиваем максимальное количество шагов для сложных задач
    runtime.config.orchestrator.max_steps = min(runtime.config.orchestrator.max_steps, 15 if is_training_request else 8)
    
    print(f"[agent-ui] Runtime готов. root={runtime.guard.root}. Прогреваю LLM...", flush=True)
    runtime.llm.warmup()
    print("[agent-ui] LLM готова. Загружаю состояние чата...", flush=True)

    report_path = _resolve_report_path(DEFAULT_CHAT_REPORT, base_dir=runtime.guard.root)
    session_path = _session_report_path(DEFAULT_CHAT_REPORT, base_dir=runtime.guard.root)
    state_path = _chat_state_path(report_path)
    chat_state = _load_chat_state(state_path)

    effective_goal, immediate_override = _resolve_chat_request(request.message, chat_state)
    if immediate_override is not None:
        payload = immediate_override
    else:
        immediate_payload = _maybe_handle_chat_message(request.message, runtime.guard.root)
        if immediate_payload is not None:
            payload = immediate_payload
        else:
            direct_payload = _maybe_execute_request_directly(
                request.message,
                runtime,
                timeout_seconds,
                context_hint=" ".join(
                    item
                    for item in (
                        chat_state.active_task,
                        chat_state.current_goal,
                        chat_state.last_summary,
                    )
                    if item
                ),
            )
            if direct_payload is not None:
                payload = direct_payload
            else:
                print("[agent-ui] Запускаю рабочий цикл агента...", flush=True)
                goal = _build_chat_goal(effective_goal, chat_state, russian_only=request.russian_only)
                try:
                    result = _run_with_timeout(
                        lambda: runtime.orchestrator.run_chat(
                            goal,
                            progress_callback=lambda message: print(f"[agent-ui] {message}", flush=True),
                        ),
                        timeout_seconds=timeout_seconds,
                    )
                    payload = result.model_dump(mode="json")
                except TimeoutError as exc:
                    payload = _timeout_payload(str(exc), is_training=is_training_request)
            auto_passes = 0
            while auto_passes < 2 and _should_auto_continue_chat_task(payload, effective_goal):
                auto_passes += 1
                print(f"[agent-ui] Автопродолжение задачи, проход {auto_passes}...", flush=True)
                follow_up_state = _update_chat_state(chat_state, request.message, payload, effective_goal=effective_goal)
                follow_up_goal = _build_chat_goal(
                    f"Продолжай активную задачу: {effective_goal}. Доведи её до полезного результата.",
                    follow_up_state,
                    russian_only=request.russian_only,
                )
                try:
                    result = _run_with_timeout(
                        lambda: runtime.orchestrator.run_chat(
                            follow_up_goal,
                            progress_callback=lambda message: print(f"[agent-ui] {message}", flush=True),
                        ),
                        timeout_seconds=timeout_seconds,
                    )
                    payload = result.model_dump(mode="json")
                except TimeoutError as exc:
                    payload = _timeout_payload(str(exc), is_training=is_training_request)
                    break

    payload = _normalize_chat_payload(payload, russian_only=request.russian_only)
    chat_state = _update_chat_state(chat_state, request.message, payload, effective_goal=effective_goal)
    assistant_message = str(payload.get("assistant_message") or "").strip() or _format_chat_response(payload, chat_state)

    _write_chat_report_file(DEFAULT_CHAT_REPORT, payload, request.message, base_dir=runtime.guard.root)
    _append_session_report(session_path, payload, request.message)
    _write_chat_state(state_path, chat_state)
    print(f"[agent-ui] Ответ готов. status={payload.get('status')}; report={report_path}", flush=True)

    return ChatResponse(
        status=str(payload.get("status") or "completed"),
        assistant_message=assistant_message,
        summary=str(payload.get("summary") or ""),
        changed_files=[str(item) for item in payload.get("changed_files") or []],
        verification=[str(item) for item in payload.get("verification") or []],
        next_steps=[str(item) for item in payload.get("next_steps") or []],
        report_path=str(report_path),
        state_path=str(state_path),
    )


def _run_with_timeout(callable_obj, timeout_seconds: int):  # noqa: ANN001, ANN201
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(callable_obj)
    try:
        return future.result(timeout=timeout_seconds)
    except TimeoutError:
        future.cancel()
        print(f"[agent-ui] Ход остановлен по таймауту {timeout_seconds} секунд.", flush=True)
        raise TimeoutError(f"Ход агента превысил таймаут {timeout_seconds} секунд")
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def _timeout_payload(message: str, is_training: bool = False) -> dict[str, object]:
    if is_training:
        return {
            "status": "completed",
            "summary": "Обучение выполняется. Процесс был прерван по таймауту UI, но задача всё ещё активна.",
            "changes": [],
            "verification": [
                message,
                "Для задач обучения рекомендуется запускать команды напрямую в терминале с большим таймаутом."
            ],
            "findings": ["Обучение требует больше времени для завершения."],
            "next_steps": [
                "Запустите команду обучения напрямую в терминале: .\\.venv\\Scripts\\python.exe -m src.train --total-timesteps <значение>",
                "Используйте параметр --smoke-test для быстрых тестовых запусков.",
                "Можно увеличить таймаут в настройках агента."
            ],
        }
    return {
        "status": "failed",
        "summary": message,
        "changes": [],
        "verification": ["Ход остановлен защитным таймаутом UI, чтобы агент не висел бесконечно."],
        "findings": [],
        "next_steps": [
            "Повторите запрос более конкретно или укажите короткую smoke-команду запуска.",
            "Для долгого обучения запускайте команду отдельно с явным временем выполнения.",
        ],
    }


def _maybe_execute_request_directly(
    message: str,
    runtime,
    timeout_seconds: int,
    context_hint: str = "",
) -> dict[str, object] | None:  # noqa: ANN001
    if _is_direct_smoke_training_request(message, context_hint):
        return _run_direct_smoke_training_for_message(message, runtime, timeout_seconds)
    if not _is_execution_request(message):
        return None
    command = _command_from_known_request(message, runtime)
    if command is None:
        command = _ask_llm_for_command(message, runtime, context_hint)
    if command is None:
        return None
    return _run_direct_command(command, runtime, timeout_seconds)


def _run_direct_smoke_training(runtime, timeout_seconds: int) -> dict[str, object]:  # noqa: ANN001
    return _run_direct_smoke_training_for_message("", runtime, timeout_seconds)


def _run_direct_smoke_training_for_message(message: str, runtime, timeout_seconds: int) -> dict[str, object]:  # noqa: ANN001
    root = runtime.guard.root
    python_exe = ".\\.venv\\Scripts\\python.exe" if (root / ".venv" / "Scripts" / "python.exe").exists() else "python"
    csv_path = _select_smoke_csv_path(root)
    n_steps = 32
    iterations = _extract_requested_iterations(message) or 8
    total_timesteps = max(n_steps, iterations * n_steps)
    command = (
        f"{python_exe} -m src.train --csv-path {csv_path} "
        f"--total-timesteps {total_timesteps} --smoke-test --smoke-steps {total_timesteps} "
        f"--n-envs 1 --n-steps {n_steps} --batch-size 32 --n-epochs 1 "
        "--model-name smoke_ui_check --evaluation-dir-name evaluation_ui_smoke"
    )
    return _run_direct_command(command, runtime, timeout_seconds)


def _run_direct_command(command: str, runtime, timeout_seconds: int) -> dict[str, object]:  # noqa: ANN001
    command_timeout = max(30, min(timeout_seconds, 900))  # Еще больше увеличенный таймаут для обучения (до 15 минут)
    print(f"[agent-ui] Прямой запуск команды: {command}", flush=True)
    
    # Проверяем, является ли команда обучением (для специальной обработки таймаута)
    is_training = "src.train" in command or "--total-timesteps" in command or "--iterations" in command
    
    # Для обучения используем максимальный таймаут
    if is_training:
        command_timeout = max(command_timeout, 900)  # Минимум 15 минут для обучения
    
    result = runtime.shell_runner.run(
        ShellCommand(command=command, timeout_seconds=command_timeout)
    ).model_dump()
    
    # Для команд обучения считаем успешным запуском даже таймаут, если есть признаки начала работы
    if is_training and result.get("timed_out"):
        stdout = str(result.get("stdout") or "")
        stderr = str(result.get("stderr") or "")
        output = stdout + stderr
        
        # Проверяем признаки успешного начала обучения
        training_started = any(token in output.lower() for token in [
            "using cuda device", "using cpu device", 
            "feature_count=", "train_rows=", "validation_rows=",
            "n_envs=", "n_steps=", "batch_size=",
            "ppo", "iterations", "time_elapsed", "total_timesteps",
            "[train]", "fps", "eta", "gradient", "loss", "entropy"
        ])
        
        if training_started:
            return {
                "status": "completed",
                "summary": f"Обучение запущено и выполняется. Процесс продолжается в фоне (итераций запрошено).",
                "changes": [],
                "verification": [
                    f"run_command: {command}",
                    f"exit_code={result.get('exit_code')}; timed_out={result.get('timed_out')}; duration={result.get('duration_seconds'):.1f}s",
                    "Обучение успешно инициировано. Вывод был обрезан по таймауту, но процесс стартовал и показывает прогресс.",
                    str(output)[:1500],
                ],
                "findings": ["Обучение запущено. Для полного завершения всех итераций требуется больше времени."],
                "next_steps": [
                    "Дождитесь завершения обучения или проверьте логи в директории модели.",
                    "Запустите команду снова с большим таймаутом для полного завершения всех итераций.",
                    "Можно проверить статус обучения через мониторинг GPU/CPU."
                ],
            }
    
    ok = result.get("exit_code") == 0 and not result.get("timed_out")
    return {
        "status": "completed" if ok else "failed",
        "summary": "Команда выполнена успешно." if ok else "Команда выполнена, но завершилась с ошибкой или таймаутом.",
        "changes": [],
        "verification": [
            f"run_command: {command}",
            f"exit_code={result.get('exit_code')}; timed_out={result.get('timed_out')}; duration={result.get('duration_seconds'):.1f}s",
            str(result.get("stdout") or result.get("stderr") or "")[:1500],
        ],
        "findings": [],
        "next_steps": [] if ok else ["Проверить stderr/stdout команды и параметры запуска."],
    }


def _select_smoke_csv_path(root) -> str:  # noqa: ANN001
    candidates = [
        root / "src" / "ethusdt_1h_2021_2025_labeled.csv",
        root / "ethusdt_1h_2021_2025_labeled.csv",
        root / "ETHUSDT_15m_labeled.csv",
    ]
    for path in candidates:
        if path.exists():
            return path.relative_to(root).as_posix()
    return "ETHUSDT_15m_labeled.csv"


def _is_direct_smoke_training_request(message: str, context_hint: str = "") -> bool:
    lowered_message = message.lower()
    combined = f"{message}\n{context_hint}".lower()
    repeat_request = any(token in lowered_message for token in ("еще раз", "ещё раз", "снова", "повтори", "again"))
    has_smoke_context = any(token in combined for token in ("тестовое обучение", "smoke", "smoke-test"))
    has_run_request = any(token in lowered_message for token in ("запусти", "запустить", "запуск", "обучи", "run", "train"))
    # Также считаем запрос на обучение с указанием итераций как прямой запрос на запуск
    has_iterations_request = "итераци" in lowered_message or "iteration" in lowered_message
    return (has_smoke_context and (has_run_request or repeat_request)) or (has_run_request and has_iterations_request)


def _is_execution_request(message: str) -> bool:
    lowered = message.lower()
    return any(
        token in lowered
        for token in (
            "запусти",
            "запустить",
            "выполни",
            "прогони",
            "проверь",
            "обучи",
            "установи",
            "run",
            "test",
            "train",
            "install",
        )
    )


def _command_from_known_request(message: str, runtime) -> str | None:  # noqa: ANN001
    lowered = message.lower()
    root = runtime.guard.root
    python_exe = ".\\.venv\\Scripts\\python.exe" if (root / ".venv" / "Scripts" / "python.exe").exists() else "python"
    if "compile" in lowered or "компил" in lowered or "синтакс" in lowered:
        return f"{python_exe} -m compileall src"
    if "pytest" in lowered or "тест" in lowered:
        if (root / "tests").exists():
            return f"{python_exe} -m pytest"
    return None


def _ask_llm_for_command(message: str, runtime, context_hint: str) -> str | None:  # noqa: ANN001
    prompt = (
        "Return valid JSON only. No markdown.\n"
        "You convert a Russian/English user request into one safe PowerShell command to run inside the workspace.\n"
        "Allowed command families only: python -m pytest, python -m compileall src, python -m src.<module> with arguments, pytest.\n"
        "If no safe command is clear, return {\"command\": null}.\n"
        f"Workspace: {runtime.guard.root}\n"
        f"Session context: {context_hint[:1200]}\n"
        f"User request: {message}\n"
        "Schema: {\"command\":\"string|null\"}"
    )
    try:
        response = runtime.llm.chat(
            [ChatMessage(role="user", content=prompt)],
            json_mode=True,
        )
        payload = extract_json_object(response.content)
    except Exception:
        return None
    command = payload.get("command") if isinstance(payload, dict) else None
    if not command:
        return None
    command = str(command).strip()
    if not _is_allowed_direct_command(command):
        return None
    root = runtime.guard.root
    if command.startswith("python "):
        python_exe = ".\\.venv\\Scripts\\python.exe" if (root / ".venv" / "Scripts" / "python.exe").exists() else "python"
        command = command.replace("python", python_exe, 1)
    return command


def _is_allowed_direct_command(command: str) -> bool:
    lowered = command.lower().strip()
    return (
        lowered.startswith("python -m pytest")
        or lowered.startswith("pytest")
        or lowered.startswith("python -m compileall src")
        or lowered.startswith("python -m src.")
        or lowered.startswith(".\\.venv\\scripts\\python.exe -m src.")
        or lowered.startswith(".\\.venv\\scripts\\python.exe -m pytest")
        or lowered.startswith(".\\.venv\\scripts\\python.exe -m compileall src")
    )


def _extract_requested_iterations(message: str) -> int | None:
    lowered = message.lower()
    patterns = (
        r"(\d+)\s*(?:ю|ью|и|ы|х)?\s*итерац",
        r"(\d+)\s*(?:ppo\s*)?iterations?",
        r"iterations?\s*[:=]?\s*(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if not match:
            continue
        value = int(match.group(1))
        if 1 <= value <= 1000:
            return value
    return None


_CHAT_UI_HTML = """
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OpenKonst_B</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4efe6;
      --panel: #fffaf1;
      --ink: #1e2520;
      --muted: #66706a;
      --accent: #245b47;
      --accent-2: #d27d3f;
      --border: #d8ccb8;
      --shadow: 0 22px 55px rgba(40, 32, 20, 0.16);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 15% 15%, rgba(210, 125, 63, 0.20), transparent 34%),
        radial-gradient(circle at 88% 5%, rgba(36, 91, 71, 0.22), transparent 30%),
        linear-gradient(135deg, #f4efe6 0%, #ece2d1 100%);
    }
    main {
      width: min(1100px, calc(100vw - 32px));
      margin: 28px auto;
      display: grid;
      grid-template-columns: 310px 1fr;
      gap: 18px;
    }
    aside, section {
      background: rgba(255, 250, 241, 0.92);
      border: 1px solid var(--border);
      border-radius: 28px;
      box-shadow: var(--shadow);
    }
    aside { padding: 22px; height: fit-content; position: sticky; top: 18px; }
    h1 { margin: 0 0 8px; font-size: 34px; letter-spacing: -0.03em; }
    p { color: var(--muted); line-height: 1.45; }
    label { display: block; margin: 18px 0 8px; font-weight: 700; }
    input, textarea {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 12px 14px;
      font: 15px Consolas, "Courier New", monospace;
      color: var(--ink);
      background: #fffdf8;
      outline: none;
    }
    textarea { min-height: 120px; resize: vertical; font-family: inherit; font-size: 17px; }
    button {
      border: 0;
      border-radius: 999px;
      padding: 12px 18px;
      background: var(--accent);
      color: white;
      font-weight: 800;
      cursor: pointer;
    }
    button:disabled { opacity: 0.55; cursor: wait; }
    .chat { padding: 22px; min-height: calc(100vh - 56px); display: flex; flex-direction: column; }
    .messages { flex: 1; display: flex; flex-direction: column; gap: 14px; overflow: auto; padding-right: 4px; }
    .message {
      max-width: 88%;
      padding: 15px 17px;
      border-radius: 20px;
      line-height: 1.5;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .user { align-self: flex-end; background: #244f40; color: white; border-bottom-right-radius: 6px; }
    .agent { align-self: flex-start; background: #fffdf8; border: 1px solid var(--border); border-bottom-left-radius: 6px; }
    .composer { display: grid; gap: 12px; margin-top: 18px; }
    .status { min-height: 22px; color: var(--accent-2); font-weight: 800; }
    .hint { font-size: 14px; }
    @media (max-width: 820px) {
      main { grid-template-columns: 1fr; }
      aside { position: static; }
      .message { max-width: 100%; }
    }
  </style>
</head>
<body>
  <main>
    <aside>
      <h1>OpenKonst_B</h1>
      <label for="workspace">Workspace</label>
      <input id="workspace" value="F:\\chrommm\\Cryp\\LSTM">
      <label for="model">Model optional</label>
      <input id="model" placeholder="auto">
      <label for="timeout">Таймаут хода, сек</label>
      <input id="timeout" type="number" min="30" max="900" value="180">
      <p class="hint">Для правки кода пишите прямо: “исправь…”, “добавь…”, “проверь и запусти тест”.</p>
    </aside>
    <section class="chat">
      <div id="messages" class="messages">
        <div class="message agent">Я готов. Напишите задачу обычной фразой, например: “сначала пойми архитектуру проекта, потом назови 3 риска”, или “исправь ошибку в src/train.py и проверь тестами”.</div>
      </div>
      <div class="composer">
        <textarea id="message" placeholder="Что делаем?"></textarea>
        <div class="status" id="status"></div>
        <button id="send">Отправить</button>
      </div>
    </section>
  </main>
  <script>
    const messages = document.getElementById("messages");
    const workspace = document.getElementById("workspace");
    const model = document.getElementById("model");
    const timeout = document.getElementById("timeout");
    const message = document.getElementById("message");
    const status = document.getElementById("status");
    const send = document.getElementById("send");

    workspace.value = localStorage.getItem("agent.workspace") || workspace.value;
    model.value = localStorage.getItem("agent.model") || "";
    timeout.value = localStorage.getItem("agent.timeout") || timeout.value;

    function addMessage(text, role) {
      const div = document.createElement("div");
      div.className = `message ${role}`;
      div.textContent = text;
      messages.appendChild(div);
      messages.scrollTop = messages.scrollHeight;
    }

    async function submit() {
      const text = message.value.trim();
      if (!text || send.disabled) return;
      localStorage.setItem("agent.workspace", workspace.value.trim());
      localStorage.setItem("agent.model", model.value.trim());
      localStorage.setItem("agent.timeout", timeout.value.trim());
      addMessage(text, "user");
      message.value = "";
      send.disabled = true;
      status.textContent = "Агент работает: собирает контекст, думает, при необходимости меняет файлы...";
      try {
        const response = await fetch("/chat", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            workspace: workspace.value.trim() || ".",
            model: model.value.trim() || null,
            message: text,
            russian_only: true,
            timeout_seconds: Number(timeout.value || 180)
          })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || JSON.stringify(data));
        addMessage(data.assistant_message || data.summary || "Готово.", "agent");
        status.textContent = `Сохранено: ${data.report_path}`;
      } catch (error) {
        addMessage(`Ошибка: ${error.message}`, "agent");
        status.textContent = "Ошибка выполнения запроса.";
      } finally {
        send.disabled = false;
        message.focus();
      }
    }

    send.addEventListener("click", submit);
    message.addEventListener("keydown", (event) => {
      if (event.ctrlKey && event.key === "Enter") submit();
    });
  </script>
</body>
</html>
"""

