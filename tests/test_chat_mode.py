from pathlib import Path
import time
from types import SimpleNamespace

import coding_agent.cli.main as cli_main
from coding_agent.core.orchestrator import Orchestrator
from coding_agent.core.session import ChatHistoryEntry, ChatSessionState


def test_build_chat_goal_includes_session_context() -> None:
    state = ChatSessionState(
        current_goal="Исправить CLI",
        active_task="Исправить CLI",
        task_status="in_progress",
        last_summary="Исправлена команда doctor final-report.",
        last_changed_files=["src/coding_agent/cli/main.py"],
        pending_verifications=["Запустить pytest для CLI"],
        messages=[
            ChatHistoryEntry(role="user", content="Добавь doctor final-report"),
            ChatHistoryEntry(role="assistant", content="Команда добавлена."),
        ],
        turn_count=2,
    )

    prompt = cli_main._build_chat_goal("Теперь добавь нормальный чат", state, russian_only=True)

    assert "Текущая долгоживущая цель сессии: Исправить CLI" in prompt
    assert "Активная задача сессии: Исправить CLI" in prompt
    assert "Режим работы для этого запроса: feature." in prompt
    assert "Последние изменённые файлы: src/coding_agent/cli/main.py" in prompt
    assert "Незавершённые проверки: Запустить pytest для CLI" in prompt
    assert "Новый запрос пользователя:\nТеперь добавь нормальный чат" in prompt


def test_update_chat_state_tracks_files_and_pending_checks() -> None:
    state = ChatSessionState()
    payload = {
        "status": "completed",
        "summary": "Обновил чат и добавил state.",
        "changes": ["Изменён src/coding_agent/cli/main.py"],
        "verification": ["read_file: проверен src/coding_agent/core/session.py"],
        "findings": [
            {
                "title": "CLI loop",
                "severity": "medium",
                "status": "verified",
                "evidence": "read_file: {\"path\": \"src/coding_agent/cli/main.py\"}",
                "recommendation": "Запустить pytest для chat mode",
            }
        ],
        "next_steps": ["Запустить pytest для chat mode"],
        "assistant_message": "Обновил чат и добавил state.\n\nЧто сделал:\n- Изменён src/coding_agent/cli/main.py",
        "changed_files": ["src/coding_agent/cli/main.py", "src/coding_agent/core/session.py"],
    }

    updated = cli_main._update_chat_state(state, "Сделай чат лучше", payload)

    assert updated.current_goal == "Сделай чат лучше"
    assert updated.active_task == "Сделай чат лучше"
    assert updated.task_status == "in_progress"
    assert "src/coding_agent/cli/main.py" in updated.last_changed_files
    assert "src/coding_agent/core/session.py" in updated.last_changed_files
    assert "Запустить pytest для chat mode" in updated.pending_verifications
    assert updated.turn_count == 1


def test_format_chat_response_is_human_readable() -> None:
    state = ChatSessionState(
        last_changed_files=["src/coding_agent/cli/main.py"],
        active_task="Сделать чат устойчивее",
        task_status="in_progress",
    )
    payload = {
        "status": "completed",
        "summary": "Чатовый режим обновлён.",
        "changes": ["Изменён цикл chat"],
        "verification": ["pytest tests/test_chat_mode.py -q"],
        "findings": [
            {
                "title": "Состояние сессии стало устойчивее",
                "severity": "medium",
                "status": "verified",
                "evidence": "state file",
                "recommendation": "Проверить сценарий продолжения диалога",
            }
        ],
        "next_steps": ["Проверить живой сценарий chat"],
    }

    text = cli_main._format_chat_response(payload, state)

    assert "Чатовый режим обновлён." in text
    assert "Что сделал:" in text
    assert "Чем проверил:" in text
    assert "Последние затронутые файлы: src/coding_agent/cli/main.py" in text
    assert "Активная задача остаётся в работе" in text


def test_chat_state_path_derived_from_report_path() -> None:
    path = cli_main._chat_state_path(Path("F:/fake/workspace/artifacts/last_chat_report.json"))
    assert path == Path("F:/fake/workspace/artifacts/last_chat_report_state.json")


def test_maybe_handle_chat_message_waits_for_path() -> None:
    payload = cli_main._maybe_handle_chat_message("мне нужно что бы ты изучил проект, путь сейчас тебе пришлю")
    assert payload is not None
    assert payload["status"] == "completed"
    assert "Жду следующий ввод" in str(payload["summary"])


def test_maybe_handle_chat_message_accepts_matching_workspace_path() -> None:
    payload = cli_main._maybe_handle_chat_message(
        "F:\\chrommm\\OpenKonst_B",
        workspace_root=Path("F:/chrommm/OpenKonst_B"),
    )
    assert payload is not None
    assert "Путь подтвердил" in str(payload["summary"])


def test_maybe_handle_chat_message_rejects_workspace_switch_inside_chat() -> None:
    payload = cli_main._maybe_handle_chat_message(
        "F:\\chrommm\\Cryp\\LSTM",
        workspace_root=Path("F:/chrommm/OpenKonst_B"),
    )
    assert payload is not None
    assert "могу переключиться на него прямо внутри чата" in str(payload["summary"])
    assert "переключу текущую chat-сессию" in str(payload["next_steps"][0]).lower()


def test_extract_requested_workspace_from_command_line_text() -> None:
    path = cli_main._extract_requested_workspace(
        "F:\\chrommm\\OpenKonst_B\\.venv\\Scripts\\agent.exe chat --workspace F:\\chrommm\\Cryp\\LSTM"
    )
    assert path == Path("F:/chrommm/Cryp/LSTM")


def test_render_chat_report_is_text_not_json() -> None:
    payload = {
        "status": "completed",
        "summary": "Жду путь к проекту.",
        "changes": [],
        "verification": [],
        "findings": [],
        "next_steps": ["Прислать путь к проекту."],
        "assistant_message": "Жду путь к проекту.\n\nЧто ещё осталось:\n- Прислать путь к проекту.",
    }
    text = cli_main._render_chat_report(payload, "изучи проект")
    assert text.startswith("# Последний запрос")
    assert "## Ответ агента" in text
    assert '"status"' not in text


def test_normalize_chat_payload_keeps_assistant_message_and_changed_files() -> None:
    payload = cli_main._normalize_chat_payload(
        {
            "status": "completed",
            "summary": "ok",
            "changes": [],
            "verification": [],
            "findings": [],
            "next_steps": [],
            "changed_files": ["src/coding_agent/cli/main.py"],
            "assistant_message": "Готово.\nИзменён src/coding_agent/cli/main.py",
        },
        russian_only=True,
    )
    assert payload["changed_files"] == ["src/coding_agent/cli/main.py"]
    assert "Готово." in str(payload["assistant_message"])


def test_resolve_chat_request_uses_active_task_for_continue() -> None:
    state = ChatSessionState(
        active_task="Сначала пойми архитектуру проекта, потом назови 3 главных риска",
        pending_verifications=["Проверить security-модули"],
    )

    effective_goal, payload = cli_main._resolve_chat_request("продолжай", state)

    assert payload is None
    assert "Продолжай активную задачу" in effective_goal
    assert "Проверить security-модули" in effective_goal


def test_resolve_chat_request_reports_session_memory() -> None:
    state = ChatSessionState(
        active_task="Понять архитектуру LSTM",
        task_status="completed",
        last_summary="Проект использует RecurrentPPO.",
        last_changed_files=["src/train.py", "src/env.py"],
        pending_verifications=["Проверить evaluate.py"],
        messages=[ChatHistoryEntry(role="user", content="назови риски")],
    )

    _effective_goal, payload = cli_main._resolve_chat_request("что ты помнишь?", state)

    assert payload is not None
    assert "Понять архитектуру LSTM" in str(payload["summary"])
    assert "src/train.py" in str(payload["summary"])


def test_resolve_chat_request_ignores_pasted_answer_fragment() -> None:
    state = ChatSessionState(active_task="Понять архитектуру LSTM")

    effective_goal, payload = cli_main._resolve_chat_request("Попробуйте различные комбинации гиперпа", state)

    assert effective_goal == "Понять архитектуру LSTM"
    assert payload is not None
    assert "обрывок прошлого ответа" in str(payload["summary"])


def test_should_auto_continue_chat_task_for_partial_completed_result() -> None:
    payload = {
        "status": "completed",
        "summary": "Сделан первый проход по архитектуре.",
        "changes": ["Прочитаны ключевые файлы"],
        "verification": [],
        "findings": [],
        "next_steps": ["Проверить security-модули", "Уточнить 3 риска"],
    }

    assert cli_main._should_auto_continue_chat_task(
        payload,
        "сначала пойми архитектуру проекта, потом назови 3 главных риска",
    )


def test_describe_action_for_chat_is_human_readable() -> None:
    text = Orchestrator._describe_action_for_chat("read_file", {"path": "src/coding_agent/core/orchestrator.py"})
    assert "Читаю файл" in text
    assert "src/coding_agent/core/orchestrator.py" in text


def test_runtime_uses_agent_config_when_workspace_has_no_local_config(monkeypatch) -> None:
    captured: dict[str, Path] = {}

    def fake_build_runtime(workspace, config_path=None, policy_path=None, model_override=None):
        captured["workspace"] = workspace
        captured["config_path"] = config_path
        captured["policy_path"] = policy_path
        return SimpleNamespace(config=SimpleNamespace(log_level="INFO"))

    monkeypatch.setattr(cli_main, "build_runtime", fake_build_runtime)
    monkeypatch.setattr(cli_main, "configure_logging", lambda *_args, **_kwargs: None)

    cli_main._runtime("F:/chrommm/Cryp/LSTM")

    assert captured["workspace"] == Path("F:/chrommm/Cryp/LSTM").resolve()
    assert captured["config_path"] == Path("F:/chrommm/OpenKonst_B/config/default.yaml").resolve()
    assert captured["policy_path"] == Path("F:/chrommm/OpenKonst_B/config/policy.yaml").resolve()


def test_select_default_chat_model_prefers_fast_installed_coding_model(monkeypatch) -> None:
    monkeypatch.setattr(
        cli_main,
        "_installed_ollama_models",
        lambda: {"qwen3.5:9b", "deepseek-coder:6.7b", "qwen2.5-coder:14b"},
    )

    selected = cli_main._select_default_chat_model("F:/chrommm/Cryp/LSTM")

    assert selected == "qwen2.5-coder:14b"


def test_chat_task_strategy_detects_analysis_mode() -> None:
    strategy = cli_main._chat_task_strategy("сначала пойми архитектуру проекта, потом назови 3 главных риска")
    assert strategy["mode"] == "analysis"
    assert "evidence" in strategy["instruction"]


def test_chat_task_strategy_detects_fix_mode() -> None:
    strategy = cli_main._chat_task_strategy("исправь баг в cli и проверь тестами")
    assert strategy["mode"] == "fix"
    assert "исправлено" in strategy["instruction"]


def test_chat_task_strategy_detects_change_and_run_as_work_mode() -> None:
    strategy = cli_main._chat_task_strategy("измени гиперпараметры и запусти тестовое обучение")
    assert strategy["mode"] in {"fix", "feature"}


def test_chat_task_strategy_detects_refactor_mode() -> None:
    strategy = cli_main._chat_task_strategy("сделай рефакторинг chat orchestrator")
    assert strategy["mode"] == "refactor"
    assert "не сломаны" in strategy["instruction"]


def test_run_chat_with_heartbeat_returns_result(monkeypatch) -> None:
    events: list[str] = []

    class _FakeOrchestrator:
        @staticmethod
        def run_chat(goal, progress_callback=None):
            if progress_callback:
                progress_callback("Читаю файл: src/main.py")
            time.sleep(0.05)
            return SimpleNamespace(status="completed", summary="ok")

    runtime = SimpleNamespace(orchestrator=_FakeOrchestrator())
    monkeypatch.setattr(cli_main.typer, "echo", lambda message: events.append(str(message)))

    result = cli_main._run_chat_with_heartbeat(
        runtime,
        "исправь баг",
        progress_callback=lambda message: events.append(f"[agent] {message}"),
        heartbeat_seconds=0.01,
    )

    assert result.summary == "ok"
    assert any("Всё ещё работаю" in item or "Читаю файл" in item for item in events)
